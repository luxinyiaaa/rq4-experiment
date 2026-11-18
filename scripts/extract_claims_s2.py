"""Extract grounded, chunk-addressed facts from eligible S2 knowledge-base pages."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from rq4.llm import call_llm
from scripts.build_kb_index import OUT as INDEX_DIR, read_jsonl, write_jsonl

PROMPT_PATH = ROOT / "prompts" / "extract_claims.md"
DEFAULT_OUT = INDEX_DIR / "page_claims.jsonl"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pages_and_chunks() -> list[dict[str, Any]]:
    pages = read_jsonl(INDEX_DIR / "indexed_pages.jsonl")
    chunks = pd.read_csv(ROOT / "data/processed/chunks_clean.csv", dtype=str, keep_default_na=False)
    chunk_map = {str(r["id"]): {"chunk_id": str(r["id"]), "text": str(r["content"])} for r in chunks.to_dict("records")}
    result = []
    for page in pages:
        page_chunks = [chunk_map[cid] for cid in page.get("chunk_ids", []) if cid in chunk_map]
        if page.get("page_type") == "news_events" or not page_chunks:
            continue
        result.append({"url": page["url"], "title": page.get("record_name") or page.get("title", ""), "chunks": page_chunks})
    return result


def _mock_page(page: dict[str, Any]) -> dict[str, Any]:
    claims = []
    for chunk in page["chunks"]:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"].strip()):
            sentence = sentence.strip()
            if len(sentence) < 28 or not re.search(r"[A-Z][a-z]+|\d|\b(?:located|based|operated|produced|part of|member)\b", sentence):
                continue
            claims.append({"claim": sentence[:240], "quote": sentence, "chunk_id": chunk["chunk_id"],
                           "subject": page["title"], "property": "page fact", "value": "",
                           "entities": re.findall(r"\b[A-Z][A-Za-z0-9&-]+(?:\s+[A-Z][A-Za-z0-9&-]+){0,3}", sentence)[:6]})
            if len(claims) == 8:
                break
        if len(claims) == 8:
            break
    return {"url": page["url"], "claims": claims}


def _valid_rows(page_batch: list[dict[str, Any]], parsed: Any) -> list[dict[str, Any]]:
    source_map = {page["url"]: {chunk["chunk_id"]: chunk["text"] for chunk in page["chunks"]} for page in page_batch}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("pages"), list):
        return [{"url": page["url"], "claims": [], "extraction_error": "Invalid JSON schema"} for page in page_batch]
    by_url = {str(row.get("url", "")): row for row in parsed["pages"] if isinstance(row, dict)}
    output = []
    for page in page_batch:
        row = by_url.get(page["url"], {})
        claims = []
        for item in row.get("claims", [])[:8] if isinstance(row.get("claims", []), list) else []:
            if not isinstance(item, dict):
                continue
            chunk_id, quote = str(item.get("chunk_id", "")), str(item.get("quote", ""))
            if not quote or chunk_id not in source_map[page["url"]] or quote not in source_map[page["url"]][chunk_id]:
                continue
            entities = [str(e).strip() for e in item.get("entities", []) if str(e).strip()]
            claims.append({"claim": str(item.get("claim", "")).strip(), "quote": quote, "chunk_id": chunk_id,
                           "subject": str(item.get("subject", "")).strip(),
                           "property": str(item.get("property", "")).strip(),
                           "value": str(item.get("value", "")).strip(), "entities": entities})
        output.append({"url": page["url"], "title": page["title"], "claims": claims})
    return output


def extract_claims(*, mock: bool = False, batch_size: int = 4, max_workers: int = 4, output: Path = DEFAULT_OUT) -> list[dict[str, Any]]:
    if not mock:
        load_dotenv(ROOT / ".env", override=True)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    pages = _pages_and_chunks()
    batches = [(offset, pages[offset:offset + batch_size]) for offset in range(0, len(pages), batch_size)]

    def process_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if mock:
            parsed = {"pages": [_mock_page(page) for page in batch]}
        else:
            payload = [{"url": page["url"], "chunks": page["chunks"]} for page in batch]
            try:
                result = call_llm("judge", [{"role": "system", "content": prompt},
                    {"role": "user", "content": "Extract the two to four most useful factual claims per page (never more than eight). Preserve each exact quote and chunk ID.\n" + json.dumps(payload, ensure_ascii=False)}],
                    json_mode=True, temperature=0.0, prompt_sha256=_hash(prompt), log_messages=True,
                    cache_dir=ROOT / "data/processed/s2/cache/llm", log_dir=ROOT / "data/processed/s2/logs", timeout=120.0)
                parsed = result.get("json") or {}
            except Exception as exc:
                return [{"url": page["url"], "title": page["title"], "claims": [],
                         "extraction_error": f"{type(exc).__name__}: {exc}"} for page in batch]
        return _valid_rows(batch, parsed)

    completed: dict[int, list[dict[str, Any]]] = {}
    max_workers = 1 if mock else max(1, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_batch, batch): offset for offset, batch in batches}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
            rows_so_far = [row for offset in sorted(completed) for row in completed[offset]]
            write_jsonl(output, rows_so_far)
            print(json.dumps({"batches_completed":len(completed),"batches_total":len(batches),
                              "pages_saved":len(rows_so_far)},separators=(",",":")),flush=True)
    rows = [row for offset in sorted(completed) for row in completed[offset]]
    write_jsonl(output, rows)
    print(json.dumps({"pages": len(rows), "pages_with_claims": sum(bool(x["claims"]) for x in rows),
                      "claims": sum(len(x["claims"]) for x in rows), "mock": mock, "output": str(output)}, indent=2))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    extract_claims(mock=args.mock, batch_size=args.batch_size, max_workers=args.max_workers, output=args.out)


if __name__ == "__main__":
    main()
