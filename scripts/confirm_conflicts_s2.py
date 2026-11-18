"""Use the configured judge to confirm same-object conflict candidates for S2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from rq4.llm import call_llm
from scripts.build_kb_index import OUT as INDEX_DIR, read_jsonl, write_jsonl

PROMPT_PATH = ROOT / "prompts" / "judge_conflict_check.md"
OUTPUT = INDEX_DIR / "conflict_confirmations.jsonl"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def confirmation_message(candidate: dict[str, Any]) -> str:
    return (
        f"Candidate attribute: {candidate.get('attribute', '')}\n"
        f"Candidate value A: {candidate.get('value_a', '')}\n"
        f"Page A: {candidate.get('title_a', candidate.get('object_a', ''))} ({candidate.get('url_a', '')})\n"
        f"Exact excerpt A:\n{candidate.get('quote_a', '')}\n\n"
        f"Candidate value B: {candidate.get('value_b', '')}\n"
        f"Page B: {candidate.get('title_b', candidate.get('object_b', ''))} ({candidate.get('url_b', '')})\n"
        f"Exact excerpt B:\n{candidate.get('quote_b', '')}"
    )


def confirm_one(candidate: dict[str, Any], system: str, *, mock: bool, output_root: Path) -> dict[str, Any]:
    if mock:
        result = {"json": {"confirmed": False, "same_object": False, "same_attribute": True,
                           "values_differ": True, "reason": "Mock confirmation rejects this candidate."},
                  "model": "mock-judge", "cache_hit": False}
    else:
        result = call_llm("judge", [{"role": "system", "content": system},
            {"role": "user", "content": confirmation_message(candidate)}],
            json_mode=True, temperature=0.0, prompt_sha256=prompt_sha256(system),
            log_messages=True, cache_dir=output_root / "cache" / "llm", log_dir=output_root / "logs")
    judgement = result.get("json") or {}
    confirmed = all(judgement.get(key) is True for key in ("confirmed", "same_object", "same_attribute", "values_differ"))
    return {"conflict_id": candidate["conflict_id"], "source_kind": candidate["source_kind"],
            "confirmed": confirmed, "judge_result": judgement, "model": result.get("model"),
            "cache_hit": result.get("cache_hit"), "judge_prompt_sha256": prompt_sha256(system)}


def run(*, mock: bool = False, workers: int = 4, output_path: Path = OUTPUT) -> list[dict[str, Any]]:
    if not mock:
        load_dotenv(ROOT / ".env", override=True)
    system = PROMPT_PATH.read_text(encoding="utf-8").split("## User template")[0].strip()
    candidates = [row for row in read_jsonl(INDEX_DIR / "conflict_candidates.jsonl")
                  if row.get("source_kind") == "same_object_attribute_candidate"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {row["conflict_id"]: row for row in read_jsonl(output_path)}
    pending = [row for row in candidates if row["conflict_id"] not in cached]
    results = dict(cached)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(confirm_one, row, system, mock=mock, output_root=output_path.parent) for row in pending]
        for future in as_completed(futures):
            row = future.result()
            results[row["conflict_id"]] = row
    ordered = [results[row["conflict_id"]] for row in candidates]
    write_jsonl(output_path, ordered)
    if output_path == OUTPUT:
        metadata_path = INDEX_DIR / "s2_index_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = {"candidates": len(ordered), "confirmed": sum(row["confirmed"] for row in ordered),
                   "rejected": sum(not row["confirmed"] for row in ordered)}
        metadata["same_object_conflict_confirmation"] = summary
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path = INDEX_DIR / "s2_index_summary.md"
        text = summary_path.read_text(encoding="utf-8")
        text = text.split("## Same-object conflict confirmation")[0].rstrip()
        text += ("\n\n## Same-object conflict confirmation\n\n"
                 f"- Candidates checked by judge: {summary['candidates']}\n"
                 f"- Confirmed: {summary['confirmed']}\n"
                 f"- Rejected: {summary['rejected']}\n")
        summary_path.write_text(text, encoding="utf-8")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = run(mock=args.mock, workers=args.workers, output_path=args.output)
    print(json.dumps({"candidates": len(rows), "confirmed": sum(row["confirmed"] for row in rows),
                      "rejected": sum(not row["confirmed"] for row in rows),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
