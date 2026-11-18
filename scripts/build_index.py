"""Build the BGE embedding index for RQ4.

Build and run a small retrieval demonstration:
    python scripts/build_index.py

Load the saved index from another script with ``load_index()``, then call the
module-level ``get_distractor_candidates(window_id, n=10)`` or
``retrieve(query, k=10)`` functions.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sample_windows import has_concrete_fact, page_windows

QUERY_PREFIX_DEFAULT = "Represent this sentence for searching relevant passages: "
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "embeddings.yaml"
DEFAULT_PAGES = PROJECT_ROOT / "data" / "processed" / "pages.csv"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks_clean.csv"
DEFAULT_GOLD_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows_screened.jsonl"
DEFAULT_OUTDIR = PROJECT_ROOT / "data" / "processed" / "index"
EXCLUDED_CANDIDATE_TYPES = {"api_docs", "search_page"}

_INDEX: "RetrievalIndex | None" = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _dataset_title(window: dict[str, Any]) -> str:
    """Extract an explicit title/name only; do not guess from surrounding prose."""
    text = str(window.get("text", ""))
    patterns = (
        r"(?im)^\s*(?:dataset\s+)?(?:title|name)\s*[:|]\s*([^\n|]+)",
        r"(?im)^\s*#\s+(.+?)\s*$",
        r'"(?:title|name)"\s*:\s*"([^"\n]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            title = _normalized_text(match.group(1)).strip(" .#*_-`")
            if len(title) >= 8:
                return title
    return ""


def _same_dataset(gold: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if gold.get("page_type") != "dataset_record" or candidate.get("page_type") != "dataset_record":
        return False
    gold_title = _dataset_title(gold)
    candidate_title = _dataset_title(candidate)
    return bool(gold_title and candidate_title and gold_title == candidate_title)


def _text_similarity(left: str, right: str) -> float:
    left_norm, right_norm = _normalized_text(left), _normalized_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def _url_template(url: str) -> str:
    parsed = urlparse(url)
    first_segment = next((part for part in parsed.path.split("/") if part), "")
    return f"{parsed.netloc.casefold()}/{first_segment}"


def _domain(url: str) -> str:
    return urlparse(url).netloc.casefold()


def _short_candidate(frame: pd.DataFrame) -> dict[str, Any] | None:
    """Use the same 3–5 chunk cut for eligible pages with no 900-char option."""
    rows = frame.sort_values("chunk_number", kind="stable")[
        ["id", "content"]
    ].to_dict("records")
    options = []
    for start in range(len(rows)):
        for count in range(3, 6):
            part = rows[start : start + count]
            if len(part) != count:
                continue
            text = "".join(str(row["content"]) for row in part if str(row["content"]).strip())
            if 600 <= len(text) < 900 and has_concrete_fact(text):
                options.append({
                    "chunk_ids": [str(row["id"]) for row in part],
                    "text": text,
                    "length": len(text),
                })
    if not options:
        return None
    return min(options, key=lambda item: (900 - item["length"], item["chunk_ids"][0]))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not config.get("model_name"):
        raise ValueError(f"Embedding model_name is missing from {path}")
    return config


def _load_encoder(
    model_name: str,
    revision: str,
    device: str | None = None,
    local_files_only: bool = False,
):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    model_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    kwargs: dict[str, Any] = {"revision": revision}
    if device:
        kwargs["device"] = device
    # A local snapshot avoids repeated HEAD requests for optional files (such
    # as PEFT adapter_config.json) when reopening an otherwise cached model.
    return SentenceTransformer(model_path, **kwargs)


def _encode(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot encode an empty text collection")
    matrix = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(matrix, dtype=np.float32)


def _prepare_records(
    pages_path: Path,
    chunks_path: Path,
    gold_windows_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pages = pd.read_csv(pages_path, dtype=str, keep_default_na=False)
    chunks = pd.read_csv(chunks_path, dtype=str, keep_default_na=False)
    if not {"url", "id", "content"}.issubset(chunks.columns):
        raise ValueError("chunks_clean.csv must contain url, id, and content")
    if not {"url", "domain", "page_type", "eligible"}.issubset(pages.columns):
        raise ValueError("pages.csv must contain url, domain, page_type, and eligible")

    page_info = pages.set_index("url").to_dict("index")
    chunk_records = []
    for row in chunks.to_dict("records"):
        info = page_info.get(row["url"], {})
        chunk_records.append({
            "chunk_id": str(row["id"]),
            "url": row["url"],
            "domain": info.get("domain") or _domain(row["url"]),
            "page_type": info.get("page_type", ""),
            "text": row["content"],
        })

    eligible = pages[
        pages["eligible"].str.casefold().eq("true")
        & ~pages["page_type"].isin(EXCLUDED_CANDIDATE_TYPES)
    ].sort_values("url", kind="stable")
    all_eligible_count = int(pages["eligible"].str.casefold().eq("true").sum())
    excluded_candidate_pages = pages[
        pages["eligible"].str.casefold().eq("true")
        & pages["page_type"].isin(EXCLUDED_CANDIDATE_TYPES)
    ]
    chunks_by_url = {url: frame for url, frame in chunks.groupby("url", sort=False)}
    window_records = []
    for row in eligible.itertuples(index=False):
        frame = chunks_by_url.get(row.url)
        if frame is None:
            continue
        options = page_windows(frame)
        selected = options[0] if options else _short_candidate(frame)
        if selected is None:
            continue
        window_records.append({
            "window_id": f"CAND-{len(window_records) + 1:06d}",
            "url": row.url,
            "domain": row.domain or _domain(row.url),
            "page_type": row.page_type,
            "url_template": _url_template(row.url),
            "chunk_ids": selected["chunk_ids"],
            "text": selected["text"],
            "length_chars": selected["length"],
            "length_rule": "900–1,500 chars" if options else "600–899 chars (short fallback)",
        })

    gold_windows = _read_jsonl(gold_windows_path)
    for window in gold_windows:
        window.setdefault("domain", _domain(window["url"]))
        window["url_template"] = _url_template(window["url"])
    candidate_urls = {window["url"] for window in window_records}
    coverage = {
        "eligible_pages_total": all_eligible_count,
        "eligible_pages_excluded_by_type": {
            key: int(value) for key, value in excluded_candidate_pages["page_type"].value_counts().items()
        },
        "eligible_pages_considered_for_candidate_windows": len(eligible),
        "eligible_pages_without_a_3_to_5_chunk_window": int(
            eligible["url"].isin(candidate_urls).eq(False).sum()
        ),
    }
    return chunk_records, window_records, gold_windows, coverage


class RetrievalIndex:
    """In-memory BGE index with cosine-scored chunk and window search."""

    def __init__(
        self,
        encoder: Any,
        config: dict[str, Any],
        chunk_records: list[dict[str, Any]],
        chunk_embeddings: np.ndarray,
        window_records: list[dict[str, Any]],
        window_embeddings: np.ndarray,
        gold_windows: list[dict[str, Any]],
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.chunk_records = chunk_records
        self.chunk_embeddings = chunk_embeddings
        self.window_records = window_records
        self.window_embeddings = window_embeddings
        self.gold_by_id = {str(window["window_id"]): window for window in gold_windows}
        self.window_by_id = {str(window["window_id"]): window for window in window_records}
        self.query_prefix = str(config.get("query_prefix", QUERY_PREFIX_DEFAULT))
        self.batch_size = int(config.get("batch_size", 64))

    def _query_embedding(self, query: str) -> np.ndarray:
        embedded = self.encoder.encode(
            [self.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedded[0], dtype=np.float32)

    def retrieve(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        query_vector = self._query_embedding(query)
        scores = self.chunk_embeddings @ query_vector
        indices = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
        return [
            {
                "chunk_id": self.chunk_records[int(index)]["chunk_id"],
                "url": self.chunk_records[int(index)]["url"],
                "similarity": float(scores[index]),
                "text": self.chunk_records[int(index)]["text"],
            }
            for index in indices
        ]

    def get_distractor_candidates(self, window_id: str, n: int = 10) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        gold = self.gold_by_id.get(str(window_id)) or self.window_by_id.get(str(window_id))
        if gold is None:
            raise KeyError(f"Unknown window_id: {window_id}")
        query_vector = self._query_embedding(str(gold["text"]))
        scores = self.window_embeddings @ query_vector
        gold_url = gold["url"]
        gold_template = gold.get("url_template") or _url_template(gold_url)
        gold_domain = gold.get("domain") or _domain(gold_url)
        kept: list[tuple[int, float, int]] = []

        for index, candidate in enumerate(self.window_records):
            if candidate["url"] == gold_url:
                continue
            if _same_dataset(gold, candidate):
                continue
            similarity_text = _text_similarity(str(gold["text"]), str(candidate["text"]))
            if similarity_text > 0.7:
                continue
            candidate_template = candidate.get("url_template") or _url_template(candidate["url"])
            candidate_domain = candidate.get("domain") or _domain(candidate["url"])
            priority = 0 if candidate_template == gold_template else 1 if candidate_domain == gold_domain else 2
            kept.append((priority, float(scores[index]), index))

        kept.sort(key=lambda item: (item[0], -item[1], self.window_records[item[2]]["url"]))
        return [
            {
                "window_id": self.window_records[index]["window_id"],
                "url": self.window_records[index]["url"],
                "domain": self.window_records[index].get("domain") or _domain(self.window_records[index]["url"]),
                "page_type": self.window_records[index]["page_type"],
                "chunk_ids": self.window_records[index].get("chunk_ids", []),
                "text": self.window_records[index]["text"],
                "similarity": score,
                "url_template": self.window_records[index].get("url_template") or _url_template(self.window_records[index]["url"]),
            }
            for _, score, index in kept[:n]
        ]


def build_index(
    pages_path: Path = DEFAULT_PAGES,
    chunks_path: Path = DEFAULT_CHUNKS,
    gold_windows_path: Path = DEFAULT_GOLD_WINDOWS,
    outdir: Path = DEFAULT_OUTDIR,
    config_path: Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> RetrievalIndex:
    """Encode all cleaned chunks and one sample-style window per eligible page."""
    config = _load_config(config_path)
    model_name = str(config["model_name"])
    revision = str(config.get("revision", "main"))
    batch_size = int(config.get("batch_size", 64))
    chunk_records, window_records, gold_windows, coverage = _prepare_records(
        pages_path, chunks_path, gold_windows_path
    )
    print(f"Encoding {len(chunk_records):,} cleaned chunks with {model_name}...", flush=True)
    encoder = _load_encoder(model_name, revision, device)
    chunk_embeddings = _encode(encoder, [row["text"] for row in chunk_records], batch_size)
    print(f"Encoding {len(window_records):,} eligible-page candidate windows...", flush=True)
    window_embeddings = _encode(encoder, [row["text"] for row in window_records], batch_size)

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "chunk_embeddings.npy", chunk_embeddings)
    np.save(outdir / "window_embeddings.npy", window_embeddings)
    _write_jsonl(outdir / "chunks.jsonl", chunk_records)
    _write_jsonl(outdir / "candidate_windows.jsonl", window_records)
    _write_jsonl(outdir / "gold_windows.jsonl", gold_windows)

    first_module = encoder._first_module() if hasattr(encoder, "_first_module") else None
    auto_model = getattr(first_module, "auto_model", None)
    model_config = getattr(auto_model, "config", None)
    model_commit = getattr(model_config, "_commit_hash", None)
    try:
        import sentence_transformers
        sentence_transformers_version = sentence_transformers.__version__
    except Exception:
        sentence_transformers_version = "unknown"
    metadata = {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_model_commit": model_commit,
        "sentence_transformers_version": sentence_transformers_version,
        "embedding_dimension": int(chunk_embeddings.shape[1]),
        "normalized_embeddings": True,
        "similarity": "cosine (dot product over normalized vectors)",
        "query_prefix": str(config.get("query_prefix", QUERY_PREFIX_DEFAULT)),
        "chunk_count": len(chunk_records),
        "candidate_window_count": len(window_records),
        "gold_window_count": len(gold_windows),
        **coverage,
        "excluded_candidate_page_types": sorted(EXCLUDED_CANDIDATE_TYPES),
        "candidate_window_rule": "first sample_windows.page_windows option (3–5 consecutive chunks; 900–1,500 chars), otherwise nearest 900-char 3–5 chunk fallback if 600–899 chars",
        "candidate_page_filter": "eligible pages except api_docs and search_page",
        "chunk_source_sha256": _sha256(chunks_path),
        "pages_source_sha256": _sha256(pages_path),
        "gold_windows_source_sha256": _sha256(gold_windows_path),
    }
    (outdir / "index_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    index = RetrievalIndex(
        encoder, config, chunk_records, chunk_embeddings,
        window_records, window_embeddings, gold_windows,
    )
    global _INDEX
    _INDEX = index
    print(f"Saved index under {outdir}")
    return index


def load_index(
    index_dir: Path = DEFAULT_OUTDIR,
    config_path: Path = DEFAULT_CONFIG,
    device: str | None = None,
) -> RetrievalIndex:
    """Load the saved vectors and model so module-level query helpers work."""
    config = _load_config(config_path)
    metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    model_name = metadata["model_name"]
    # Reopen the exact Hub commit recorded during construction; ``main`` moves.
    revision = metadata.get("resolved_model_commit") or metadata.get(
        "requested_revision", config.get("revision", "main")
    )
    encoder = _load_encoder(model_name, revision, device, local_files_only=True)
    index = RetrievalIndex(
        encoder=encoder,
        config=config,
        chunk_records=_read_jsonl(index_dir / "chunks.jsonl"),
        chunk_embeddings=np.load(index_dir / "chunk_embeddings.npy", mmap_mode="r"),
        window_records=_read_jsonl(index_dir / "candidate_windows.jsonl"),
        window_embeddings=np.load(index_dir / "window_embeddings.npy", mmap_mode="r"),
        gold_windows=_read_jsonl(index_dir / "gold_windows.jsonl"),
    )
    global _INDEX
    _INDEX = index
    return index


def get_distractor_candidates(window_id: str, n: int = 10) -> list[dict[str, Any]]:
    if _INDEX is None:
        raise RuntimeError("Index is not loaded; call load_index() or build_index() first")
    return _INDEX.get_distractor_candidates(window_id, n=n)


def retrieve(query: str, k: int = 10) -> list[dict[str, Any]]:
    if _INDEX is None:
        raise RuntimeError("Index is not loaded; call load_index() or build_index() first")
    return _INDEX.retrieve(query, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--gold-windows", type=Path, default=DEFAULT_GOLD_WINDOWS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="sentence-transformers device, e.g. cpu, mps, cuda")
    args = parser.parse_args()

    index = build_index(
        pages_path=args.pages,
        chunks_path=args.chunks,
        gold_windows_path=args.gold_windows,
        outdir=args.outdir,
        config_path=args.config,
        device=args.device,
    )
    samples = index.gold_by_id
    sample_ids = list(samples)
    if sample_ids:
        chosen_indices = sorted({0, len(sample_ids) // 2, len(sample_ids) - 1})
        print("\nDistractor candidate smoke checks:")
        for position in chosen_indices:
            window_id = sample_ids[position]
            print(f"\n{window_id} ({samples[window_id]['url']})")
            for candidate in index.get_distractor_candidates(window_id, n=10):
                excerpt = re.sub(r"\s+", " ", candidate["text"]).strip()[:180]
                print(
                    f"  {candidate['domain']} | similarity={candidate['similarity']:.4f} | {excerpt}"
                )

    test_query = "Which organisation coordinates the Argo Program as part of the global ocean observing system?"
    print(f"\nRetrieve smoke query: {test_query}")
    for result in index.retrieve(test_query, k=10):
        excerpt = re.sub(r"\s+", " ", result["text"]).strip()[:180]
        print(f"  {result['chunk_id']} | {result['url']} | {result['similarity']:.4f} | {excerpt}")


if __name__ == "__main__":
    main()
