"""Sample pages and build one evidence window per page.

Example:
    python scripts/sample_windows.py \
      --pages data/processed/pages.csv \
      --chunks data/processed/chunks_clean.csv \
      --out data/processed/windows.jsonl

The sample is deterministic for a given seed. It enforces the page-type quotas
and per-domain cap, then selects a 3–5 consecutive-chunk window nearest 1,200
characters (preferring the 900–1,500 character range) with a simple factuality
signal: a digit or a non-empty ``field | value`` pattern.
"""
import argparse
import json
import random
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd


QUOTAS = {"dataset_record": 25, "about_science": 20, "vocabulary": 5}
MIN_CHUNKS = 3
MAX_CHUNKS = 5
MIN_WINDOW_CHARS = 900
MAX_WINDOW_CHARS = 1500
TARGET_WINDOW_CHARS = 1200
FACT_PATTERN = re.compile(r"\d|[^|\n]+\|[^|\n]+")


def has_concrete_fact(text: str) -> bool:
    return bool(FACT_PATTERN.search(text))


def page_windows(frame: pd.DataFrame) -> list[dict]:
    """Return eligible consecutive windows, best length match first."""
    frame = frame.sort_values("chunk_number", kind="stable")
    rows = frame[["id", "content"]].to_dict("records")
    options = []
    for start in range(len(rows)):
        for count in range(MIN_CHUNKS, MAX_CHUNKS + 1):
            part = rows[start : start + count]
            if len(part) != count:
                continue
            # The exporter splits a page's text into fixed-size chunks; concatenate
            # directly to avoid inserting spaces/newlines in the middle of words.
            text = "".join(str(row["content"]) for row in part if str(row["content"]).strip())
            if not (MIN_WINDOW_CHARS <= len(text) <= MAX_WINDOW_CHARS):
                continue
            if not has_concrete_fact(text):
                continue
            options.append({
                "chunk_ids": [str(row["id"]) for row in part],
                "text": text,
                "length": len(text),
            })
    options.sort(key=lambda item: (abs(item["length"] - TARGET_WINDOW_CHARS), item["chunk_ids"][0]))
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", required=True, type=Path, help="cleaned pages.csv")
    parser.add_argument("--chunks", required=True, type=Path, help="cleaned chunks_clean.csv")
    parser.add_argument("--out", required=True, type=Path, help="output windows.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-domain", type=int, default=5,
                        help="per-domain cap for about_science and vocabulary (default: 5)")
    parser.add_argument("--dataset-record-max-per-domain", type=int, default=8,
                        help="per-domain cap for dataset_record (default: 8)")
    args = parser.parse_args()

    pages = pd.read_csv(args.pages, dtype={"url": str, "domain": str, "page_type": str})
    chunks = pd.read_csv(args.chunks, dtype={"url": str, "id": str, "content": str}, keep_default_na=False)
    eligible = pages[(pages["eligible"].astype(str).str.lower() == "true") & pages["page_type"].isin(QUOTAS)]

    chunks_by_url = {url: frame for url, frame in chunks.groupby("url", sort=False)}
    candidates = {page_type: [] for page_type in QUOTAS}
    page_rows = {row.url: row for row in eligible.itertuples(index=False)}
    for page_type in QUOTAS:
        urls = eligible.loc[eligible["page_type"] == page_type, "url"].tolist()
        for url in urls:
            frame = chunks_by_url.get(url)
            if frame is None:
                continue
            options = page_windows(frame)
            if options:
                row = page_rows[url]
                candidates[page_type].append({
                    "url": url,
                    "domain": row.domain or urlparse(url).netloc,
                    "page_type": page_type,
                    "options": options,
                })

    rng = random.Random(args.seed)
    for values in candidates.values():
        rng.shuffle(values)

    selected = []
    chosen_counts = Counter()
    domains = Counter()
    # Interleave page types so an early type cannot consume the domain cap first.
    while any(chosen_counts[t] < q for t, q in QUOTAS.items()):
        progressed = False
        for page_type, quota in QUOTAS.items():
            if chosen_counts[page_type] >= quota:
                continue
            pool = candidates[page_type]
            domain_cap = (args.dataset_record_max_per_domain
                          if page_type == "dataset_record" else args.max_per_domain)
            # Caps are per page type and domain; each selected page contributes
            # to its category's site count independently.
            picked_domain_count = Counter(
                (item["page_type"], item["domain"]) for item in selected
            )
            pick_index = next((i for i, item in enumerate(pool)
                               if picked_domain_count[(page_type, item["domain"])] < domain_cap), None)
            if pick_index is None:
                continue
            item = pool.pop(pick_index)
            window = item["options"][0]
            selected.append({
                "window_id": f"W{len(selected) + 1:03d}",
                "url": item["url"],
                "domain": item["domain"],
                "page_type": item["page_type"],
                "chunk_ids": window["chunk_ids"],
                "text": window["text"],
            })
            chosen_counts[page_type] += 1
            domains[item["domain"]] += 1
            progressed = True
        if not progressed:
            missing = {t: q - chosen_counts[t] for t, q in QUOTAS.items() if chosen_counts[t] < q}
            available_by_domain = {
                page_type: dict(sorted(Counter(item["domain"] for item in candidates[page_type]).items()))
                for page_type in QUOTAS
            }
            raise RuntimeError(
                "Could not fill requested sample quotas under the per-domain cap; "
                f"missing: {missing}; available candidates by type/domain: {available_by_domain}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for item in selected:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    lengths = [len(item["text"]) for item in selected]
    length_series = pd.Series(lengths)
    print("Page counts by type:")
    for page_type, quota in QUOTAS.items():
        print(f"  {page_type}: {chosen_counts[page_type]} / {quota}")
    print("Pages per domain (by type):")
    by_domain_type = Counter((item["domain"], item["page_type"]) for item in selected)
    for domain, count in sorted(domains.items()):
        breakdown = ", ".join(
            f"{page_type}={by_domain_type[(domain, page_type)]}"
            for page_type in QUOTAS if by_domain_type[(domain, page_type)]
        )
        print(f"  {domain}: {count} total ({breakdown})")
    print("Window text length (characters):")
    print("  " + json.dumps({
        "min": int(length_series.min()),
        "p25": float(length_series.quantile(0.25)),
        "median": float(length_series.median()),
        "p75": float(length_series.quantile(0.75)),
        "max": int(length_series.max()),
    }))
    print(f"Wrote {len(selected)} windows to {args.out}")
    print("Three randomly selected windows (seeded):")
    for item in rng.sample(selected, k=3):
        print(f"\n--- {item['window_id']} | {item['page_type']} | {item['url']} | {len(item['text'])} chars ---")
        print(item["text"])


if __name__ == "__main__":
    main()
