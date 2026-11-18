"""First-pass cleaning of the ENVRI knowledge-base CSV (Supabase `site_pages` export).

Reproduces the numbers in the RQ4 plan (13 Oct 2026):
  35,543 chunks / 1,196 pages in  ->  18,780 chunks / 1,144 pages kept
  -> 1,015 English pages with at least 600 characters of text

Steps
  1. Drop empty chunks.
  2. Drop boilerplate: chunks whose exact text appears on 3 or more different pages
     (cookie notices, funding statements, menus).
  3. Drop repeated chunks within the same page.
  4. Rebuild pages from the kept chunks, tag language (word-count heuristic) and
     page type (URL rules), and mark pages eligible for sampling.

Usage
  python clean_kb.py --input data/raw/site_pages_rows.csv --outdir data/processed

Outputs (in --outdir)
  chunks_clean.csv     kept chunks (no embedding column)
  pages.csv            one row per page: url, domain, chunks, chars, lang, page_type, eligible
  snapshot_info.json   input file hash, row counts, crawl dates, counts after each step
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

BOILERPLATE_MIN_PAGES = 3
MIN_PAGE_CHARS = 600

EN_WORDS = set("the and of to in for is are with on that this by from as be an or which data at".split())
FR_WORDS = set("le la les des et du une est pour dans sur au aux par en que qui".split())

DATASET_DOMAINS = {
    "edmed.seadatanet.org", "meta.icos-cp.eu", "erddap.ifremer.fr", "data.blue-cloud.org",
    "data.marine.copernicus.eu", "dataselection.euro-argo.eu", "iagos.aeris-data.fr",
    "stac-browser.ifremer.fr", "data.fieldsites.se", "deims.org",
}
NEWS_PATTERN = re.compile(
    r"/(news|events?|tag|category|20\d\d|blog|communication|actualite|articles|news-and-events|news-events)/"
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def page_language(text: str) -> str:
    words = re.findall(r"[a-zA-Zàâçéèêëîïôûùüÿœ']+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    fr = sum(w in FR_WORDS for w in words)
    if en + fr < 10:
        return "unclear"
    return "en" if en >= 2 * fr else "other"


def page_type(url: str) -> str:
    p = urlparse(url)
    domain, path = p.netloc, p.path + "/"
    # Machine-readable API output and search/listing pages come first, so that
    # e.g. dar.elter-ri.eu/api/external-datasets/... is not taken for a dataset record.
    if "/api/" in path or domain.startswith("developer"):
        return "api_docs"
    if "/search" in path:
        return "search_page"
    if domain in DATASET_DOMAINS or "external-datasets" in path:
        return "dataset_record"
    if domain.startswith("vocab"):
        return "vocabulary"
    if NEWS_PATTERN.search(path):
        return "news_events"
    return "about_science"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="site_pages CSV export")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.input, nrows=0).columns
    usecols = [c for c in header if c.strip().lower() != "embedding"]
    df = pd.read_csv(args.input, usecols=usecols, dtype=str, keep_default_na=False)
    df["chunk_number"] = pd.to_numeric(df["chunk_number"])
    n_in, pages_in = len(df), df["url"].nunique()

    crawled = df["metadata"].str.extract(r'"crawled_at":\s*"([^"]+)"')[0]

    non_empty = df[df["content"].str.strip() != ""]
    pages_per_text = non_empty.groupby("content")["url"].nunique()
    non_empty = non_empty.assign(pages_with_same_text=non_empty["content"].map(pages_per_text))
    no_boiler = non_empty[non_empty["pages_with_same_text"] < BOILERPLATE_MIN_PAGES]
    kept = no_boiler.drop_duplicates(["url", "content"]).drop(columns=["pages_with_same_text"])

    pages = (
        kept.sort_values(["url", "chunk_number"])
        .groupby("url")
        .agg(text=("content", "".join), chunks=("id", "size"))
    )
    pages["domain"] = [urlparse(u).netloc for u in pages.index]
    pages["chars"] = pages["text"].str.len()
    pages["lang"] = pages["text"].map(page_language)
    pages["page_type"] = [page_type(u) for u in pages.index]
    pages["eligible"] = (pages["lang"] == "en") & (pages["chars"] >= MIN_PAGE_CHARS)

    kept.to_csv(args.outdir / "chunks_clean.csv", index=False)
    pages.drop(columns=["text"]).to_csv(args.outdir / "pages.csv")

    info = {
        "input_file": str(args.input),
        "input_sha256": file_sha256(args.input),
        "input_chunks": n_in,
        "input_pages": pages_in,
        "crawled_at_min": crawled.min(),
        "crawled_at_max": crawled.max(),
        "removed_empty": n_in - len(non_empty),
        "removed_boilerplate": len(non_empty) - len(no_boiler),
        "removed_within_page_repeats": len(no_boiler) - len(kept),
        "kept_chunks": len(kept),
        "kept_pages": len(pages),
        "eligible_pages": int(pages["eligible"].sum()),
        "eligible_pages_by_type": pages[pages["eligible"]]["page_type"].value_counts().to_dict(),
    }
    (args.outdir / "snapshot_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
