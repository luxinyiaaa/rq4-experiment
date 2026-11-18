"""Link extracted claims that share a specific named entity across pages."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_kb_index import OUT as INDEX_DIR, read_jsonl, write_jsonl

RELATION = re.compile(r"\b(?:produced by|operated by|hosted by|part of|coordinated by|provided by|maintained by|managed by|run by|located in|member of|uses|measures)\b", re.I)
GENERIC = {"research infrastructure", "data", "project", "network", "instrument", "station", "variable", "service", "infrastructure"}


def _specific(name: str) -> bool:
    name = re.sub(r"\s+", " ", name).strip(" .,;:")
    if len(name) < 3 or name.casefold() in GENERIC:
        return False
    words = name.split()
    return name.isupper() or any(w[:1].isupper() for w in words) or bool(re.search(r"\b[A-Z]{2,}\b", name))


def build_bridges(claim_path: Path = INDEX_DIR / "page_claims.jsonl", output: Path = INDEX_DIR / "claim_bridges.jsonl") -> list[dict]:
    claims_by_entity: dict[str, list[dict]] = defaultdict(list)
    for page in read_jsonl(claim_path):
        for claim in page.get("claims", []):
            for entity in claim.get("entities", []):
                entity = re.sub(r"\s+", " ", str(entity)).strip(" .,;:")
                if _specific(entity):
                    claims_by_entity[entity.casefold()].append({**claim, "url": page["url"], "page_title": page.get("title", ""), "entity": entity})
    bridges = []
    seen = set()
    for entity_key, rows in claims_by_entity.items():
        by_url: dict[str, list[dict]] = defaultdict(list)
        for row in rows: by_url[row["url"]].append(row)
        urls = list(by_url)
        for i, url_a in enumerate(urls):
            for url_b in urls[i + 1:]:
                if url_a == url_b: continue
                domain_a, domain_b = urlparse(url_a).netloc, urlparse(url_b).netloc
                cross_site = domain_a.casefold() != domain_b.casefold()
                for a in by_url[url_a]:
                    if not RELATION.search(a.get("quote", "")) or entity_key not in str(a.get("quote", "")).casefold():
                        continue
                    for b in by_url[url_b]:
                        if entity_key not in str(b.get("page_title", "")).casefold():
                            continue
                        fact_b = b.get("claim", "")
                        concrete = bool(b.get("property") and b.get("value")) or bool(re.search(r"\b(?:18|19|20|21)\d{2}\b|\b\d+(?:\.\d+)?\b|\b(?:in|at|near|located|based|member|station|instrument)\b", fact_b, re.I))
                        if not concrete: continue
                        key = (entity_key, a["url"], a["chunk_id"], b["url"], b["chunk_id"])
                        if key in seen: continue
                        seen.add(key)
                        bridges.append({"bridge_id": f"BR-{len(bridges)+1:05d}", "entity": a["entity"],
                            "cross_site": cross_site,
                            "fact_a": {k:a.get(k) for k in ("url","page_title","claim","quote","chunk_id","subject","property","value")},
                            "fact_b": {k:b.get(k) for k in ("url","page_title","claim","quote","chunk_id","subject","property","value")}})
    bridges.sort(key=lambda r: (not r["cross_site"], r["entity"].casefold(), r["bridge_id"]))
    write_jsonl(output, bridges)
    print(json.dumps({"claims": sum(len(p.get("claims", [])) for p in read_jsonl(claim_path)),
                      "bridges": len(bridges), "cross_site_bridges": sum(b["cross_site"] for b in bridges),
                      "output": str(output)}, indent=2))
    return bridges


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims",type=Path,default=INDEX_DIR/"page_claims.jsonl")
    parser.add_argument("--out",type=Path,default=INDEX_DIR/"claim_bridges.jsonl")
    args=parser.parse_args(); build_bridges(args.claims,args.out)


if __name__=="__main__": main()
