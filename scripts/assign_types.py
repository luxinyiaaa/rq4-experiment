"""Assign the RQ4 question types and retrieve comparison/false-premise partners."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_index import DEFAULT_OUTDIR, load_index
from scripts.build_index import _read_jsonl, _url_template

INPUT_WINDOWS = PROJECT_ROOT / "data" / "processed" / "windows_screened.jsonl"
OUTPUT_PLAN = PROJECT_ROOT / "data" / "processed" / "question_plan.jsonl"
TYPE_COUNTS = {
    "list_all": 12,
    "comparison": 11,
    "paraphrased": 14,
    "false_premise": 13,
}


def assign_records(
    windows: list[dict[str, Any]],
    candidates_for: Callable[[str, int], list[dict[str, Any]]],
    *,
    candidate_count: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Create a reproducible 50-item plan while ensuring required partners exist."""
    dataset = [row for row in windows if row["page_type"] == "dataset_record"]
    other = [row for row in windows if row["page_type"] in {"about_science", "vocabulary"}]
    if len(dataset) != 23 or len(other) != 27:
        raise ValueError(
            f"Expected 23 dataset_record and 27 about_science/vocabulary windows; got {len(dataset)} and {len(other)}"
        )

    comparison_partners: dict[str, dict[str, Any]] = {}
    for row in dataset:
        found = candidates_for(str(row["window_id"]), candidate_count)
        same_template = [
            candidate for candidate in found
            if candidate.get("page_type") == "dataset_record"
            and candidate.get("url_template", _url_template(candidate["url"]))
            == _url_template(row["url"])
        ]
        if same_template:
            comparison_partners[str(row["window_id"])] = same_template[0]

    false_similars: dict[str, dict[str, Any]] = {}
    for row in other:
        found = candidates_for(str(row["window_id"]), 10)
        if found:
            false_similars[str(row["window_id"])] = found[0]

    rng = random.Random(seed)
    eligible_comparison = [row for row in dataset if str(row["window_id"]) in comparison_partners]
    if len(eligible_comparison) < TYPE_COUNTS["comparison"]:
        raise RuntimeError(
            f"Only {len(eligible_comparison)} dataset windows have a same-template partner; "
            f"{TYPE_COUNTS['comparison']} are required."
        )
    rng.shuffle(eligible_comparison)
    comparison_ids = {str(row["window_id"]) for row in eligible_comparison[: TYPE_COUNTS["comparison"]]}
    list_all = [row for row in dataset if str(row["window_id"]) not in comparison_ids]
    rng.shuffle(list_all)
    if len(list_all) != TYPE_COUNTS["list_all"]:
        raise AssertionError("Dataset type allocation did not produce the requested counts")

    eligible_false = [row for row in other if str(row["window_id"]) in false_similars]
    if len(eligible_false) < TYPE_COUNTS["false_premise"]:
        raise RuntimeError(
            f"Only {len(eligible_false)} windows have a distractor candidate; "
            f"{TYPE_COUNTS['false_premise']} are required."
        )
    rng.shuffle(eligible_false)
    false_ids = {str(row["window_id"]) for row in eligible_false[: TYPE_COUNTS["false_premise"]]}
    paraphrased = [row for row in other if str(row["window_id"]) not in false_ids]
    rng.shuffle(paraphrased)
    if len(paraphrased) != TYPE_COUNTS["paraphrased"]:
        raise AssertionError("Non-dataset type allocation did not produce the requested counts")

    type_by_id: dict[str, str] = {}
    for row in list_all:
        type_by_id[str(row["window_id"])] = "list_all"
    for row in eligible_comparison[: TYPE_COUNTS["comparison"]]:
        type_by_id[str(row["window_id"])] = "comparison"
    for row in paraphrased:
        type_by_id[str(row["window_id"])] = "paraphrased"
    for row in eligible_false[: TYPE_COUNTS["false_premise"]]:
        type_by_id[str(row["window_id"])] = "false_premise"

    plan = []
    for window in windows:
        window_id = str(window["window_id"])
        item = {
            "id": window_id,
            "window_id": window_id,
            "type": type_by_id[window_id],
            "page_type": window["page_type"],
            "url": window["url"],
            "domain": window.get("domain") or _url_template(window["url"]).split("/", 1)[0],
            "url_template": _url_template(window["url"]),
            "text": window["text"],
            "chunk_ids": window["chunk_ids"],
        }
        if item["type"] == "comparison":
            partner = comparison_partners[window_id]
            item["partner"] = {
                "window_id": partner["window_id"],
                "url": partner["url"],
                "domain": partner["domain"],
                "page_type": partner["page_type"],
                "url_template": partner["url_template"],
                "text": partner["text"],
                "chunk_ids": partner["chunk_ids"],
            }
        elif item["type"] == "false_premise":
            similar = false_similars[window_id]
            item["similar"] = {
                "window_id": similar["window_id"],
                "url": similar["url"],
                "domain": similar["domain"],
                "page_type": similar["page_type"],
                "text": similar["text"],
                "chunk_ids": similar["chunk_ids"],
            }
        plan.append(item)
    return plan


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=INPUT_WINDOWS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_PLAN)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = load_index(args.index_dir)
    windows = _read_jsonl(args.windows)
    plan = assign_records(
        windows,
        index.get_distractor_candidates,
        candidate_count=len(index.window_records),
        seed=args.seed,
    )
    write_jsonl(args.out, plan)
    print(f"Question plan: {len(plan)} items")
    for question_type, count in Counter(row["type"] for row in plan).items():
        print(f"  {question_type}: {count}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
