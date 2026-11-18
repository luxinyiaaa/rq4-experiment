"""Retry only failed Faithfulness measurements in the frozen final run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_index import _read_jsonl, _write_jsonl
from scripts.score_answers import _run_ragas


def retry(scores_path: Path, answers_path: Path, questions_path: Path) -> None:
    scores = _read_jsonl(scores_path)
    answers = {(str(r["id"]), str(r["condition"])): r for r in _read_jsonl(answers_path)}
    questions = {str(r["id"]): r for r in _read_jsonl(questions_path)}
    failed = [r for r in scores if (r.get("faithfulness") or {}).get("status") == "error"]
    print(f"Retrying {len(failed)} failed Faithfulness records with the pilot settings.", flush=True)
    for row in failed:
        key = (str(row["id"]), str(row["condition"]))
        try:
            value, raw = _run_ragas("faithfulness", answers[key], questions[key[0]])
            row["faithfulness"] = value
            row.setdefault("raw_outputs", {})["ragas_faithfulness_retry"] = raw
            print(f"{key[0]} {key[1]}: {value.get('status')} {value.get('value')}", flush=True)
        except Exception as exc:
            row["faithfulness"] = {"value": None, "status": "error", "reason": f"Retry {type(exc).__name__}: {exc}"}
            row.setdefault("raw_outputs", {})["ragas_faithfulness_retry"] = {"error": row["faithfulness"]["reason"]}
            print(f"{key[0]} {key[1]}: retry failed: {row['faithfulness']['reason']}", flush=True)
    _write_jsonl(scores_path, scores)
    print(f"Updated {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=PROJECT_ROOT / "outputs/final/scores_final.jsonl")
    parser.add_argument("--answers", type=Path, default=PROJECT_ROOT / "outputs/final/answers_final.jsonl")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "outputs/final/questions_final.jsonl")
    args = parser.parse_args()
    retry(args.scores, args.answers, args.questions)


if __name__ == "__main__":
    main()
