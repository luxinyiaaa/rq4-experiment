"""Prepare rows for the v4 workbook; XLSX authoring is in build_v4.mjs."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from outputs.api_scoring.rerun_kbfix import build_reports as reports
from outputs.api_scoring.rerun_kbfix.run_rerun import ANSWER_SOURCE, complete_jsonl

OUT = reports.OUT
MODEL_NAMES = {"openai": "GPT-4.1", "openai_gpt56_sol": "GPT-5.6 Sol",
               "openai_gpt6_astra": "GPT-6 Astra", "egi": "EGI gpt-oss-120b"}
HEADERS = ["question_id", "model", "question_type", "page_type", "source_url", "question",
           "reference_answer", "keypoints", "full_llm_answer", "primary_correct_judge_only",
           "primary_correctness_reason", "secondary_correct_judge_only", "primary_abstention",
           "secondary_abstention", "primary_completeness", "secondary_completeness",
           "primary_relevance", "secondary_relevance", "primary_citation_support",
           "secondary_citation_support", "primary_citation_completeness",
           "secondary_citation_completeness", "primary_keypoint_labels", "secondary_keypoint_labels"]


def collection_dates(answers: list[dict]) -> tuple[str, str]:
    log_path = ANSWER_SOURCE.parent / "run.log"
    events = complete_jsonl(log_path) if log_path.exists() else []
    starts = [x["time_utc"] for x in events if x.get("event") == "run_started"]
    ends = [x["time_utc"] for x in events if x.get("event") == "run_ended"]
    stamps = [x["request_time_utc"] for x in answers]
    return (min(starts) if starts else min(stamps), max(ends) if ends else max(stamps))


def main() -> None:
    snapshot = json.loads((OUT / "input_snapshot.json").read_text(encoding="utf-8"))
    answers = complete_jsonl(ANSWER_SOURCE)
    if len(answers) != 1200 or len(snapshot["answer_ids"]) != 1200:
        raise ValueError("Refusing workbook before all 1,200 answers are collected and snapshotted")
    start, end = collection_dates(answers)
    rows = []
    for key in snapshot["answer_ids"]:
        item = json.loads((OUT / "items" / (key + ".json")).read_text(encoding="utf-8"))
        by_judge = {j["judge"]: j for j in item["judgements"]
                    if j.get("metric") == "answer_correctness_completeness"}
        primary = by_judge["primary"]
        labels = primary["labels"]
        correct = bool(item["metrics"]["answer_correctness"]["primary"])
        rule = ("Correct: every decisive keypoint is entailed and no keypoint is contradicted." if correct
                else "Incorrect: at least one decisive keypoint is not entailed or a keypoint is contradicted.")
        reasons = [f"{i}. {label} — {trace.get('reason', '')}" for i, (label, trace) in
                   enumerate(zip(labels, primary["trace"]), 1)]
        reason = rule + "\nPrimary judge keypoint reasons:\n" + "\n".join(reasons)
        metrics = item["metrics"]
        def score(metric: str, judge: str):
            val = metrics[metric][judge]
            if isinstance(val, dict):
                val = val.get("score")
            return "" if val is None else round(float(val), 2)
        row = [item["id"], MODEL_NAMES[item["model"]], item["type"], item["page_type"],
               item["source_url"], item["question"], item["reference_answer"],
               "\n".join(f"{i}. {'[DECISIVE] ' if kp.get('decisive') else ''}{kp['text']}"
                         for i, kp in enumerate(item["keypoints"], 1)), item["answer"],
               correct, reason, bool(metrics["answer_correctness"]["secondary"]),
               metrics["appropriate_abstention"]["primary"]["label"],
               metrics["appropriate_abstention"]["secondary"]["label"],
               score("completeness", "primary"), score("completeness", "secondary"),
               score("response_relevance", "primary"), score("response_relevance", "secondary"),
               score("citation_support", "primary"), score("citation_support", "secondary"),
               score("citation_completeness", "primary"), score("citation_completeness", "secondary"),
               "; ".join(labels), "; ".join(by_judge["secondary"]["labels"])]
        if len(row) != len(HEADERS):
            raise ValueError("Workbook row width differs from v3")
        rows.append(row)
    if len(rows) != 1200 or len({(r[0], r[1]) for r in rows}) != 1200:
        raise ValueError("Workbook rows are missing or duplicated")
    (OUT / "v4_workbook_data.json").write_text(json.dumps(
        {"headers": HEADERS, "rows": rows, "collection_start_utc": start,
         "collection_end_utc": end}, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(rows)} workbook rows")


if __name__ == "__main__":
    main()
