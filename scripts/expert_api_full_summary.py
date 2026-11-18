"""Summarize completed full Expert API answers and export a workbook."""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import expert_api_pilot as pilot
from expert_api_full_run import MODEL_NAMES, load_questions, read_jsonl, is_success

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/api_answers/full"
ANSWERS_PATH = OUTPUT_DIR / "answers.jsonl"
DATA_PATH = OUTPUT_DIR / "summary_workbook_data.json"
BUILDER_PATH = ROOT / "scripts/expert_api_full_summary_workbook.mjs"
ARTIFACT_NODE_MODULES = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"
MARKER_PATH = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/plugins/openai-primary-runtime/plugins/spreadsheets/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs"
KB_PATHS = (ROOT / "data/raw/site_pages_rows.csv", ROOT / "site_pages_rows.csv")


def clean_latest(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        latest[(str(record.get("id", "")), str(record.get("model_name", "")))] = record
    return latest


def main() -> None:
    questions = load_questions()
    q_by_id = {q["id"]: q for q in questions}
    answers = read_jsonl(ANSWERS_PATH)
    latest = clean_latest(answers)
    kb_urls = pilot.load_kb_urls()
    expected_sets = {"main": [q for q in questions if q["set"] == "main"], "extra": [q for q in questions if q["set"] == "extra"]}

    # Keep exactly one answer column per requested model, preferring a success if any attempt succeeded.
    for key in list(latest):
        pair_rows = [r for r in answers if (str(r.get("id", "")), str(r.get("model_name", ""))) == key]
        successes = [r for r in pair_rows if is_success(r)]
        if successes:
            latest[key] = successes[-1]

    workbook_rows = []
    for q in questions:
        row = {
            "id": q["id"], "set": q["set"], "type": q.get("type", ""), "page_type": q.get("page_type", ""),
            "question": q.get("question", ""), "reference_answer": q.get("reference_answer", ""), "source_url": q.get("url", ""),
        }
        for model in MODEL_NAMES:
            record = latest.get((q["id"], model))
            row[model] = record.get("answer", "") if record and is_success(record) else ""
        workbook_rows.append(row)
    DATA_PATH.write_text(json.dumps({"rows": workbook_rows}, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Full Environmental Expert API Run Summary", "",
        "No scoring was performed.", "",
        "| Model | Set | Succeeded | Failed | Mean latency (s) | Maximum latency (s) | Mean answer words | Answers with URL | Mean distinct URLs | Cited URLs in KB | Answers citing source URL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        for set_name, set_questions in expected_sets.items():
            selected = [latest.get((q["id"], model)) for q in set_questions]
            successful = [r for r in selected if r and is_success(r)]
            failed_count = len(set_questions) - len(successful)
            latencies = [float(r.get("latency_seconds") or 0) for r in successful]
            word_counts = [len(str(r.get("answer", "")).split()) for r in successful]
            parsed_urls = {r["id"]: pilot.extract_urls(str(r.get("answer", ""))) for r in successful}
            cited_urls = [(url, r) for r in successful for url in parsed_urls.get(r["id"], [])]
            answers_with_url = sum(bool(parsed_urls.get(r["id"])) for r in successful)
            mean_distinct = statistics.mean([len(parsed_urls.get(r["id"], [])) for r in successful]) if successful else None
            kb_hits = sum(pilot.normalize_url(url) in kb_urls for url, _ in cited_urls)
            source_hits = sum(any(pilot.normalize_url(url) == pilot.normalize_url(str(q_by_id[r["id"]].get("url", ""))) for url in parsed_urls.get(r["id"], [])) for r in successful)
            format_pct = lambda n, d: f"{100*n/d:.1f}%" if d else "NA"
            lines.append("| " + " | ".join([
                model, set_name, str(len(successful)), str(failed_count),
                f"{statistics.mean(latencies):.2f}" if latencies else "NA",
                f"{max(latencies):.2f}" if latencies else "NA",
                f"{statistics.mean(word_counts):.1f}" if word_counts else "NA",
                format_pct(answers_with_url, len(successful)),
                f"{mean_distinct:.2f}" if mean_distinct is not None else "NA",
                format_pct(kb_hits, len(cited_urls)),
                format_pct(source_hits, len(successful)),
            ]) + " |")
    lines += ["", "URL and source-page matching use the same normalization as the pilot: scheme, `www.`, trailing slash and case are ignored. URL shares use successful answers; cited-URL KB share is calculated over extracted URL occurrences.", ""]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    if not MARKER_PATH.exists() or not ARTIFACT_NODE_MODULES.exists():
        raise RuntimeError("The spreadsheet artifact runtime is not available.")
    subprocess.run(["node", str(MARKER_PATH), "--operation-kind", "create", "--expected-output-count", "1", "--output-format", "xlsx"], check=True)
    with tempfile.TemporaryDirectory(prefix="expert-api-summary-") as temp:
        temp_path = Path(temp)
        scope = temp_path / "node_modules" / "@oai"
        scope.mkdir(parents=True)
        (scope / "artifact-tool").symlink_to(ARTIFACT_NODE_MODULES / "@oai" / "artifact-tool")
        temporary_builder = temp_path / "build_summary.mjs"
        temporary_builder.write_text(BUILDER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run(["node", str(temporary_builder), str(DATA_PATH), str(OUTPUT_DIR / "answers.xlsx")], cwd=temp_path, check=True)
    print(f"Wrote summary for {len(questions)} questions and {len(answers)} request-attempt records.")


if __name__ == "__main__":
    main()
