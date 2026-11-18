"""Five isolated KB repair checks; never writes to the collection answers file."""
from __future__ import annotations

import json
from pathlib import Path

import expert_api_full_run as run
import expert_api_pilot as pilot


PAIRS = (
    ("AS-095-f1c1fd1835_2", "egi"),
    ("DR-001-1fd19d68d6_1", "openai"),
    ("DR-002-45b7adbfaa_1", "openai_gpt56_sol"),
    ("DR-003-edb664c96e_1", "openai_gpt6_astra"),
    ("DR-004-baff698f11_1", "egi"),
)
OUTPUT = run.ROOT / "outputs/api_answers/rerun_kbfix/smoke_test.jsonl"
OLD_ANSWERS = run.ROOT / "outputs/api_answers/full/answers.jsonl"
SCORES = run.ROOT / "outputs/api_scoring/full300/items"


def main() -> None:
    questions = {row["id"]: row for row in run.load_questions(("main",))}
    old = {run.pair_key(row): row for row in run.read_jsonl(OLD_ANSWERS) if run.is_success(row)}
    for question_id, model in PAIRS:
        question = questions[question_id]
        if question_id.startswith("DR-") and not any(
            name in question["url"].lower() for name in ("edmed", "seadatanet")
        ):
            raise ValueError(f"Not an EDMED/SeaDataNet question: {question_id}")
        if (question_id, model) not in old:
            raise ValueError(f"No archived valid answer: {question_id} {model}")
        score_path = SCORES / f"{question_id}__{model}.json"
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if score.get("primary_abstention_label") != "abstains":
            raise ValueError(f"Archived answer is not an abstention: {question_id} {model}")

    if OUTPUT.exists() and OUTPUT.stat().st_size:
        raise SystemExit("Smoke-test file already has results; refusing to repeat calls.")
    config = json.loads(run.PILOT_CONFIG.read_text(encoding="utf-8"))
    suffix = config.get("query_suffix")
    if not isinstance(suffix, str) or not suffix:
        raise SystemExit("The pilot query suffix is unavailable; no API calls were made.")
    endpoint = pilot.load_env_file(run.ROOT / ".env").get("ENVRI_EXPERT_URL", "").strip()
    if not endpoint:
        raise SystemExit("The API configuration is missing; no API calls were made.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT.parent / "smoke_test.log"
    last_request_at = [0.0]
    run.STOP_RUN.clear()
    for question_id, model in PAIRS:
        question = questions[question_id]
        query = f"{question['question']}\n\n{suffix}"
        result, _ = run.call_attempt(endpoint, question, model, query, "smoke_test", 0,
                                     last_request_at, log_path, 2)
        if result is None:
            break
        saved = {
            "id": question_id,
            "model_name": model,
            "source_url": question["url"],
            "request_time_utc": result["request_time_utc"],
            "http_status": result["http_status"],
            "latency_seconds": result["latency_seconds"],
            "error": result["error"],
            "old_answer": old[(question_id, model)]["answer"],
            "new_answer": result["answer"],
        }
        with OUTPUT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
            handle.flush()
        run.append_attempt_log(log_path, result)
        print(f"Smoke call {question_id} {model}: HTTP {result['http_status']} ({result['error'] or 'ok'}).", flush=True)
        run.record_stop_failure(log_path, result)
        if run.STOP_RUN.is_set():
            break


if __name__ == "__main__":
    main()
