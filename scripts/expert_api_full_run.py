"""Run resumable Environmental Expert API answers for the S2 question set."""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

import expert_api_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILES = (
    (ROOT / "outputs/rawpage_questions/question_set_v3/questions_main.jsonl", "main"),
    (ROOT / "outputs/rawpage_questions/question_set_v3/questions_extra.jsonl", "extra"),
)
PILOT_CONFIG = ROOT / "outputs/api_answers/pilot/pilot_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/api_answers/rerun_kbfix"
MODEL_NAMES = ("openai", "openai_gpt56_sol", "openai_gpt6_astra", "egi")
TIMEOUT_SECONDS = 300
PAUSE_SECONDS = 5
RETRY_WAIT_SECONDS = 30
LOCK = threading.Lock()
STOP_RUN = threading.Event()
FAILURE_STREAK = 0
MALFORMED_MARKERS = ("<|start|>", "<|channel|>", "<|call|>")
QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted")
UNAVAILABLE_STOP_COUNT = 16
UNAVAILABLE_RE = re.compile(
    r"(?:could not|couldn't|cannot|can't|unable to|not able to|wasn[’']t able to|"
    r"weren[’']t able to|did not|didn[’']t)\s+(?:locate|find|retrieve|access)|"
    r"no (?:direct |relevant )?(?:mention|document|source|information|page).{0,60}(?:found|available|of)|"
    r"(?:information|documentation|source|page).{0,60}(?:could not be found|was not found|is not available)",
    re.IGNORECASE,
)
TECHNICAL_FAILURE_RE = re.compile(
    r"\bdue to (?:a |an )?(?:technical (?:issue|error|problem|failure)|"
    r"(?:search|retrieval|documentation) (?:error|failure|timeout)|timeout)\b|"
    r"\b(?:documentation )?search(?:es| service| tool)? (?:timed out|failed)\b|"
    r"\b(?:documentation )?retrieval (?:failed|timed out)\b",
    re.IGNORECASE,
)


def load_questions(set_names: tuple[str, ...] = ("main", "extra")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, set_name in QUESTION_FILES:
        if set_name not in set_names:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["set"] = set_name
                rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs are not unique across main and extra sets.")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore a partial final line left by an interrupted write.
                    continue
    return rows


def is_success(row: dict[str, Any]) -> bool:
    status = row.get("http_status")
    return (
        isinstance(status, int) and 200 <= status < 300 and not row.get("error")
        and isinstance(row.get("answer"), str)
        and not is_malformed_answer(row["answer"])
        and not is_technical_failure_answer(row["answer"])
    )


def is_malformed_answer(answer: Any) -> bool:
    return isinstance(answer, str) and any(marker in answer for marker in MALFORMED_MARKERS)


def is_technical_failure_answer(answer: Any) -> bool:
    return isinstance(answer, str) and bool(TECHNICAL_FAILURE_RE.search(answer))


def append_record(answers_path: Path, log_path: Path, row: dict[str, Any]) -> None:
    if not is_success(row):
        raise ValueError("Failed calls must not be saved as answers.")
    json_line = json.dumps(row, ensure_ascii=False)
    with LOCK:
        with answers_path.open("a", encoding="utf-8") as handle:
            handle.write(json_line + "\n")
            handle.flush()
    append_attempt_log(log_path, row)


def append_attempt_log(log_path: Path, row: dict[str, Any]) -> None:
    answer = row.get("answer")
    log_line = json.dumps({
        "time_utc": row["request_time_utc"], "id": row["id"], "set": row["set"],
        "model_name": row["model_name"], "attempt_phase": row["attempt_phase"],
        "http_status": row["http_status"], "latency_seconds": row["latency_seconds"],
        "retries": row["retries"], "error": row["error"],
        "answer_length": len(answer) if isinstance(answer, str) else None,
        "short_answer_under_50": is_success(row) and len(answer) < 50,
    }, ensure_ascii=False)
    with LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(log_line + "\n")
            handle.flush()


def record_stop_failure(log_path: Path, row: dict[str, Any]) -> None:
    global FAILURE_STREAK
    with LOCK:
        if ((isinstance(row["http_status"], int) and row["http_status"] >= 400)
                or is_malformed_answer(row.get("answer"))
                or is_technical_failure_answer(row.get("answer"))):
            FAILURE_STREAK += 1
        else:
            FAILURE_STREAK = 0
        if FAILURE_STREAK >= 3 and not STOP_RUN.is_set():
            STOP_RUN.set()
            reason = "Run stopped after three consecutive HTTP errors, control-token outputs, or technical search failures."
        else:
            reason = None
    if reason:
        append_run_event(log_path, reason)
        print(reason, flush=True)


def append_run_event(log_path: Path, reason: str) -> None:
    event = json.dumps({
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": "run_stopped",
        "reason": reason,
    }, ensure_ascii=False)
    with LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(event + "\n")
            handle.flush()


def append_run_boundary(log_path: Path, event_name: str, **details: Any) -> None:
    event = json.dumps({
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        **details,
    }, ensure_ascii=False)
    with LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(event + "\n")
            handle.flush()


def call_attempt(endpoint: str, question: dict[str, Any], model_name: str, query: str,
                 phase: str, retries: int, last_request_at: list[float],
                 log_path: Path, delay_seconds: float = 0) -> tuple[dict[str, Any] | None, float]:
    if last_request_at[0] > 0:
        wait = PAUSE_SECONDS - (time.monotonic() - last_request_at[0])
        if wait > 0 and STOP_RUN.wait(wait):
            return None, time.monotonic()
    if STOP_RUN.is_set():
        return None, time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    last_request_at[0] = started
    status: int | None = None
    payload: Any = None
    error = ""
    response_body = ""
    try:
        response = requests.post(endpoint, json={"query": query, "model_name": model_name}, timeout=TIMEOUT_SECONDS)
        status = response.status_code
        response_body = response.text or ""
        if any(marker in response_body.lower() for marker in QUOTA_MARKERS):
            error = "openai_quota_exhausted"
            STOP_RUN.set()
            append_run_event(log_path, "Run stopped: API response body indicated exhausted OpenAI quota.")
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            error = error or "non-JSON response"
        if not error and not response.ok:
            error = f"HTTP {status}"
        if not error and (not isinstance(payload, dict) or not isinstance(payload.get("answer"), str)):
            error = "response JSON missing string answer field"
        if not error and is_malformed_answer(payload.get("answer", "")):
            error = "malformed_output"
        if not error and is_technical_failure_answer(payload.get("answer", "")):
            error = "technical_search_failure"
    except requests.RequestException as exc:
        error = "timeout" if isinstance(exc, requests.Timeout) else type(exc).__name__
    elapsed = round(time.monotonic() - started, 3)
    if delay_seconds:
        time.sleep(delay_seconds)
    answer = payload.get("answer", "") if isinstance(payload, dict) else ""
    if not isinstance(answer, str):
        answer = ""
    row = {
        "id": question["id"], "set": question["set"], "type": question.get("type", ""),
        "page_type": question.get("page_type", ""), "question": question.get("question", ""),
        "source_url": question.get("url", ""), "model_name": model_name, "query": query,
        "request_time_utc": timestamp, "latency_seconds": elapsed, "http_status": status,
        "retries": retries, "attempt_phase": phase, "answer": answer,
        "model_name_returned": payload.get("model_name", payload.get("model")) if isinstance(payload, dict) else None,
        "provider_returned": payload.get("provider") if isinstance(payload, dict) else None,
        "error_body": response_body[:500] if error else "",
        "error": error,
    }
    return row, time.monotonic()


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("id", "")), str(row.get("model_name", ""))


def terminal_records(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[pair_key(row)] = row
    return latest


def print_status(path: Path, questions: list[dict[str, Any]], set_names: tuple[str, ...] = ("main", "extra")) -> None:
    rows = read_jsonl(path)
    selected_ids_by_set = {
        set_name: {str(row["id"]) for row in questions if row.get("set") == set_name}
        for set_name in set_names
    }
    selected_ids = set().union(*selected_ids_by_set.values())
    selected_rows = [row for row in rows if str(row.get("id", "")) in selected_ids]
    print("model_name,set,calls_attempted,succeeded,failed,remaining")
    for model in MODEL_NAMES:
        for set_name in set_names:
            target_ids = selected_ids_by_set[set_name]
            attempted_ids = {
                str(row.get("id", "")) for row in selected_rows
                if row.get("model_name") == model and str(row.get("id", "")) in target_ids
            }
            succeeded_ids = {
                str(row.get("id", "")) for row in selected_rows
                if row.get("model_name") == model and str(row.get("id", "")) in target_ids and is_success(row)
            }
            attempted = len(attempted_ids)
            succeeded = len(succeeded_ids)
            total = len(target_ids)
            print(f"{model},{set_name},{attempted},{succeeded},{attempted-succeeded},{max(0,total-succeeded)}")


def run(args: argparse.Namespace) -> None:
    global FAILURE_STREAK
    STOP_RUN.clear()
    FAILURE_STREAK = 0
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers.jsonl"
    log_path = output_dir / "run.log"
    env_values = pilot.load_env_file(ROOT / ".env")
    endpoint = env_values.get("ENVRI_EXPERT_URL", "").strip()
    if not endpoint:
        raise SystemExit("ENVRI_EXPERT_URL is missing from .env; no API calls were made.")
    suffix_config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    suffix = suffix_config.get("query_suffix")
    if not isinstance(suffix, str) or not suffix:
        raise SystemExit("The pilot query suffix is unavailable; no API calls were made.")
    questions = load_questions(tuple(args.sets.split(",")))
    random.Random(42).shuffle(questions)
    if args.limit is not None:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit("No questions selected.")
    previous = read_jsonl(answers_path)
    successes = {pair_key(row) for row in previous if is_success(row)}
    total = len(questions)
    append_run_boundary(log_path, "run_started", sets=args.sets, questions=total,
                        parallel=args.parallel, delay_seconds=args.delay_seconds)
    failed_jobs: list[dict[str, Any]] = []
    last_request_at_by_model = {model: [0.0] for model in MODEL_NAMES}
    serial_last_request_at = [0.0]
    unavailable_windows = {model: deque(maxlen=20) for model in MODEL_NAMES}
    print(f"Selected {total} questions, {len(successes)} successful question/model pairs already saved; parallel={args.parallel}.", flush=True)

    def record_result(row: dict[str, Any]) -> None:
        if is_success(row):
            append_record(answers_path, log_path, row)
            reason = None
            with LOCK:
                model = row["model_name"]
                window = unavailable_windows[model]
                window.append(bool(UNAVAILABLE_RE.search(row["answer"])))
                if len(window) == 20 and sum(window) >= UNAVAILABLE_STOP_COUNT and not STOP_RUN.is_set():
                    STOP_RUN.set()
                    reason = f"Run stopped: {sum(window)}/20 consecutive new {model} answers say information could not be found."
            if reason:
                append_run_event(log_path, reason)
                print(reason, flush=True)
        else:
            append_attempt_log(log_path, row)
        record_stop_failure(log_path, row)

    def worker(model_names: list[str], question_rows: list[dict[str, Any]], serial_all_models: bool = False) -> None:
        def one(q: dict[str, Any], model: str) -> None:
            if STOP_RUN.is_set():
                return
            key = (q["id"], model)
            if key in successes:
                return
            query = f"{q['question']}\n\n{suffix}"
            retry_count = 0
            last_request_at = serial_last_request_at if serial_all_models else last_request_at_by_model[model]
            row, _ = call_attempt(endpoint, q, model, query, "initial", retry_count, last_request_at, log_path,
                                  args.delay_seconds)
            if row is None:
                return
            record_result(row)
            print(f"Recorded {q['id']} {model} initial: HTTP {row['http_status']} ({row['error'] or 'ok'}).", flush=True)
            if STOP_RUN.is_set():
                return
            should_retry = row["error"] in {"timeout", "malformed_output", "technical_search_failure"} or (isinstance(row["http_status"], int) and row["http_status"] >= 500)
            if should_retry:
                if STOP_RUN.wait(RETRY_WAIT_SECONDS):
                    return
                if STOP_RUN.is_set():
                    return
                retry_count += 1
                row, _ = call_attempt(endpoint, q, model, query, "initial_retry", retry_count, last_request_at, log_path,
                                      args.delay_seconds)
                if row is None:
                    return
                record_result(row)
                print(f"Recorded {q['id']} {model} retry: HTTP {row['http_status']} ({row['error'] or 'ok'}).", flush=True)
            if is_success(row):
                successes.add(key)
            else:
                if row.get("error") not in {"malformed_output", "technical_search_failure"} and not STOP_RUN.is_set():
                    with LOCK:
                        failed_jobs.append({"question": q, "model_name": model, "query": query, "retries": retry_count})

        if serial_all_models:
            for q in question_rows:
                for model in model_names:
                    one(q, model)
        else:
            model = model_names[0]
            for q in question_rows:
                one(q, model)
    if args.parallel == 1:
        worker(list(MODEL_NAMES), questions, serial_all_models=True)
    else:
        threads = [threading.Thread(target=worker, args=([model], questions), name=f"expert-{model}") for model in MODEL_NAMES]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    # Begin the extra retry phase only after every model has completed its first pass.
    def final_retry_worker(jobs: list[dict[str, Any]], serial_last: list[float] | None = None) -> None:
        for item in jobs:
            q = item["question"]
            model = item["model_name"]
            last_request_at = serial_last if serial_last is not None else last_request_at_by_model[model]
            if STOP_RUN.is_set():
                return
            row, _ = call_attempt(endpoint, q, model, item["query"], "final_retry", item["retries"] + 1, last_request_at,
                                  log_path, args.delay_seconds)
            if row is None:
                return
            record_result(row)
            print(f"Recorded {q['id']} {model} final_retry: HTTP {row['http_status']} ({row['error'] or 'ok'}).", flush=True)
            if is_success(row):
                successes.add((q["id"], model))

    if STOP_RUN.is_set():
        print("Run stopped; no final retry phase was started.", flush=True)
    elif args.parallel == 1:
        final_retry_worker(failed_jobs, serial_last_request_at)
    else:
        retry_threads = []
        for model in MODEL_NAMES:
            model_jobs = [job for job in failed_jobs if job["model_name"] == model]
            thread = threading.Thread(target=final_retry_worker, args=(model_jobs,), name=f"expert-final-{model}")
            retry_threads.append(thread)
            thread.start()
        for thread in retry_threads:
            thread.join()
    print_status(answers_path, questions, tuple(args.sets.split(",")))
    append_run_boundary(log_path, "run_ended", stopped=STOP_RUN.is_set(),
                        successful_pairs=len(successes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets", choices=("main", "extra", "main,extra"), default="main,extra",
        help="Question sets to process: main, extra, or main,extra (default).",
    )
    parser.add_argument("--parallel", type=int, choices=(1, 4), default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-seconds", type=float, default=0,
                        help="Seconds to sleep after each request finishes, whether successful or failed.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions in the seeded order (used for a small test).")
    parser.add_argument("--status", action="store_true", help="Print progress for the selected output directory without making requests.")
    args = parser.parse_args()
    if args.status:
        questions = load_questions(tuple(args.sets.split(",")))
        random.Random(42).shuffle(questions)
        if args.limit is not None:
            questions = questions[:args.limit]
        print_status(args.output_dir / "answers.jsonl", questions, tuple(args.sets.split(",")))
        return
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive.")
    if args.delay_seconds < 0:
        raise SystemExit("--delay-seconds must be nonnegative.")
    run(args)


if __name__ == "__main__":
    main()
