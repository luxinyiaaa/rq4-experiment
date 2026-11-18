"""Resume the 300-question scoring run with a durable cache for every judge call."""
from __future__ import annotations

import hashlib
import argparse
from datetime import datetime, timezone
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from outputs.api_scoring.full import run_full as old
from outputs.api_scoring.full import relevance_rescore
from scripts import api_scoring_pilot as base
from scripts import expert_api_full_run as answer_checks

OUT = Path(os.environ.get("RERUN_SCORING_DIR", Path(__file__).resolve().parent))
ITEMS = OUT / "items"
CACHE = OUT / "judge_cache"
SNAPSHOT = OUT / "input_snapshot.json"
ANSWER_SOURCE = Path(os.environ.get("RERUN_ANSWERS_PATH", ROOT / "outputs/api_answers/rerun_kbfix/answers.jsonl"))
FULL300_ITEMS = ROOT / "outputs/api_scoring/full300/items"
TOKEN_LIMIT = 40_000_000
RULE_VERSION = old.RULE_VERSION + "-final-relevance-full300"
STOP = threading.Event()
QUOTA_STOP = threading.Event()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def complete_jsonl(path: Path) -> list[dict]:
    """Read a live append-only file without accepting its unterminated final line."""
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def read_selected() -> tuple[dict, dict, dict]:
    questions = {q["id"]: q for q in base.read_jsonl(old.QUESTIONS)}
    selected = defaultdict(list)
    source_rows = complete_jsonl(ANSWER_SOURCE)
    for row in source_rows:
        if (row.get("set") == "main" and row.get("id") in questions
                and row.get("model_name") in old.MODELS and answer_checks.is_success(row)):
            selected[f"{row['id']}__{row['model_name']}"].append(row)
    duplicate = [key for key, rows in selected.items() if len(rows) != 1]
    if duplicate:
        raise ValueError(f"Duplicate valid answer pairs: {duplicate[:10]}")
    if any(row.get("id") not in questions or row.get("model_name") not in old.MODELS
           for row in source_rows):
        raise ValueError("Answer file contains an unknown question or model")
    if any(not answer_checks.is_success(row) for row in source_rows):
        raise ValueError("Answer file contains a failed or invalid call")
    if SNAPSHOT.exists():
        snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if snap.get("input_source") not in (str(ANSWER_SOURCE), str(ANSWER_SOURCE.relative_to(ROOT))
                                            if ANSWER_SOURCE.is_relative_to(ROOT) else ""):
            raise ValueError("Input source differs from the saved snapshot")
    else:
        snap = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "input_source": str(ANSWER_SOURCE),
                "question_source": str(old.QUESTIONS.relative_to(ROOT)),
                "question_ids": sorted(questions), "answer_ids": [],
                "answer_text_sha256": {}, "token_limit": TOKEN_LIMIT}
    for key in snap["answer_ids"]:
        if len(selected[key]) != 1:
            raise ValueError(f"Previously snapshotted answer is missing: {key}")
    for key, rows in selected.items():
        row = rows[0]
        digest = hashlib.sha256(row["answer"].encode("utf-8")).hexdigest()
        if key in snap["answer_text_sha256"] and digest != snap["answer_text_sha256"][key]:
            raise ValueError(f"Answer text changed since the input snapshot: {key}")
        snap["answer_text_sha256"][key] = digest
    snap["answer_ids"] = sorted(selected)
    snap["answer_count"] = len(selected)
    snap["question_count"] = len(questions)
    snap["model_count"] = len(old.MODELS)
    snap["source_answer_rows_at_snapshot"] = len(source_rows)
    snap["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(SNAPSHOT, snap)
    answers = {key: selected[key][0] for key in snap["answer_ids"]}
    return snap, questions, answers


class CachedJudges(old.LimitedJudges):
    def __init__(self) -> None:
        old.OUT = OUT  # The inherited judge logger writes only in full300.
        super().__init__()
        self.cache_lock = threading.Lock()
        self.key_locks: dict[str, threading.Lock] = {}
        self.initial_tokens = 0
        log = OUT / "judge_calls.jsonl"
        if log.exists():
            for row in base.read_jsonl(log):
                self.initial_tokens += int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))

    def actual_tokens(self) -> int:
        with self.lock:
            current = sum(v["input"] + v["output"] for v in self.tokens.values())
        return self.initial_tokens + current

    def prompt(self, role, name, payload, system_extra=""):
        if name == "response_relevance":
            result = self.call(role, [
                {"role": "system", "content": relevance_rescore.PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], json_mode=True, temperature=0)["json"]
            if type(result.get("score")) is not int or result["score"] not in range(1, 6):
                raise ValueError("Invalid final-rubric relevance score")
            return result
        return super().prompt(role, name, payload, system_extra)

    def call(self, role, messages, **kwargs):
        payload = {"role": role, "model": self.settings[role]["model"],
                   "messages": messages, "kwargs": kwargs, "rule_version": RULE_VERSION}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                        default=str).encode("utf-8")).hexdigest()
        path = CACHE / (key + ".json")
        with self.cache_lock:
            lock = self.key_locks.setdefault(key, threading.Lock())
        with lock:
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                result["cache_hit"] = True
                return result
            if QUOTA_STOP.is_set():
                raise RuntimeError("OpenRouter HTTP 402; judging paused.")
            if STOP.is_set() or self.actual_tokens() > TOKEN_LIMIT:
                STOP.set()
                raise RuntimeError("Actual judge usage exceeded 40 million tokens.")
            try:
                result = super().call(role, messages, **kwargs)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 402:
                    QUOTA_STOP.set()
                raise
            if any(m.get("role") == "system" and "several statements share the same set of cited URLs" in m.get("content", "")
                   for m in messages) and not isinstance(result.get("json", {}).get("statements"), list):
                raise ValueError("Invalid grouped citation response")
            write_json(path, result)
            if self.actual_tokens() > TOKEN_LIMIT:
                STOP.set()
            return result


def prepare_reuse(answers: dict) -> tuple[int, list[str]]:
    ITEMS.mkdir(parents=True, exist_ok=True)
    reused = 0
    pending = []
    for key, answer in answers.items():
        target = ITEMS / (key + ".json")
        if target.exists():
            item = json.loads(target.read_text(encoding="utf-8"))
            if item.get("answer") != answer["answer"]:
                raise ValueError(f"A full300 item has a different answer: {key}")
            reused += 1
            continue
        source = FULL300_ITEMS / (key + ".json")
        if source.exists():
            item = json.loads(source.read_text(encoding="utf-8"))
            if item.get("answer") == answer["answer"]:
                shutil.copy2(source, target)
                reused += 1
                continue
        pending.append(key)
    return reused, pending


def print_status() -> None:
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    keys = snap.get("answer_ids", [])
    finished = sum((ITEMS / (key + ".json")).exists() for key in keys)
    log = OUT / "judge_calls.jsonl"
    calls = len(complete_jsonl(log)) if log.exists() else 0
    tokens = sum(int(x.get("input_tokens", 0)) + int(x.get("output_tokens", 0))
                 for x in complete_jsonl(log)) if log.exists() else 0
    live = complete_jsonl(ANSWER_SOURCE) if ANSWER_SOURCE.exists() else []
    print(json.dumps({"complete_answer_lines_now": len(live),
                      "snapshotted_answers": len(keys), "scored_items": finished,
                      "pending_snapshotted": len(keys) - finished, "judge_calls": calls,
                      "actual_tokens": tokens, "token_limit": TOKEN_LIMIT}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Read local scoring progress only.")
    parser.add_argument("--prepare", action="store_true", help="Snapshot complete answer lines without judging.")
    args = parser.parse_args()
    if args.status:
        print_status()
        return
    snap, questions, answers = read_selected()
    reused, pending = prepare_reuse(answers)
    if args.prepare:
        print_status()
        return
    probe_key = os.getenv("FULL300_PROBE_KEY")
    if probe_key:
        if probe_key not in pending:
            raise ValueError("Probe key is not pending")
        pending = [probe_key]
    print(f"Input {len(answers)}; cached/reused items {reused}; pending {len(pending)}.", flush=True)
    if not pending:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    judges = CachedJudges()
    if judges.settings["judge_primary"]["model"] != "openai/gpt-6-sol":
        raise ValueError("Primary judge model changed.")
    if "qwen3.7-plus" not in judges.settings["judge_secondary"]["model"]:
        raise ValueError("Secondary judge model changed.")
    evidence = old.Evidence(old.raw_pages())

    def process(key: str) -> str:
        row = answers[key]
        q = questions[row["id"]]
        item = {"id": row["id"], "model": row["model_name"], "type": q["type"],
                "page_type": q["page_type"], "source_url": row["source_url"],
                "question": q["question"], "reference_answer": q["reference_answer"],
                "keypoints": q["keypoints"], "answer": row["answer"], "metrics": {},
                "judgements": []}
        old.score_new(item, q, judges)
        old.score_grouped_citations(item, judges, evidence)
        item["rule_version"] = RULE_VERSION
        write_json(ITEMS / (key + ".json"), item)
        return key

    errors = []
    completed = 0
    for offset in range(0, len(pending), 64):
        if STOP.is_set() or QUOTA_STOP.is_set():
            break
        batch = pending[offset:offset + 64]
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(process, key): key for key in batch}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                    completed += 1
                except Exception as exc:
                    errors.append({"answer_id": key, "error_type": type(exc).__name__,
                                   "http_status": getattr(exc, "status_code", None)})
                if completed % 20 == 0 or STOP.is_set():
                    print(f"Completed {completed}/{len(pending)} new items; actual tokens {judges.actual_tokens()}.", flush=True)
        if STOP.is_set() or QUOTA_STOP.is_set():
            break
    summary = {"answer_count": len(answers), "cached_or_reused_items": reused,
               "new_items_completed": completed, "pending_items": len(pending) - completed,
               "actual_calls": sum(1 for _ in (OUT / "judge_calls.jsonl").open(encoding="utf-8"))
                   if (OUT / "judge_calls.jsonl").exists() else 0,
               "actual_tokens": judges.actual_tokens(), "token_limit": TOKEN_LIMIT,
               "token_limit_exceeded": STOP.is_set(), "http_402_stop": QUOTA_STOP.is_set(),
               "errors": errors}
    write_json(OUT / ("probe_status.json" if probe_key else "run_status.json"), summary)
    print(json.dumps(summary if probe_key else {k: v for k, v in summary.items() if k != "errors"}), flush=True)
    if STOP.is_set():
        raise SystemExit("Stopped after actual usage exceeded 40 million tokens.")
    if QUOTA_STOP.is_set():
        raise SystemExit("Stopped after OpenRouter returned HTTP 402.")
    if errors or completed != len(pending):
        raise SystemExit("Some items are incomplete; rerun to resume from cached judge calls.")


if __name__ == "__main__":
    main()
