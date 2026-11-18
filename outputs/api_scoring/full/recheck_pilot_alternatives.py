"""Rejudge pilot alternative labels using pilot-v2 correctness, without sample access."""
from __future__ import annotations

import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import api_scoring_pilot as base

OUT = Path(__file__).resolve().parent
PILOT = ROOT / "outputs/api_scoring/pilot/items"
PILOT_V2 = ROOT / "outputs/api_scoring/pilot_v2/items"
ITEMS = OUT / "items"
RECHECKS = OUT / "alternative_answer_rechecks.jsonl"
PROMPT_VERSION = "pilot-v2-correctness_primary-alt-recheck_2026-10-27"
LOCK = threading.Lock()


def read_rechecks():
    if not RECHECKS.exists():
        return {}
    return {(x["answer_id"], x["model"]): x for x in
            (json.loads(line) for line in RECHECKS.open(encoding="utf-8") if line.strip())}


def save_item(path, item):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def source_for(name):
    if (PILOT / f"{name}.json").exists():
        return "original pilot"
    if (PILOT_V2 / f"{name}.json").exists():
        return "pilot v2"
    return "full run"


def original_cases():
    data = json.loads((OUT / "alternative_answers_data.json").read_text(encoding="utf-8"))
    return data["headers"], data["rows"]


def judge_one(item, judges):
    payload = {"question": item["question"], "reference_answer": item["reference_answer"],
               "keypoints": item["keypoints"], "answer": item["answer"]}
    messages = [{"role": "system", "content": base.ALT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    for attempt in range(4):
        try:
            result = judges.call("judge_primary", messages, json_mode=True)["json"]
            if result.get("alternative_answer") not in {"yes", "no"} or not isinstance(result.get("reason"), str):
                raise ValueError("Invalid alternative-answer label")
            return result
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(3 * 2 ** attempt, 20))


def workbook_row(item):
    return {"answer_id": item["id"], "model": item["model"],
            "question": item["question"], "reference_answer": item["reference_answer"],
            "keypoints": "\n".join(("[DECISIVE] " if p.get("decisive") else "") + p["text"]
                                   for p in item["keypoints"]),
            "answer": item["answer"], "judge_reason": item["alternative_answer"].get("reason", ""),
            "my_decision": ""}


def main():
    headers, before = original_cases()
    pilot = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PILOT_V2.glob("*.json"))]
    eligible = [x for x in pilot if x["primary_abstention_label"] != "abstains"
                and not x["metrics"]["answer_correctness"]["primary"]]
    cached = read_rechecks()
    pending = [x for x in eligible if (x["id"], x["model"]) not in cached]
    print("pilot answers eligible for alternative recheck", len(eligible), "pending", len(pending), flush=True)
    if pending:
        base.OUT = OUT
        judges = base.Judges()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(judge_one, x, judges): x for x in pending}
            for future in as_completed(futures):
                item = futures[future]
                result = future.result()
                row = {"answer_id": item["id"], "model": item["model"],
                       "prompt_version": PROMPT_VERSION, "primary_correct": False,
                       "alternative_answer": result["alternative_answer"],
                       "reason": result["reason"]}
                with LOCK:
                    with RECHECKS.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cached[(item["id"], item["model"])] = row
                print("rechecked", len(cached), "/", len(eligible), flush=True)
    if len(cached) < len(eligible):
        raise RuntimeError("Alternative rechecks incomplete")
    eligible_pairs = {(x["id"], x["model"]) for x in eligible}
    updated = []
    for pv2 in pilot:
        path = ITEMS / f"{pv2['id']}__{pv2['model']}.json"
        if not path.exists():
            continue  # One pilot question is outside the complete four-model full cohort.
        full = json.loads(path.read_text(encoding="utf-8"))
        pair = (full["id"], full["model"])
        if pair in eligible_pairs:
            new = cached[pair]
            full["alternative_answer"] = {"alternative_answer": new["alternative_answer"],
                                          "reason": new["reason"]}
            full["alternative_answer_source"] = "full pilot recheck"
        else:
            full.pop("alternative_answer", None)
            full["alternative_answer_source"] = "not applicable under pilot v2 correctness/abstention"
        save_item(path, full)
        updated.append(full)
    # The existing workbook was already restricted to cases outside the blind sample.
    # Keep its full-run cases; rebuild its pilot cases solely from pilot item files.
    full_run_rows = [x for x in before if source_for(f"{x['answer_id']}__{x['model']}") == "full run"]
    new_pilot_rows = [workbook_row(x) for x in updated
                      if x.get("alternative_answer", {}).get("alternative_answer") == "yes"]
    after = sorted(full_run_rows + new_pilot_rows, key=lambda x: (x["answer_id"], x["model"]))
    (OUT / "alternative_answers_data.json").write_text(json.dumps({
        "headers": headers, "rows": after}, ensure_ascii=False), encoding="utf-8")
    initial = {(x["answer_id"], x["model"]): x for x in before}
    final = {(x["answer_id"], x["model"]): x for x in after}
    audit = []
    for pair in sorted(set(initial) | set(final)):
        name = f"{pair[0]}__{pair[1]}"
        source = source_for(name)
        if source == "full run":
            path = ITEMS / f"{name}.json"
            item = json.loads(path.read_text(encoding="utf-8"))
        else:
            item = next(x for x in updated if (x["id"],x["model"]) == pair)
        audit.append({"answer_id": pair[0], "model": pair[1],
                      "primary_correct": item["metrics"]["answer_correctness"]["primary"],
                      "secondary_correct": item["metrics"]["answer_correctness"]["secondary"],
                      "initial_alt_source": source if pair in initial else "not listed",
                      "initially_listed": pair in initial,
                      "rechecked_alt_label": cached[pair]["alternative_answer"] if pair in cached else "",
                      "finally_listed": pair in final})
    with (OUT / "alternative_answer_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0]))
        writer.writeheader(); writer.writerows(audit)
    # Refresh the prior full-run usage counter from its append-only response log.
    usage = json.loads((OUT / "usage.json").read_text(encoding="utf-8"))
    totals = {role: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
              for role in ("judge_primary", "judge_secondary")}
    for entry in (json.loads(line) for line in (OUT / "judge_calls.jsonl").open() if line.strip()):
        totals[entry["role"]]["calls"] += 1
        totals[entry["role"]]["input_tokens"] += entry["input_tokens"]
        totals[entry["role"]]["output_tokens"] += entry["output_tokens"]
    usage["new_full_run_calls_and_tokens"] = totals
    usage["pilot_alternative_recheck_count"] = len(eligible)
    (OUT / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    print("alternative list", len(before), "to", len(after), "cases; pilot items updated", len(updated), flush=True)


if __name__ == "__main__":
    main()
