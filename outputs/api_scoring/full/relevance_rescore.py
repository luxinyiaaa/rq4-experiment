"""Archive and rerun metric 3 for the full API cohort and S1."""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from outputs.api_scoring.full import run_full
from scripts import api_scoring_pilot as base

API = ROOT / "outputs/api_scoring/full"
S1 = ROOT / "outputs/metrics_v2"
PROMPT = """Rate response relevance on a scale of 1 to 5. Relevance measures only how much of the answer is about something other than the thing the question asks about. Do not consider correctness, completeness, length, or whether the answer abstains; other metrics measure these.
The thing the question asks about is the specific record, project, organisation, event or concept named in the question, including its parts and properties.
Sort the content of the answer into three kinds. Count each sentence, bullet point or table row as one unit.
(A) Addresses the question: content that answers what was asked, whether right or wrong; correcting a false premise; saying that the information was not found or is not in the documentation; explanations, background or further facts about the thing the question asks about; one sentence saying which organisation or project it belongs to.
(B) Ignore: source lines, URLs and citation markers; disclaimers such as 'unverified' or 'based on general knowledge'; offers of further help, requests for a link or source, and pointers to where the information might be found; headings and formatting.
(C) About something else: content about a different record, project, organisation, event or topic, including related projects and other activities of the parent organisation; general background about the field that is not about the thing asked; descriptions of what unrelated pages contain.
If unsure whether a unit is A or C, ask whether it is still about the thing the question asks about; if yes, it is A.
Score: 5 = no C; 4 = one or two units of C; 3 = three or more units of C, about as much as A; 2 = clearly more C than A, and A is at most one sentence; 1 = no A (the answer addresses a different question).
Give the score and a one-sentence reason that names the C content, if there is any.
Return one JSON object with keys score (integer) and reason (one sentence)."""


def archive():
    api_archive = API / "relevance_v1_archive"
    s1_archive = S1 / "relevance_v1_archive"
    api_archive.mkdir(exist_ok=True)
    s1_archive.mkdir(exist_ok=True)
    if not (api_archive / "items").exists():
        shutil.copytree(API / "items", api_archive / "items")
        for name in ("main_table.md", "report.md", "summary.csv", "model_comparisons.csv",
                     "judge_agreement.csv", "family_bias.csv", "usage.json", "judge_calls.jsonl"):
            shutil.copy2(API / name, api_archive / name)
    if not (s1_archive / "judgements.jsonl").exists():
        shutil.copy2(S1 / "items/judgements.jsonl", s1_archive / "judgements.jsonl")
        shutil.copy2(S1 / "items/answer_metrics_wide.csv", s1_archive / "answer_metrics_wide.csv")
        for name in ("main_table_S1.md", "metric_by_condition_S1.csv", "robustness_paired_S1.csv",
                     "retrieval_loss_S1.csv", "judge_agreement.csv", "metric_by_type_C4_S1.csv",
                     "metrics_v2_summary.md"):
            shutil.copy2(S1 / "summary" / name, s1_archive / name)


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run_stage(stage, rows):
    out = API if stage == "api" else S1
    progress = out / "relevance_v2_progress"
    progress.mkdir(exist_ok=True)
    judges = run_full.LimitedJudges()
    base.OUT = progress
    jobs = []
    for row in rows:
        key = row["id"] + ("__" + row["model"] if stage == "api" else "__" + row["condition"])
        for judge, role in (("primary", "judge_primary"), ("secondary", "judge_secondary")):
            path = progress / (key + "__" + judge + ".json")
            if not path.exists():
                jobs.append((row, judge, role, path))
    def work(job):
        row, judge, role, path = job
        response = judges.call(role, [{"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps({"question": row["question"], "answer": row["answer"]}, ensure_ascii=False)}],
            json_mode=True, temperature=0)["json"]
        if type(response.get("score")) is not int or response["score"] not in range(1, 6) or not isinstance(response.get("reason"), str):
            raise ValueError("Invalid relevance label")
        write_json(path, {"score": response["score"], "reason": response["reason"]})
        return path.name
    print(f"{stage}: {len(rows)} answers, {len(jobs)} judge calls pending", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 40 == 0 or n == len(futures):
                print(f"{stage}: {n}/{len(futures)} calls complete", flush=True)
    for row in rows:
        key = row["id"] + ("__" + row["model"] if stage == "api" else "__" + row["condition"])
        for judge in ("primary", "secondary"):
            row["metrics"]["response_relevance"][judge] = json.loads((progress / (key + "__" + judge + ".json")).read_text(encoding="utf-8"))
        if stage == "api":
            write_json(API / "items" / (key + ".json"), row)
    if stage == "s1":
        target = S1 / "items/judgements.jsonl"
        temp = target.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        temp.replace(target)


def main():
    archive()
    snap = json.loads((API / "input_snapshot.json").read_text(encoding="utf-8"))
    api = [json.loads((API / "items" / (key + ".json")).read_text(encoding="utf-8")) for key in snap["answer_ids"]]
    if len(api) != 376:
        raise ValueError("Expected 376 API answers")
    run_stage("api", api)
    questions = {x["id"]: x for x in base.read_jsonl(ROOT / "outputs/final/questions_final.jsonl")}
    s1 = base.read_jsonl(S1 / "items/judgements.jsonl")
    if len(s1) != 176:
        raise ValueError("Expected 176 S1 answers")
    for row in s1:
        row["question"] = questions[row["id"]]["question"]
    run_stage("s1", s1)
    # The S1 source record does not store the question; retain its original schema.
    for row in s1:
        del row["question"]
    target = S1 / "items/judgements.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in s1), encoding="utf-8")


if __name__ == "__main__":
    main()
