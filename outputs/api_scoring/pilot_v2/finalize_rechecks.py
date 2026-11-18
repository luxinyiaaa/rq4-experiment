"""Resolve one inconsistent keypoint label and enforce the 30k citation cap."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import run_pilot_v2 as v2
from scripts import api_scoring_pilot as base
from scripts import metrics_v2 as s1


def main():
    base.OUT = v2.OUT
    base.norm_url = v2.norm_url
    judges = base.Judges()
    item_file = v2.ITEMS / "AS-073-622317da41_2__egi.json"
    item = json.loads(item_file.read_text(encoding="utf-8"))
    point = next(p for p in item["keypoints"] if "breakout working sessions" in p["text"])
    question = {"question": item["question"], "reference_answer": item["reference_answer"],
                "keypoints": [point], "type": item["type"]}
    response = s1._ragchecker_run("judge_primary", {"id": item["id"], "condition": "API",
        "answer": item["answer"], "passages": []}, question, "response2answer",
        settings=judges.settings["judge_primary"],
        mock_llm=lambda role, messages, **kw: v2.keypoint_call(judges, role, messages,
            item["answer"], item["question"], **kw))
    label = str(response["values"][0])
    audit = {"id": item["id"], "model": item["model"], "judge": "primary",
             "keypoint": point["text"], "initial_v2_label": "Contradiction",
             "recheck_label": label, "trace": response["trace"]}
    (v2.OUT / "keypoint_recheck.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    if label == "Entailment":
        judgment = next(j for j in item["judgements"] if j.get("metric") == "answer_correctness_completeness" and j.get("judge") == "primary")
        i = next(i for i,p in enumerate(item["keypoints"]) if p["text"] == point["text"])
        judgment["labels"][i] = label
        judgment["trace"][i] = response["trace"][0]
        judgment["targeted_recheck"] = "Rechecked because the first v2 label contradicted its own explanation and the explicit false-premise rule."
        result = s1.keypoint_summary(item["keypoints"],judgment["labels"])
        item["metrics"]["answer_correctness"]["primary"] = result["correct"]
        item["metrics"]["completeness"]["primary"] = result["completeness"]
        item_file.write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding="utf-8")
        changes_path = v2.OUT / "correctness_changes.csv"
        rows = list(csv.DictReader(changes_path.open(newline="",encoding="utf-8")))
        rows = [r for r in rows if not (r["id"]==item["id"] and r["model"]==item["model"] and r["judge"]=="primary")]
        with changes_path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=["id","model","judge","old_correct","new_correct","changed_keypoints","reason"])
            writer.writeheader();writer.writerows(rows)
    selected=json.loads((v2.OUT/"citation_disagreements.json").read_text(encoding="utf-8"))
    v2.rejudge_all_chunks(selected,v2.raw_page_map(),judges)
    items=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(v2.ITEMS.glob("*.json"))]
    base.report(items,judges)
    print("Keypoint recheck",label,"all-chunks sample",len(selected),flush=True)


if __name__=="__main__":
    main()
