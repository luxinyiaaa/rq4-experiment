"""Recompute final-experiment statistics and prepare blinded validation sheets."""
from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_index import _read_jsonl, _write_jsonl
from scripts.check_questions import _write_xlsx_minimal
from scripts.score_answers import _score_number

OUT = ROOT / "outputs" / "final"
V2 = OUT / "human_validation"
SEED = 42
BOOTSTRAPS = 2000


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap(values: list[float], seed: int = SEED) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), (BOOTSTRAPS, len(arr)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def newcombe_paired(left: list[bool], right: list[bool]) -> tuple[float, float, float]:
    """Newcombe (1998) method 10 paired-score interval (no continuity correction).

    Let a,b,c,d be the paired 2x2 table (both correct, left only, right only,
    neither); e=a+b, f=c+d, g=a+c, h=b+d, and
    phi=(a*d-b*c)/sqrt(e*f*g*h). The Wilson intervals for each marginal
    proportion are combined with the paired-correlation adjustment. For
    delta=p_right-p_left (so right is p1 and left is p2):
      lower = delta - sqrt((p1-L1)^2 + (U2-p2)^2 - 2*phi*(p1-L1)*(U2-p2))
      upper = delta + sqrt((U1-p1)^2 + (p2-L2)^2 - 2*phi*(U1-p1)*(p2-L2))
    No continuity correction is applied.
    """
    n = len(left)
    if n == 0:
        return 0.0, 0.0, 0.0
    a = sum(x and y for x, y in zip(left, right))
    b = sum(x and not y for x, y in zip(left, right))
    c = sum(not x and y for x, y in zip(left, right))
    dcell = sum(not x and not y for x, y in zip(left, right))
    e, f, g, h = a + b, c + dcell, a + c, b + dcell
    denom = math.sqrt(e * f * g * h)
    phi = (a * dcell - b * c) / denom if denom else 0.0
    p_left, p_right = e / n, g / n
    left_lo, left_hi = wilson(e, n)
    right_lo, right_hi = wilson(g, n)
    delta = p_right - p_left
    # The reported contrast is p_right - p_left, so right is method-10 p1.
    lower_var = (p_right - right_lo) ** 2 + (left_hi - p_left) ** 2 - 2 * phi * (p_right - right_lo) * (left_hi - p_left)
    upper_var = (right_hi - p_right) ** 2 + (p_left - left_lo) ** 2 - 2 * phi * (right_hi - p_right) * (p_left - left_lo)
    lo = delta - math.sqrt(max(0.0, lower_var))
    hi = delta + math.sqrt(max(0.0, upper_var))
    return delta, max(-1.0, lo), min(1.0, hi)


def metric(score: dict[str, Any], name: str) -> float | None:
    if name == "correctness":
        v = (score.get("ragchecker") or {}).get("correctness")
        return float(v) if isinstance(v, bool) else None
    if name == "correctness_v2":
        v = score.get("correctness_v2")
        return float(v) if isinstance(v, bool) else None
    if name == "completeness":
        v = (score.get("ragchecker") or {}).get("completeness")
        return float(v) if isinstance(v, (int, float)) else None
    if name == "faithfulness":
        return _score_number(score.get("faithfulness"))
    if name == "context_recall":
        return _score_number(score.get("context_recall"))
    if name == "flask":
        v = (score.get("flask") or {}).get("score")
        return float(v) if isinstance(v, (int, float)) else None
    raise KeyError(name)


def _condition_summary(scores: list[dict[str, Any]], conds: dict[str, dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    rows = []
    groups=[]
    for c in ("C1", "C2", "C3", "C4"):
        groups.append(("condition",c,"all",[s for s in scores if s["condition"]==c]))
        for typ in ("list_all","comparison","paraphrased","false_premise"):
            groups.append(("condition_by_type",c,typ,[s for s in scores if s["condition"]==c and s.get("type")==typ]))
    for pos in ("first","middle","last"):
        groups.append(("C2_by_gold_position","C2",pos,[s for s in scores if s["condition"]=="C2" and s.get("gold_position")==pos]))
    for table,c,subgroup,group in groups:
        for name in ("correctness", "correctness_v2", "abstention_rate", "gold_chunk_hit_at_10", "gold_page_hit_at_10"):
            if name in {"correctness","correctness_v2"}: vals = [metric(s, name) for s in group]
            elif name == "abstention_rate": vals = [float((s.get("abstention") or {}).get("label") == "abstains") for s in group if (s.get("abstention") or {}).get("label")]
            elif name == "gold_chunk_hit_at_10": vals = [float(conds[s["id"]].get("c4_gold_hit_rank") is not None) for s in group] if c == "C4" else []
            else: vals = [float((conds[s["id"]].get("c4_gold_page_hit_rank") or {}).get("gold") is not None) for s in group] if c == "C4" else []
            vals = [v for v in vals if v is not None]
            if name in {"correctness", "correctness_v2", "abstention_rate", "gold_chunk_hit_at_10", "gold_page_hit_at_10"} and vals:
                k = int(sum(vals)); lo, hi = wilson(k, len(vals))
                rows.append({"analysis_set": subset, "table": table, "condition": c, "subgroup": subgroup, "metric": name, "n": len(vals), "estimate": k / len(vals), "ci95_lower": lo, "ci95_upper": hi, "interval": "Wilson"})
        for name in ("completeness", "faithfulness", "context_recall", "flask"):
            if name == "context_recall" and c != "C4": continue
            vals = [metric(s, name) for s in group]
            vals = [v for v in vals if v is not None]
            mean, lo, hi = bootstrap(vals)
            rows.append({"analysis_set": subset, "table": table, "condition": c, "subgroup": subgroup, "metric": name, "n": len(vals), "estimate": mean, "ci95_lower": lo, "ci95_upper": hi, "interval": "bootstrap percentile"})
        if c == "C4" and table == "condition":
            labels = []
            for s in group:
                refusal = (s.get("abstention") or {}).get("label")
                correct = metric(s, "correctness")
                labels.append("abstained_or_partial" if refusal in {"abstains", "partially_answers"} else ("correct" if correct == 1 else "wrong_answer"))
            for label in ("correct", "wrong_answer", "abstained_or_partial"):
                k = labels.count(label); lo, hi = wilson(k, len(labels))
                rows.append({"analysis_set": subset, "table": "C4_three_class", "condition": c, "subgroup": label, "metric": "count_proportion", "n": len(labels), "count": k, "estimate": k / len(labels), "ci95_lower": lo, "ci95_upper": hi, "interval": "Wilson"})
            answered = [s for s in group if (s.get("abstention") or {}).get("label") == "answers"]
            good = sum(metric(s, "correctness") == 1 for s in answered)
            lo, hi = wilson(good, len(answered))
            rows.append({"analysis_set": subset, "table": "C4_three_class", "condition": c, "subgroup": "selective_accuracy_among_answers", "metric": "proportion", "n": len(answered), "count": good, "estimate": good / len(answered) if answered else None, "ci95_lower": lo, "ci95_upper": hi, "interval": "Wilson"})
    return rows


def build_stats(scores: list[dict[str, Any]], conditions: list[dict[str, Any]], pilot_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cond_map = {str(x["id"]): x for x in conditions}
    stats_rows = []
    paired_rows = []
    for label, data in (("all_44", scores), ("main_34", [s for s in scores if s["id"] not in pilot_ids])):
        stats_rows.extend(_condition_summary(data, cond_map, label))
        byid: dict[str, dict[str, dict[str, Any]]] = {}
        for s in data: byid.setdefault(s["id"], {})[s["condition"]] = s
        for right, comparison in (("C2", "robustness_C2_minus_C1"), ("C4", "retrieval_effect_C4_minus_C1")):
            pairs = [(g["C1"], g[right]) for g in byid.values() if "C1" in g and right in g]
            for correctness_metric in ("correctness","correctness_v2"):
                valid=[(metric(a,correctness_metric),metric(b,correctness_metric)) for a,b in pairs]
                valid=[(a,b_) for a,b_ in valid if a is not None and b_ is not None]
                leftb=[a==1 for a,_ in valid];rightb=[b_==1 for _,b_ in valid]
                if not valid: continue
                d,lo,hi=newcombe_paired(leftb,rightb)
                b=sum(a and not b_ for a,b_ in zip(leftb,rightb));c=sum((not a) and b_ for a,b_ in zip(leftb,rightb))
                p=float(binomtest(min(b,c),b+c,0.5).pvalue) if b+c else 1.0
                paired_rows.append({"analysis_set":label,"comparison":comparison,"metric":f"{correctness_metric}_difference","direction":f"{right} - C1","n_pairs":len(valid),"estimate":d,"ci95_lower":lo,"ci95_upper":hi,"interval":"Newcombe (1998) method 10","test":"McNemar exact","p_value":p,"discordant_C1_only":b,"discordant_right_only":c})
            for m in ("completeness", "faithfulness"):
                vals = [(metric(a, m), metric(b_, m)) for a, b_ in pairs]
                vals = [(a, b_) for a, b_ in vals if a is not None and b_ is not None]
                dif = [b_ - a for a, b_ in vals]
                mean, blo, bhi = bootstrap(dif)
                test = wilcoxon(dif, zero_method="wilcox", method="auto") if any(x != 0 for x in dif) else None
                paired_rows.append({"analysis_set": label, "comparison": comparison, "metric": f"{m}_difference", "direction": f"{right} - C1", "n_pairs": len(vals), "estimate": mean, "ci95_lower": blo, "ci95_upper": bhi, "interval": "bootstrap percentile", "test": "Wilcoxon signed-rank", "p_value": float(test.pvalue) if test else 1.0})
    c4cross=[]
    for label, data in (("all_44", scores), ("main_34", [s for s in scores if s["id"] not in pilot_ids])):
        for page_hit in (False, True):
            for chunk_hit in (False, True):
                group=[]
                for s in data:
                    if s["condition"] != "C4": continue
                    c=cond_map[s["id"]]
                    ph=(c.get("c4_gold_page_hit_rank") or {}).get("gold") is not None
                    ch=c.get("c4_gold_hit_rank") is not None
                    if (ph,ch)==(page_hit,chunk_hit): group.append(s)
                k=sum(metric(s,"correctness")==1 for s in group); lo,hi=wilson(k,len(group))
                c4cross.append({"analysis_set":label,"gold_page_hit":page_hit,"gold_chunk_hit":chunk_hit,"n":len(group),"correct":k,"correct_rate":k/len(group) if group else None,"wilson95_lower":lo,"wilson95_upper":hi})
    refusal_cross=[]
    for label,data in (("all_44",scores),("main_34",[s for s in scores if s["id"] not in pilot_ids])):
        for refusal in ("answers","partially_answers","abstains","missing"):
            for zero in (True,False,None):
                group=[]
                for s in data:
                    if s["condition"]!="C4": continue
                    recall=_score_number(s.get("context_recall")); iszero=None if recall is None else bool(np.isclose(recall,0.0))
                    if ((s.get("abstention") or {}).get("label") or "missing")==refusal and iszero is zero: group.append(s)
                refusal_cross.append({"analysis_set":label,"abstention_label":refusal,"context_recall_zero":"missing" if zero is None else str(zero).lower(),"count":len(group)})
    return pd.DataFrame(stats_rows),pd.DataFrame(paired_rows),pd.DataFrame(c4cross),pd.DataFrame(refusal_cross)


def update_manual_flags(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for s in scores:
        rag=s.get("ragchecker") or {}; refusal=(s.get("abstention") or {}).get("label")
        flask=(s.get("flask") or {}).get("score")
        rules=[]
        if isinstance(rag.get("correctness"),bool) and isinstance(flask,int) and ((rag["correctness"] and flask<=3) or (not rag["correctness"] and flask>=4)):
            rules.append("correctness_flask_disagreement")
        if refusal=="abstains" and rag.get("correctness") is True: rules.append("abstains_but_ragchecker_correct")
        if s.get("condition")=="C4" and refusal=="answers" and _score_number(s.get("context_recall"))==0: rules.append("C4_answers_context_recall_zero")
        s["needs_manual_check_rules"]=rules
        s["needs_manual_check"]=bool(rules)
    return scores


def load_old_blind() -> list[dict[str, str]]:
    path=OUT/"human_check_blind.xlsx"
    if not path.exists(): return []
    # Read the existing workbook directly so the hidden key is never opened.
    from zipfile import ZipFile
    from xml.etree import ElementTree as ET
    ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        root=ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows=[]
        for row in root.findall(".//m:sheetData/m:row",ns):
            vals=[]
            for cell in row.findall("m:c",ns): vals.append("".join(t.text or "" for t in cell.findall(".//m:t",ns)))
            rows.append(vals)
    if not rows: return []
    head=rows[0]
    return [dict(zip(head,r)) for r in rows[1:]]


def human_validation(scores: list[dict[str, Any]], questions: list[dict[str, Any]], answers: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> None:
    V2.mkdir(parents=True,exist_ok=True)
    qmap={q["id"]:q for q in questions}; amap={(a["id"],a["condition"]):a for a in answers}; cmap={c["id"]:c for c in conditions}
    old=load_old_blind()
    rng=random.Random(SEED)
    selected=[]
    for cond,n in (("C1",12),("C2",12),("C3",12),("C4",24)):
        candidates=[s for s in scores if s["condition"]==cond]
        # Weighted sampling without replacement gives false-premise items twice the selection weight.
        chosen=[]; remaining=candidates[:]
        while len(chosen)<n:
            weights=[2.0 if x.get("type")=="false_premise" else 1.0 for x in remaining]
            x=rng.choices(remaining,weights=weights,k=1)[0]; chosen.append(x); remaining.remove(x)
        selected.extend(chosen)
    # Retain old blinded IDs when an existing sampled answer also falls into the new draw.
    old_idmap={}
    for r in old:
        matches=[a for a in answers if a.get("question")==r.get("question") and a.get("answer")==r.get("answer")]
        if len(matches)==1: old_idmap[(matches[0]["id"],matches[0]["condition"])]=r.get("item_id")
    rng.shuffle(selected)
    blind=[];key=[]
    max_points=max((len(qmap[s["id"]].get("keypoints",[])) for s in selected),default=0)
    for i,s in enumerate(selected,1):
        q=qmap[s["id"]]; item=old_idmap.get((s["id"],s["condition"])) or f"HV{i:03d}"
        row={"item_id":item,"question":q["question"],"reference_answer":q["reference_answer"],"keypoint_definitions":"\n".join(f"K{j} {'[D]' if kp.get('decisive') else '[ ]'}: {kp.get('text','')}" for j,kp in enumerate(q.get("keypoints",[]),1)),"keypoint_rating_scale":"Enter present, partial, or absent in each keypoint column.","human_rating_guide":"human_correct: yes / no / not_applicable; human_abstention: answers / abstains / partially_answers; extra_wrong_content: yes / no."}
        for j in range(1,max_points+1): row[f"keypoint_{j}"]="" if j<=len(q.get("keypoints",[])) else "not_applicable"
        row.update({"human_correct":"","human_abstention":"","extra_wrong_content":"","note":""})
        row["answer"]=s["answer"]
        blind.append(row)
        key.append({"item_id":item,"question_id":s["id"],"condition":s["condition"],"type":s.get("type"),"automatic_correctness":(s.get("ragchecker") or {}).get("correctness"),"automatic_abstention":(s.get("abstention") or {}).get("label"),"extra_contradiction":s.get("extra_contradiction"),"correctness_v2":s.get("correctness_v2")})
    pd.DataFrame(blind).to_excel(V2/"answers_blind.xlsx",index=False) if _has_openpyxl() else _write_xlsx_minimal(pd.DataFrame(blind),V2/"answers_blind.xlsx")
    # Separate second-rater copy has no human input columns.
    blind_no=[{k:v for k,v in r.items() if k not in {"human_correct","human_abstention","extra_wrong_content","note"}} for r in blind]
    _write_xlsx_minimal(pd.DataFrame(blind_no),V2/"answers_blind_second_rater.xlsx")
    faith_candidates=[s for s in scores if s["condition"] in {"C1","C2","C4"} and (s.get("abstention") or {}).get("label")!="abstains"]
    faith=[]
    frng=random.Random(SEED); byc={c:[s for s in faith_candidates if s["condition"]==c] for c in ("C1","C2","C4")}
    for c,n in (("C1",7),("C2",7),("C4",6)): faith.extend(frng.sample(byc[c],min(n,len(byc[c]))))
    _write_jsonl(V2/"faithfulness_sample.jsonl",[{"id":s["id"],"condition":s["condition"]} for s in faith])
    frows=[]
    for s in faith:
        a=amap[(s["id"],s["condition"])]; raw=s.get("raw_outputs",{}).get("ragas_faithfulness",{})
        statements=raw.get("statements",[]) if isinstance(raw,dict) else []
        verdicts=raw.get("faithfulness_results",[]) if isinstance(raw,dict) else []
        for j,statement in enumerate(statements or ["(Ragas statement details were not retained; answer-level score only.)"]):
            frows.append({"item_id":f"FV-{s['id']}-{s['condition']}-{j+1}","question":s["question"],"answer":s["answer"],"passages_shown":"\n\n".join(p.get("text","") for p in a.get("passages",[])),"ragas_statement":statement,"ragas_judgment":verdicts[j] if j<len(verdicts) else "Not retained","ragas_faithfulness_score":_score_number(s.get("faithfulness")),"human_agreement":"","human_rating_guide":"Enter agree or disagree."})
        key.append({"item_id":f"FV-{s['id']}-{s['condition']}","form":"faithfulness","question_id":s["id"],"condition":s["condition"],"type":s.get("type"),"faithfulness_score":_score_number(s.get("faithfulness"))})
    faith_df=pd.DataFrame(frows)
    if not (V2/"faithfulness_rows.jsonl").exists():
        _write_xlsx_minimal(faith_df,V2/"faithfulness_check.xlsx")
        faith_no=faith_df.drop(columns=["human_agreement"],errors="ignore");_write_xlsx_minimal(faith_no,V2/"faithfulness_check_second_rater.xlsx")
    else:
        detailed=_read_jsonl(V2/"faithfulness_rows.jsonl")
        score_lookup={(s["id"],s["condition"]):_score_number(s.get("faithfulness")) for s in scores}
        for row in detailed:
            row.setdefault("human_rating_guide","Enter agree or disagree.")
            parts=str(row.get("item_id","")).split("-")
            if len(parts)>=4: row["ragas_faithfulness_score"]=score_lookup.get((parts[1],parts[2]))
        _write_jsonl(V2/"faithfulness_rows.jsonl",detailed)
        detail_df=pd.DataFrame(detailed);_write_xlsx_minimal(detail_df,V2/"faithfulness_check.xlsx")
        _write_xlsx_minimal(detail_df.drop(columns=["human_agreement"],errors="ignore"),V2/"faithfulness_check_second_rater.xlsx")
    c4=[s for s in scores if s["condition"]=="C4"]; crng=random.Random(SEED); chosen=crng.sample(c4,min(20,len(c4))); cr=[]
    for s in chosen:
        a=amap[(s["id"],"C4")]
        cr.append({"item_id":f"CR-{s['id']}","question":s["question"],"reference_answer":s["reference_answer"],"top_10_passages":"\n\n".join(f"[{p.get('passage_id')}] {p.get('url')}\n{p.get('text','')}" for p in a.get("passages",[])),"human_context_recall":"","human_rating_guide":"Choose all, part, or none."})
        key.append({"item_id":f"CR-{s['id']}","form":"context_recall","question_id":s["id"],"condition":"C4","type":s.get("type"),"context_recall":_score_number(s.get("context_recall"))})
    crdf=pd.DataFrame(cr);_write_xlsx_minimal(crdf,V2/"context_recall_check.xlsx");_write_xlsx_minimal(crdf.drop(columns=["human_context_recall"]),V2/"context_recall_check_second_rater.xlsx")
    # Create the key, but never read it back in this run.
    pd.DataFrame(key).to_csv(V2/"answers_key.csv",index=False)


def _has_openpyxl() -> bool:
    try: import openpyxl  # noqa: F401
    except ImportError: return False
    return True


def main() -> None:
    score_path=OUT/"scores_final_revised.jsonl"
    scores=_read_jsonl(score_path if score_path.exists() else OUT/"scores_final.jsonl"); conditions=_read_jsonl(OUT/"conditions_final.jsonl"); questions=_read_jsonl(OUT/"questions_final.jsonl"); answers=_read_jsonl(OUT/"answers_final.jsonl")
    pilot_path=ROOT/"outputs"/"pilot"/"questions_reviewed.jsonl"
    pilot_ids={r["id"] for r in _read_jsonl(pilot_path)}
    scores=update_manual_flags(scores)
    _write_jsonl(score_path,scores)
    summary,paired,cross,refusal_cross=build_stats(scores,conditions,pilot_ids)
    summary.to_csv(OUT/"score_summary_revised.csv",index=False); summary.to_csv(OUT/"score_summary_final.csv",index=False); summary.to_csv(OUT/"score_summary_basic.csv",index=False)
    paired.to_csv(OUT/"paired_tests_revised.csv",index=False); paired.to_csv(OUT/"paired_tests_final.csv",index=False)
    cross.to_csv(OUT/"c4_retrieval_cross_tab.csv",index=False); refusal_cross.to_csv(OUT/"c4_abstention_contextrecall_revised.csv",index=False); refusal_cross.to_csv(OUT/"c4_abstention_contextrecall_final.csv",index=False)
    pmap={}
    for s in scores: pmap.setdefault(s["id"],{})[s["condition"]]=s
    diffs=[]
    for qid,g in pmap.items():
        for c,name in (("C2","robustness_C2_minus_C1"),("C4","retrieval_effect_C4_minus_C1")):
            if "C1" not in g or c not in g: continue
            for m in ("correctness","correctness_v2","completeness","faithfulness","flask"):
                a,b=metric(g["C1"],m),metric(g[c],m)
                if a is not None and b is not None: diffs.append({"id":qid,"comparison":name,"metric":m,"direction":f"{c} - C1","difference":b-a})
    pd.DataFrame(diffs).to_csv(OUT/"paired_drops_basic.csv",index=False)
    needs=[{"id":s["id"],"condition":s["condition"],"type":s.get("type"),"rules":"; ".join(s["needs_manual_check_rules"]),"question":s["question"],"answer":s["answer"],"correctness":(s.get("ragchecker") or {}).get("correctness"),"correctness_v2":s.get("correctness_v2"),"flask":(s.get("flask") or {}).get("score"),"abstention":(s.get("abstention") or {}).get("label"),"context_recall":_score_number(s.get("context_recall"))} for s in scores if s["needs_manual_check"]]
    pd.DataFrame(needs).to_csv(OUT/"needs_manual_check_revised.csv",index=False);pd.DataFrame(needs).to_csv(OUT/"needs_manual_check_final.csv",index=False)
    _write_xlsx_minimal(pd.DataFrame(needs),OUT/"manual_check_disagreements_revised.xlsx");_write_xlsx_minimal(pd.DataFrame(needs),OUT/"manual_check_disagreements.xlsx")
    score_by_id={}
    for s in scores: score_by_id.setdefault(s["id"],{})[s["condition"]]=s
    cond_map={str(c["id"]):c for c in conditions}
    review_rows=[]
    for q in questions:
        row={"id":q["id"],"type":q.get("type"),"question":q.get("question","")}
        for c in ("C1","C2","C3","C4"):
            s=score_by_id.get(q["id"],{}).get(c)
            if not s: continue
            row[f"{c}_correctness"]=(s.get("ragchecker") or {}).get("correctness")
            row[f"{c}_correctness_v2"]=s.get("correctness_v2")
            row[f"{c}_extra_contradiction"]=s.get("extra_contradiction")
            row[f"{c}_completeness"]=(s.get("ragchecker") or {}).get("completeness")
            row[f"{c}_FLASK"]=(s.get("flask") or {}).get("score")
            row[f"{c}_abstention"]=(s.get("abstention") or {}).get("label")
            row[f"{c}_faithfulness"]=_score_number(s.get("faithfulness"))
            row[f"{c}_needs_manual_check"]="; ".join(s.get("needs_manual_check_rules",[]))
            if c=="C4": row["C4_ContextRecall"]=_score_number(s.get("context_recall"))
        cr=cond_map[q["id"]]
        row["C4_gold_chunk_rank"]=cr.get("c4_gold_hit_rank")
        row["C4_gold_page_rank"]=(cr.get("c4_gold_page_hit_rank") or {}).get("gold")
        row["C4_partner_page_rank"]=(cr.get("c4_gold_page_hit_rank") or {}).get("partner")
        review_rows.append(row)
    _write_xlsx_minimal(pd.DataFrame(review_rows),OUT/"scores_review_final.xlsx")
    human_validation(scores,questions,answers,conditions)
    closed_extra={r["id"]:r for r in _read_jsonl(OUT/"closed_book_missing_checks.jsonl")} if (OUT/"closed_book_missing_checks.jsonl").exists() else {}
    closed_rows=[]
    for q in questions:
        checked=closed_extra.get(q["id"],q)
        cval=checked.get("closed_book_correct") if checked.get("closed_book_status")=="checked" else None
        row={"id":q["id"],"type":q.get("type"),"closed_book_status":checked.get("closed_book_status"),"closed_book_correct":cval}
        for c in ("C1","C2","C3","C4"):
            s=next((x for x in scores if x["id"]==q["id"] and x["condition"]==c),None)
            row[f"{c}_correct"]=metric(s,"correctness") if s else None
        closed_rows.append(row)
    closed_frame=pd.DataFrame(closed_rows);closed_frame.to_csv(OUT/"closed_book_correctness_crosscheck.csv",index=False)
    print("Revised condition summaries (Wilson intervals for proportions; bootstrap for means):")
    print(summary[summary.table=="condition"].to_string(index=False))
    print("Condition x question type:");print(summary[summary.table=="condition_by_type"].to_string(index=False))
    print("C2 by gold position:");print(summary[summary.table=="C2_by_gold_position"].to_string(index=False))
    print("C4 three-class outcomes and selective accuracy:");print(summary[summary.table=="C4_three_class"].to_string(index=False))
    print("Paired differences, C2-C1 robustness and C4-C1 retrieval effect:");print(paired.to_string(index=False))
    print("C4 retrieval cross-tab:");print(cross.to_string(index=False))
    print("C4 abstention x ContextRecall=0:");print(refusal_cross.to_string(index=False))
    completed=[r for r in closed_rows if r["closed_book_status"]=="checked"]
    closed_correct=[r for r in completed if r["closed_book_correct"] is True]
    print(f"Closed-book correct: {len(closed_correct)}/{len(completed)} completed ({len(closed_rows)} total).")
    for c in ("C1","C2","C3","C4"):
        vals=[bool(r[f"{c}_correct"]) for r in closed_correct if r[f"{c}_correct"] is not None]
        lo,hi=wilson(sum(vals),len(vals))
        print(f"C{c[-1]} correctness among closed-book-correct questions: {sum(vals)}/{len(vals)}; Wilson 95% CI {lo}–{hi}")
    print(f"Needs manual check: {len(needs)} rows; files saved under {OUT}.")
    print(pd.DataFrame(needs)[["id","condition","rules"]].to_string(index=False) if needs else "None")
    print(f"Human validation sheets written under {V2}; the key was written and not reopened.")


if __name__=="__main__": main()
