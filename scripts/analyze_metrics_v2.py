"""Produce aggregate RQ4 metrics-v2 tables from cached item judgements."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics_v2"
ITEMS = OUT / "items"
SUMMARY = OUT / "summary"
CONDITIONS = ("C4", "C1", "C2", "C3")
JUDGES = ("primary", "secondary")
BOOTSTRAPS = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if not n:
        return None, None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap(values: list[float], rng: np.random.Generator) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    a = np.asarray(values, dtype=float)
    draws = a[rng.integers(0, len(a), (BOOTSTRAPS, len(a)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(a.mean()), float(lo), float(hi)


def newcombe_paired(left: list[bool], right: list[bool]) -> tuple[float, float, float]:
    """Newcombe (1998) method 10, paired Wilson intervals, phi adjustment, no continuity correction."""
    n = len(left)
    a = sum(x and y for x, y in zip(left, right))
    b = sum(x and not y for x, y in zip(left, right))
    c = sum(not x and y for x, y in zip(left, right))
    d = sum(not x and not y for x, y in zip(left, right))
    e, f, g, h = a + b, c + d, a + c, b + d
    denominator = math.sqrt(e * f * g * h)
    phi = (a * d - b * c) / denominator if denominator else 0.0
    p_left, p_right = e / n, g / n
    ll, lu = wilson(e, n)
    rl, ru = wilson(g, n)
    delta = p_right - p_left
    lower_var = (p_right - rl) ** 2 + (lu - p_left) ** 2 - 2 * phi * (p_right - rl) * (lu - p_left)
    upper_var = (ru - p_right) ** 2 + (p_left - ll) ** 2 - 2 * phi * (ru - p_right) * (p_left - ll)
    return delta, max(-1.0, delta - math.sqrt(max(0.0, lower_var))), min(1.0, delta + math.sqrt(max(0.0, upper_var)))


def get_value(record: dict[str, Any], metric: str, judge: str, component: str | None = None) -> Any:
    value = record.get("metrics", {}).get(metric, {}).get(judge)
    if value is None or (isinstance(value, dict) and value.get("status") == "excluded"):
        return None
    if metric == "appropriate_abstention":
        measure = value.get("measure", {}) if isinstance(value, dict) else {}
        if measure.get("status") == "excluded":
            return None
        if component and measure.get("name") != component:
            return None
        return measure.get("value")
    if metric == "answer_correctness":
        return value if isinstance(value, bool) else None
    if metric == "evidence_faithfulness":
        return value.get("value") if isinstance(value, dict) and value.get("status") != "excluded" else None
    if metric == "factual_accuracy":
        return value.get("supported_share") if isinstance(value, dict) else None
    if metric == "response_relevance":
        return value.get("score") if isinstance(value, dict) else None
    if metric == "instruction_fulfilment":
        return value.get("DRFR") if isinstance(value, dict) else None
    if metric in {"citation_support", "citation_completeness"}:
        return value.get("score") if isinstance(value, dict) else None
    if isinstance(value, dict):
        return value.get("value")
    return value


def factual_unknown_share(record: dict[str, Any], judge: str) -> float | None:
    x = record.get("metrics", {}).get("factual_accuracy", {}).get(judge)
    return x.get("cannot_be_determined_share") if isinstance(x, dict) else None


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def bootstrap_row(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    mean, lo, hi = bootstrap(values, rng)
    return {"n_applicable": len(values), "estimate": mean, "ci95_lower": lo, "ci95_upper": hi, "interval": "bootstrap percentile"}


def condition_tables(records: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(42)
    metric_specs = [
        ("Content quality", "Answer correctness", "answer_correctness", "binary", None),
        ("Content quality", "Evidence faithfulness", "evidence_faithfulness", "mean", None),
        ("Content quality", "Factual accuracy: supported among decidable claims", "factual_accuracy", "mean", None),
        ("Content quality", "Factual accuracy: cannot be determined among all claims", "factual_accuracy_unknown", "mean", None),
        ("Task fulfilment", "Response relevance", "response_relevance", "mean", None),
        ("Task fulfilment", "Instruction fulfilment", "instruction_fulfilment", "mean", None),
        ("Task fulfilment", "Completeness", "completeness", "mean", None),
        ("Evidence and citation quality", "Retrieved-evidence relevance", "retrieved_evidence_relevance", "mean", None),
        ("Evidence and citation quality", "Retrieved-evidence coverage", "retrieved_evidence_coverage", "mean", None),
        ("Evidence and citation quality", "Citation support", "citation_support", "mean", None),
        ("Evidence and citation quality", "Citation completeness", "citation_completeness", "mean", None),
        ("Reliability", "Appropriate abstention: abstention rate", "appropriate_abstention", "binary", "abstention_rate"),
        ("Reliability", "Appropriate abstention: false-abstention rate", "appropriate_abstention", "binary", "false_abstention_rate"),
    ]
    rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for dimension, label, metric, kind, component in metric_specs:
        for judge in JUDGES:
            row: dict[str, Any] = {"dimension": dimension, "metric": label, "judge": judge}
            for condition in CONDITIONS:
                subset = [r for r in records if r["condition"] == condition]
                vals = []
                for record in subset:
                    if metric == "factual_accuracy_unknown":
                        v = factual_unknown_share(record, judge)
                    else:
                        v = get_value(record, metric, judge, component)
                    if v is not None:
                        vals.append(v)
                if kind == "binary":
                    k = sum(bool(v) for v in vals)
                    lo, hi = wilson(k, len(vals))
                    data = {"n_applicable": len(vals), "successes": k, "estimate": k / len(vals) if vals else None,
                            "ci95_lower": lo, "ci95_upper": hi, "interval": "Wilson"}
                else:
                    data = bootstrap_row([float(v) for v in vals], rng)
                row.update({f"{condition}_{key}": value for key, value in data.items()})
            rows.append(row)
    for dimension, label, metric, kind, component in metric_specs:
        for judge in JUDGES:
            for qtype in sorted({str(questions[str(r["id"])].get("type", "")) for r in records if r["condition"] == "C4"}):
                subset = [r for r in records if r["condition"] == "C4" and str(questions[str(r["id"])].get("type", "")) == qtype]
                vals = []
                for record in subset:
                    v = factual_unknown_share(record, judge) if metric == "factual_accuracy_unknown" else get_value(record, metric, judge, component)
                    if v is not None: vals.append(v)
                if kind == "binary":
                    k=sum(bool(v) for v in vals); lo,hi=wilson(k,len(vals))
                    data={"n_applicable":len(vals),"successes":k,"estimate":k/len(vals) if vals else None,"ci95_lower":lo,"ci95_upper":hi,"interval":"Wilson"}
                else: data=bootstrap_row([float(v) for v in vals],rng)
                type_rows.append({"dimension":dimension,"metric":label,"judge":judge,"type":qtype,**data})
    return rows, type_rows


def holm(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records: by_question[str(r["id"])][str(r["condition"])] = r
    metrics = ["answer_correctness", "evidence_faithfulness", "factual_accuracy", "response_relevance",
               "instruction_fulfilment", "completeness", "citation_support", "citation_completeness"]
    output=[]
    for judge in JUDGES:
        for right in ("C2", "C4"):
            pending=[]
            for metric in metrics:
                pairs=[]
                for group in by_question.values():
                    if "C1" not in group or right not in group: continue
                    left_v=get_value(group["C1"],metric,judge); right_v=get_value(group[right],metric,judge)
                    if left_v is not None and right_v is not None: pairs.append((left_v,right_v))
                if not pairs: continue
                left=[x for x,_ in pairs]; rv=[y for _,y in pairs]
                if metric == "answer_correctness":
                    lb=[bool(x) for x in left]; rb=[bool(y) for y in rv]
                    estimate,lo,hi=newcombe_paired(lb,rb)
                    b=sum(x and not y for x,y in zip(lb,rb)); c=sum(not x and y for x,y in zip(lb,rb))
                    p=binomtest(min(b,c),b+c,0.5,alternative="two-sided").pvalue if b+c else 1.0
                    result={"estimate":estimate,"ci95_lower":lo,"ci95_upper":hi,"interval":"Newcombe (1998) method 10","test":"Exact McNemar","p_value":float(p)}
                else:
                    diffs=np.asarray([float(y)-float(x) for x,y in pairs])
                    rng=np.random.default_rng(42)
                    draws=diffs[rng.integers(0,len(diffs),(BOOTSTRAPS,len(diffs)))].mean(axis=1)
                    lo,hi=np.quantile(draws,[.025,.975])
                    try: p=float(wilcoxon(diffs,zero_method="wilcox",alternative="two-sided").pvalue) if np.any(diffs) else 1.0
                    except ValueError: p=1.0
                    result={"estimate":float(np.mean(diffs)),"ci95_lower":float(lo),"ci95_upper":float(hi),"interval":"paired bootstrap percentile","test":"Wilcoxon signed-rank","p_value":p}
                pending.append({"judge":judge,"comparison":f"{right} - C1","metric":metric,"n_pairs":len(pairs),**result})
            adjusted=holm([r["p_value"] for r in pending])
            for row,p_adj in zip(pending,adjusted): row["p_holm_adjusted"]=p_adj; output.append(row)
    return output


def _kappa(left: list[Any], right: list[Any], weighted: bool = False) -> float | None:
    if not left: return None
    levels=sorted(set(left)|set(right),key=str); n=len(left)
    observed=0.0; expected=0.0
    lc=Counter(left); rc=Counter(right)
    for i,a in enumerate(levels):
        for j,b in enumerate(levels):
            weight=((i-j)/(len(levels)-1))**2 if weighted and len(levels)>1 else (0.0 if i==j else 1.0)
            obs=sum(x==a and y==b for x,y in zip(left,right))/n
            exp=(lc[a]/n)*(rc[b]/n)
            observed+=weight*obs; expected+=weight*exp
    return 1-observed/expected if expected else None


def agreement_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics=["answer_correctness","evidence_faithfulness","factual_accuracy","factual_accuracy_unknown","response_relevance","instruction_fulfilment","completeness","retrieved_evidence_relevance","retrieved_evidence_coverage","citation_support","citation_completeness","appropriate_abstention"]
    rows=[]
    for metric in metrics:
        pairs=[]
        for r in records:
            if metric=="appropriate_abstention":
                p=r.get("metrics",{}).get(metric,{}).get("primary",{}).get("label")
                s=r.get("metrics",{}).get(metric,{}).get("secondary",{}).get("label")
            elif metric == "factual_accuracy_unknown":
                p=factual_unknown_share(r,"primary"); s=factual_unknown_share(r,"secondary")
            else:
                p=get_value(r,metric,"primary"); s=get_value(r,metric,"secondary")
            if p is not None and s is not None: pairs.append((p,s))
        if not pairs: continue
        pvals=[x for x,_ in pairs]; svals=[y for _,y in pairs]
        numeric=all(isinstance(x,(int,float,bool)) for x in pvals+svals)
        rho=float(spearmanr(pvals,svals).statistic) if numeric and len(set(pvals))>1 and len(set(svals))>1 else None
        if metric in {"answer_correctness","appropriate_abstention"}: rho=None
        if metric == "appropriate_abstention":
            primary_best = secondary_best = 0
            for r in records:
                p=r.get("metrics",{}).get(metric,{}).get("primary",{}).get("label")
                s=r.get("metrics",{}).get(metric,{}).get("secondary",{}).get("label")
                primary_best += (p=="abstains") if r["condition"]=="C3" else (p!="abstains")
                secondary_best += (s=="abstains") if r["condition"]=="C3" else (s!="abstains")
            pbest=primary_best/len(records); sbest=secondary_best/len(records)
        else:
            best=True if metric=="answer_correctness" else 5 if metric=="response_relevance" else 1.0
            pbest=sum(x==best for x in pvals)/len(pvals); sbest=sum(x==best for x in svals)/len(svals)
        rows.append({"metric":metric,"n_pairs":len(pairs),"exact_agreement":sum(x==y for x,y in pairs)/len(pairs),
                     "cohen_kappa":_kappa(pvals,svals) if metric in {"answer_correctness","appropriate_abstention"} else None,"quadratic_weighted_kappa":_kappa(pvals,svals,weighted=True) if metric=="response_relevance" else None,
                     "spearman":rho,"agreement_within_one_point":sum(abs(float(x)-float(y))<=1 for x,y in pairs)/len(pairs) if metric=="response_relevance" else None,
                     "primary_best_share":pbest,"secondary_best_share":sbest,
                     "note":"Kappa can be unstable when nearly all answers receive the same value."})
    return rows


def discrimination_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics=["answer_correctness","evidence_faithfulness","factual_accuracy","factual_accuracy_unknown","response_relevance","instruction_fulfilment","completeness","retrieved_evidence_relevance","retrieved_evidence_coverage","citation_support","citation_completeness","appropriate_abstention"]
    rows=[]
    for metric in metrics:
        for judge in JUDGES:
            for condition in CONDITIONS:
                vals=[]
                for r in records:
                    if r["condition"]!=condition: continue
                    if metric=="appropriate_abstention":
                        v=r.get("metrics",{}).get(metric,{}).get(judge,{})
                        v=v.get("measure",{}).get("value") if v.get("measure",{}).get("status")!="excluded" else None
                    elif metric=="factual_accuracy_unknown": v=factual_unknown_share(r,judge)
                    else: v=get_value(r,metric,judge)
                    if v is not None: vals.append(float(v))
                if not vals: continue
                best=(1.0 if condition=="C3" else 0.0) if metric=="appropriate_abstention" else (5.0 if metric=="response_relevance" else 1.0)
                worst=0.0 if best==1.0 else 1.0
                if metric=="response_relevance": worst=1.0
                rows.append({"metric":metric,"judge":judge,"condition":condition,"n":len(vals),"best_share":sum(v==best for v in vals)/len(vals),"worst_share":sum(v==worst for v in vals)/len(vals),"standard_deviation":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0})
    return rows


def c4_attribution(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],list[dict[str, Any]]]:
    rmap={(r["id"],r["condition"]):r for r in records}
    item_rows=[]
    for r in records:
        if r["condition"]!="C4": continue
        out={"id":r["id"]}
        for judge in JUDGES:
            correct=get_value(r,"answer_correctness",judge)
            if correct is not False: out[f"category_{judge}"]=None;continue
            retrieval=next((j for j in r.get("judgements",[]) if j.get("metric")=="retrieval" and j.get("judge")==judge),{})
            all_got=retrieval.get("decisive_keypoints_all_retrieved")
            abst=r.get("metrics",{}).get("appropriate_abstention",{}).get(judge,{}).get("label")
            category="retrieval_failure" if not all_got else "false_abstention" if abst=="abstains" else "generation_failure"
            out[f"category_{judge}"]=category
        item_rows.append(out)
    summary=[]
    for judge in JUDGES:
        judge_items=[x for x in item_rows if x.get(f"category_{judge}")]
        for category in ("retrieval_failure","false_abstention","generation_failure"):
            group=[x for x in judge_items if x[f"category_{judge}"]==category]
            metrics=[]
            for x in group:
                r=rmap[(x["id"],"C4")]
                metrics.append((get_value(r,"evidence_faithfulness",judge),get_value(r,"factual_accuracy",judge),get_value(r,"citation_support",judge)))
            summary.append({"judge":judge,"category":category,"count":len(group),
                            "mean_faithfulness":_mean([x[0] for x in metrics]),"mean_factual_accuracy":_mean([x[1] for x in metrics]),
                            "mean_citation_support":_mean([x[2] for x in metrics])})
    changed=sum(x.get("category_primary")!=x.get("category_secondary") for x in item_rows if x.get("category_primary") or x.get("category_secondary"))
    for row in summary: row["answers_changing_category_under_secondary"]=changed
    return summary,item_rows


def _mean(values: list[Any]) -> float | None:
    clean=[float(v) for v in values if isinstance(v,(int,float))]
    return float(np.mean(clean)) if clean else None


def correct_incorrect(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields=["evidence_faithfulness","factual_accuracy","factual_accuracy_unknown","response_relevance","instruction_fulfilment","completeness","retrieved_evidence_relevance","retrieved_evidence_coverage","citation_support","citation_completeness"]
    out=[]
    for outcome in (True,False):
        subset=[r for r in records if r["condition"]=="C4" and get_value(r,"answer_correctness","primary")==outcome]
        row={"C4_correctness":"correct" if outcome else "not_correct","n":len(subset)}
        for metric in fields:
            if metric=="factual_accuracy_unknown": vals=[factual_unknown_share(r,"primary") for r in subset]
            else: vals=[get_value(r,metric,"primary") for r in subset]
            row[metric]=_mean(vals)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n",encoding="utf-8");return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(rows)


def _summary_wide(rows: list[dict[str, Any]], questions: dict[str,dict[str,Any]]) -> list[dict[str,Any]]:
    wide=[]
    for r in rows:
        row={"id":r["id"],"condition":r["condition"],"type":r.get("type")}
        for metric in ("answer_correctness","evidence_faithfulness","factual_accuracy","response_relevance","instruction_fulfilment","completeness","retrieved_evidence_relevance","retrieved_evidence_coverage","citation_support","citation_completeness"):
            for judge in JUDGES:
                row[f"{metric}_{judge}"]=get_value(r,metric,judge)
        for judge in JUDGES:
            row[f"appropriate_abstention_{judge}"]=r.get("metrics",{}).get("appropriate_abstention",{}).get(judge,{}).get("label")
        wide.append(row)
    return wide


def _md_table(rows: list[dict[str,Any]], columns: list[str] | None=None) -> str:
    if not rows: return "_No applicable rows._"
    columns=columns or list(rows[0])
    def fmt(v: Any) -> str:
        if v is None: return "—"
        if isinstance(v,float): return f"{v:.3f}"
        return str(v).replace("|","\\|").replace("\n"," ")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(fmt(r.get(c)) for c in columns)+" |" for r in rows)


def _error_exclusions(records: list[dict[str,Any]]) -> dict[str,int]:
    counts=Counter()
    for r in records:
        for metric,judges in r.get("metrics",{}).items():
            if not isinstance(judges,dict): continue
            for value in judges.values():
                if isinstance(value,dict) and value.get("status")=="excluded": counts[f"{metric}: {value.get('reason','unspecified')}"]+=1
    return dict(counts)


def _log_usage(started: str) -> tuple[dict[str,int],dict[str,int],int]:
    calls=Counter();tokens=Counter();parse_errors=0
    for role in ("judge_primary","judge_secondary"):
        path=OUT/"logs"/f"{role}.jsonl"
        if not path.exists(): continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try: event=json.loads(line)
            except json.JSONDecodeError: continue
            if event.get("timestamp","")<started: continue
            if event.get("event")=="ragas_metric_call" or event.get("api_request_sent") is True:
                if (event.get("event")=="ragas_metric_call" and event.get("http_status",0)<400) or event.get("status")=="success":
                    calls[role]+=1;tokens[role]+=int((event.get("token_usage") or {}).get("total_tokens",0) or 0)
            if str(event.get("error_type","")).lower() in {"jsondecodeerror","incompleteoutputexception"}:
                parse_errors+=1
    return calls,tokens,parse_errors


def analyze() -> dict[str,Any]:
    SUMMARY.mkdir(parents=True,exist_ok=True)
    records=read_jsonl(ITEMS/"judgements.jsonl")
    questions={str(r["id"]):r for r in read_jsonl(ROOT/"outputs/final/questions_final.jsonl")}
    condition,type_rows=condition_tables(records,questions)
    paired=paired_table(records)
    agree=agreement_table(records)
    discr=discrimination_table(records)
    c4_summary,c4_items=c4_attribution(records)
    comparison=correct_incorrect(records)
    write_csv(SUMMARY/"metric_by_condition_S1.csv",condition)
    write_csv(SUMMARY/"metric_by_type_C4_S1.csv",type_rows)
    write_csv(SUMMARY/"robustness_paired_S1.csv",paired)
    write_csv(SUMMARY/"judge_agreement.csv",agree)
    write_csv(SUMMARY/"discrimination.csv",discr)
    write_csv(SUMMARY/"c4_error_attribution_S1.csv",c4_summary)
    write_csv(SUMMARY/"correct_vs_incorrect_C4.csv",comparison)
    write_csv(ITEMS/"answer_metrics_wide.csv",_summary_wide(records,questions))
    write_csv(ITEMS/"c4_error_attribution_items.csv",c4_items)
    # Compare the primary judge's correctness with the earlier Qwen result without exposing row details.
    earlier=read_jsonl(ROOT/"outputs/final/scores_final_revised.jsonl")
    old={(str(x["id"]),str(x["condition"])):(x.get("ragchecker") or {}).get("correctness") for x in earlier}
    mismatch=0;matched=0
    for r in records:
        current=get_value(r,"answer_correctness","primary");legacy=old.get((str(r["id"]),str(r["condition"])))
        if isinstance(current,bool) and isinstance(legacy,bool): matched+=1;mismatch+=current!=legacy
    started_data=json.loads((ITEMS/"full_run_state.json").read_text(encoding="utf-8"))
    calls,tokens,parse_errors=_log_usage(str(started_data["started_at"]))
    exclusions=_error_exclusions(records)
    model_ids={}
    for role in JUDGES:
        settings_path=ROOT/"configs/models.yaml"
        import yaml
        config=yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        model_ids[role]=config[f"judge_{role}"]["model"]
    lines=["# Metrics v2 full-run summary","",f"Scored {len(records)} S1 answers. No S2 answer set was available, so S2 tables were not produced.","",
           "## Judges and usage", "",f"- Primary judge model: `{model_ids['primary']}`.",f"- Secondary judge model: `{model_ids['secondary']}`.",
           f"- Calls since full-run start: primary {calls['judge_primary']:,} ({tokens['judge_primary']:,} tokens); secondary {calls['judge_secondary']:,} ({tokens['judge_secondary']:,} tokens).",
           f"- Unresolved parse failures: {parse_errors}.","- No model prices are configured; cost is not estimated.","- Reused secondary-judge results: none; prior prompts or inputs were not identical.",
           f"- Primary answer-correctness disagreement with earlier Qwen results: {mismatch}/{matched} matched answers.","",
           "## Applicability exclusions",""]
    lines.extend(f"- {k}: {v}" for k,v in sorted(exclusions.items()))
    tables=[("Metrics by condition",condition,["dimension","metric","judge"]+[f"{c}_{suffix}" for c in CONDITIONS for suffix in ("n_applicable","estimate","ci95_lower","ci95_upper","interval")],"Binary rates use 95% Wilson intervals; means use 10,000 question-level bootstrap resamples (seed 42)."),
            ("C4 metrics by question type",type_rows,["dimension","metric","judge","type","n_applicable","estimate","ci95_lower","ci95_upper","interval"],"Binary rates use 95% Wilson intervals; means use 10,000 question-level bootstrap resamples (seed 42)."),
            ("Paired robustness",paired,None,"Binary differences use Newcombe (1998) method 10 and exact McNemar; score differences use paired bootstrap intervals and Wilcoxon signed-rank tests. Holm adjustment is within judge and comparison."),
            ("Judge agreement",agree,None,"Agreement is calculated on pairs for which both judges produced an applicable value. Kappa is unstable when nearly all answers receive the same value."),
            ("Discrimination",discr,None,"Best/worst shares are descriptive; standard deviation is computed over applicable answers."),
            ("C4 error attribution",c4_summary,None,"Categories use primary-judge retrieval and answer labels; the table also reports category changes under the secondary judge."),
            ("Correct vs not correct C4 answers",comparison,None,"Descriptive means of metrics 2–10 using the primary judge.")]
    for title,table,cols,note in tables:
        lines.extend([f"## {title}","",_md_table(table,cols),"",note,""])
    (SUMMARY/"metrics_v2_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"answers":len(records),"files":len(list(SUMMARY.glob("*"))),"calls_primary":calls["judge_primary"],"calls_secondary":calls["judge_secondary"],"tokens_primary":tokens["judge_primary"],"tokens_secondary":tokens["judge_secondary"],"parse_errors":parse_errors,"correctness_mismatches":mismatch,"correctness_matched":matched}


if __name__ == "__main__":
    print(json.dumps(analyze(),sort_keys=True))
