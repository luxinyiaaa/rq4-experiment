"""Aggregate full scores and prepare blind-validation workbook inputs."""
from __future__ import annotations

import csv
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.analyze_metrics_v2 import _kappa, wilson
from outputs.api_scoring.full.run_full import MODELS, OUT, ITEMS, EXCLUDE

METRICS = ("answer_correctness", "completeness", "response_relevance",
           "citation_support", "citation_completeness")
RATES = ("answer_correctness", "false_abstention_rate", "overall_abstention_rate",
         "partly_answers_rate")
LABELS = {"answer_correctness": "Correct", "completeness": "Complete",
          "response_relevance": "Relevance", "citation_support": "Citation support",
          "citation_completeness": "Citation completeness",
          "false_abstention_rate": "False abstain", "overall_abstention_rate": "Overall abstain",
          "partly_answers_rate": "Partly answers"}


def value(item, metric, judge):
    v = item["metrics"][metric][judge]
    if metric in ("answer_correctness", "completeness"):
        return v
    if metric in ("response_relevance", "citation_support", "citation_completeness"):
        return v["score"]
    if metric == "appropriate_abstention":
        return v["label"]
    raise KeyError(metric)


def bootstrap(values, seed=42):
    a = np.asarray(values, dtype=float)
    if len(a) < 10:
        return None
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), (5000, len(a)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [.025, .975])]


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarise(items):
    rows = []
    for grouping in ("model", "type", "page_type"):
        for model in MODELS:
            groups = ["all"] if grouping == "model" else sorted({x[grouping] for x in items if x["model"] == model})
            for group in groups:
                subset = [x for x in items if x["model"] == model and (group == "all" or x[grouping] == group)]
                for judge in ("primary", "secondary"):
                    row = {"grouping": grouping, "model": model, "group": group, "judge": judge, "n": len(subset)}
                    for metric in METRICS:
                        vals = [value(x, metric, judge) for x in subset]
                        vals = [float(v) for v in vals if v is not None]
                        row[metric + "_n"] = len(vals)
                        row[metric] = float(np.mean(vals)) if vals else None
                        row[metric + "_ci95"] = (wilson(round(sum(vals)), len(vals)) if metric == "answer_correctness"
                            else bootstrap(vals)) if len(vals) >= 10 else None
                    labels = [x["primary_abstention_label"] for x in subset]
                    for name, label in (("false_abstention_rate", "abstains"),
                                        ("overall_abstention_rate", "abstains"),
                                        ("partly_answers_rate", "partly_answers")):
                        hits = sum(x == label for x in labels)
                        row[name] = hits / len(labels)
                        row[name + "_n"] = len(labels)
                        row[name + "_ci95"] = wilson(hits, len(labels)) if len(labels) >= 10 else None
                    rows.append(row)
    write_csv(OUT / "summary.csv", rows)
    return rows


def agreement(items):
    rows = []
    for metric in (*METRICS, "appropriate_abstention"):
        pairs = [(value(x, metric, "primary"), value(x, metric, "secondary")) for x in items]
        pairs = [(p, s) for p, s in pairs if p is not None and s is not None]
        pvals = [p for p, _ in pairs]
        svals = [s for _, s in pairs]
        row = {"metric": metric, "n_pairs": len(pairs),
               "exact_agreement": sum(p == s for p, s in pairs) / len(pairs) if pairs else None,
               "exact_agreement_ci95": wilson(sum(p == s for p,s in pairs),len(pairs)) if len(pairs)>=10 else None,
               "cohen_kappa": _kappa(pvals, svals) if metric in ("answer_correctness", "appropriate_abstention") and pairs else None,
               "quadratic_weighted_kappa": _kappa(pvals, svals, weighted=True) if metric == "response_relevance" and pairs else None,
               "spearman": float(spearmanr(pvals, svals).statistic) if metric != "appropriate_abstention"
                   and len(pairs) > 1 and len(set(pvals)) > 1 and len(set(svals)) > 1 else None,
               "agreement_within_one_point": sum(abs(p-s) <= 1 for p,s in pairs)/len(pairs)
                   if metric == "response_relevance" and pairs else None,
               "mean_absolute_difference": float(np.mean([abs(float(p)-float(s)) for p,s in pairs]))
                   if metric != "appropriate_abstention" and pairs else None}
        rows.append(row)
    write_csv(OUT / "judge_agreement.csv", rows)
    return rows


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    result = [None] * m
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (m-rank) * ps[i]))
        result[i] = running
    return result


def newcombe_paired(a, b):
    n = len(a)
    plus = sum(x and not y for x, y in zip(a,b))
    minus = sum(y and not x for x, y in zip(a,b))
    pplus, pminus = plus/n, minus/n
    lp, up = wilson(plus, n)
    lm, um = wilson(minus, n)
    d = pplus-pminus
    return [d - np.sqrt((pplus-lp)**2 + (um-pminus)**2),
            d + np.sqrt((up-pplus)**2 + (pminus-lm)**2)]


def comparisons(items):
    lookup = {(x["id"], x["model"]): x for x in items}
    ids = sorted({x["id"] for x in items})
    rows = []
    for judge in ("primary", "secondary"):
        for metric in ("answer_correctness", "completeness", "response_relevance", "citation_support"):
            group = []
            for model_a, model_b in itertools.combinations(MODELS, 2):
                paired = [(value(lookup[i, model_a], metric, judge),
                           value(lookup[i, model_b], metric, judge)) for i in ids]
                paired = [(float(a),float(b)) for a,b in paired if a is not None and b is not None]
                if not paired:
                    continue
                a, b = zip(*paired)
                diffs = [x-y for x,y in paired]
                if metric == "answer_correctness":
                    ci = newcombe_paired(a,b) if len(paired) >= 10 else None
                    plus = sum(x==1 and y==0 for x,y in paired)
                    minus = sum(x==0 and y==1 for x,y in paired)
                    p = binomtest(min(plus,minus),plus+minus,.5).pvalue if plus+minus else 1.0
                    test = "exact McNemar"
                else:
                    ci = bootstrap(diffs) if len(paired) >= 10 else None
                    p = float(wilcoxon(diffs, zero_method="wilcox").pvalue) if any(diffs) else 1.0
                    test = "Wilcoxon signed-rank"
                group.append({"judge": judge, "metric": metric, "model_a": model_a,
                    "model_b": model_b, "n_pairs": len(paired), "mean_paired_difference": float(np.mean(diffs)),
                    "ci95": ci, "test": test, "p": p})
            adjusted = holm([x["p"] for x in group])
            for row, adj in zip(group, adjusted):row["p_holm"] = adj
            rows.extend(group)
    write_csv(OUT / "model_comparisons.csv", rows)
    return rows


def bias(items):
    ids = sorted({x["id"] for x in items})
    lookup = {(x["id"],x["model"]):x for x in items}
    rows = []
    for metric in ("answer_correctness", "completeness", "response_relevance"):
        by_model = {}
        for model in MODELS:
            d = [float(value(lookup[i,model],metric,"primary"))-
                 float(value(lookup[i,model],metric,"secondary")) for i in ids]
            by_model[model] = d
            rows.append({"metric":metric,"model":model,"n":len(d),
                         "primary_minus_secondary":float(np.mean(d)),"ci95":bootstrap(d)})
        for model in MODELS:
            if model == "egi":
                continue
            contrast = [a-b for a,b in zip(by_model[model],by_model["egi"])]
            rows.append({"metric":metric,"model":model+" minus EGI","n":len(ids),
                         "primary_minus_secondary":float(np.mean(contrast)),"ci95":bootstrap(contrast)})
        family = [(by_model["openai"][i]+by_model["openai_gpt56_sol"][i]+
                   by_model["openai_gpt6_astra"][i])/3-by_model["egi"][i] for i in range(len(ids))]
        rows.append({"metric":metric,"model":"OpenAI family minus EGI","n":len(ids),
                     "primary_minus_secondary":float(np.mean(family)),"ci95":bootstrap(family)})
    write_csv(OUT / "family_bias.csv",rows)
    return rows


def outside_urls(items):
    rows=[]
    for item in items:
        cited=[]
        for entry in item["metrics"]["citation_support"]["primary"].get("cited_statements",[]):
            for url in entry.get("missing_kb_urls",[]):
                cited.append((url,entry["statement_index"]))
        for url,idx in cited:
            rows.append({"model":item["model"],"answer_id":item["id"],
                         "statement_index":idx,"cited_url":url})
    write_csv(OUT / "cited_urls_outside_kb.csv",rows,
              ["model","answer_id","statement_index","cited_url"])
    return rows


def validation(items):
    old=set(json.loads(EXCLUDE.read_text())["question_ids"])
    byid={x["id"]:x for x in items if x["model"]==MODELS[0] and x["id"] not in old}
    strata=defaultdict(list)
    for qid,item in byid.items():strata[item["page_type"]].append(qid)
    n=min(40,len(byid))
    exact={k:n*len(v)/len(byid) for k,v in strata.items()}
    counts={k:int(exact[k]) for k in strata}
    for k in sorted(strata,key=lambda k:(-(exact[k]-counts[k]),k))[:n-sum(counts.values())]:counts[k]+=1
    rng=random.Random(42)
    selected=[]
    for k in sorted(strata):selected.extend(rng.sample(sorted(strata[k]),counts[k]))
    rng.shuffle(selected)
    lookup={(x["id"],x["model"]):x for x in items}
    human=[]; key=[]
    for pos,qid in enumerate(selected):
        model=MODELS[pos%4]
        item=lookup[qid,model]
        sample=f"HV{pos+1:03d}"
        decisive=[p["text"] for p in item["keypoints"] if p.get("decisive")]
        kp_lines=[]
        decisive_number=0
        for p in item["keypoints"]:
            if p.get("decisive"):
                decisive_number+=1
                kp_lines.append(f"[DECISIVE {decisive_number}] {p['text']}")
            else:
                kp_lines.append(p["text"])
        h={"sample_id":sample,"question":item["question"],"question_type":item["type"],
           "reference_answer":item["reference_answer"],
           "keypoints":"\n".join(kp_lines),
           "answer_text":item["answer"],"correct":""}
        for i in range(8):h[f"decisive_keypoint_{i+1}_present"]="" if i<len(decisive) else "N/A"
        h.update({"relevance":"","abstention":"","note":""})
        human.append(h)
        labels={judge:{m:value(item,m,judge) for m in (*METRICS,"appropriate_abstention")}
                for judge in ("primary","secondary")}
        labels["keypoint_labels"]={j:next((x["labels"] for x in item["judgements"]
           if x.get("metric")=="answer_correctness_completeness" and x.get("judge")==j),[])
           for j in ("primary","secondary")}
        labels["citation_support_labels"]={j:[x["entailed"] for x in item["metrics"]["citation_support"][j].get("cited_statements",[])]
            for j in ("primary","secondary")}
        labels["needs_attribution_labels"]={j:[x["needs_attribution"] for x in item["metrics"]["citation_completeness"][j].get("statements",[])]
            for j in ("primary","secondary")}
        labels["alternative_answer"]=item.get("alternative_answer")
        key.append({"sample_id":sample,"answer_id":qid,"model":model,"judge_labels":json.dumps(labels,ensure_ascii=False)})
    hv=OUT/"human_validation"
    hv.mkdir(exist_ok=True)
    write_csv(hv/"human_validation_key.csv",key)
    (hv/"human_validation_data.json").write_text(json.dumps({"headers":list(human[0]),"rows":human},ensure_ascii=False))
    (hv/"human_validation_sample.json").write_text(json.dumps({"question_ids":selected,
        "page_type_counts":dict(Counter(byid[i]["page_type"] for i in selected)),
        "assigned_models":dict(Counter(x["model"] for x in key))},indent=2))
    return [(x["answer_id"], x["model"]) for x in key]


def alternatives(items,sample):
    sample=set(sample)
    result=[]
    for item in items:
        alt=item.get("alternative_answer",{})
        if alt.get("alternative_answer")!="yes":continue
        result.append({"answer_id":item["id"],"model":item["model"],
                       "question":item["question"],"reference_answer":item["reference_answer"],
                       "keypoints":"\n".join(("[DECISIVE] " if p.get("decisive") else "")+p["text"] for p in item["keypoints"]),
                       "answer":item["answer"],"judge_reason":alt.get("reason",""),
                       "my_decision":""})
    (OUT/"alternative_answers_data.json").write_text(json.dumps({"headers":list(result[0]) if result else
        ["answer_id","model","question","reference_answer","keypoints","answer","judge_reason","my_decision"],
        "rows":[x for x in result if (x["answer_id"],x["model"]) not in sample]},ensure_ascii=False))
    (OUT/"human_validation/alternative_answers_in_sample_data.json").write_text(json.dumps({
        "headers":list(result[0]) if result else ["answer_id","model","question","reference_answer","keypoints","answer","judge_reason","my_decision"],
        "rows":[x for x in result if (x["answer_id"],x["model"]) in sample]},ensure_ascii=False))
    return result


def usage(items):
    log=OUT/"judge_calls.jsonl"
    events=[json.loads(line) for line in log.open() if line.strip()] if log.exists() else []
    current=defaultdict(Counter)
    for x in events:
        current[x["role"]]["calls"]+=1
        current[x["role"]]["input_tokens"]+=x["input_tokens"]
        current[x["role"]]["output_tokens"]+=x["output_tokens"]
    prior=json.loads((ROOT/"outputs/api_scoring/pilot_v2/usage.json").read_text())
    reused=sum(x.get("reused_from_pilot_v2",False) for x in items)
    # Pilot v2 log includes multiple scoring passes; show it separately, not as new usage.
    result={"new_full_run_calls_and_tokens":current,"reused_pilot_v2_answers":reused,
            "pilot_v2_total_calls_and_tokens":prior["by_judge"],
            "pilot_alternative_recheck_count": sum(1 for line in (OUT/"alternative_answer_rechecks.jsonl").open()
                if line.strip()) if (OUT/"alternative_answer_rechecks.jsonl").exists() else 0}
    (OUT/"usage.json").write_text(json.dumps(result,indent=2))
    return result


def cell(row,metric):
    v=row.get(metric);n=row.get(metric+"_n",row["n"])
    if v is None:return "—"
    fmt=(f"{100*v:.1f}%" if metric in RATES else f"{v:.3f}")
    ci=row.get(metric+"_ci95")
    if isinstance(ci,str) and ci.startswith("("):
        try:ci=json.loads(ci.replace("(","[").replace(")","]"))
        except Exception:ci=None
    if ci and n>=10:
        fmt+=f" [{100*ci[0]:.1f}, {100*ci[1]:.1f}]" if metric in RATES else f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{fmt} (n={n})"


def markdown_table(rows,cols):
    out=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for r in rows:out.append("| "+" | ".join(str(r.get(c,"")) for c in cols)+" |")
    return "\n".join(out)


def report(snapshot,summary,comp,agree,bias,outside,alternatives,usage,sample):
    lines=["# Full API-answer scoring", "", f"Scored {snapshot['question_count']} complete main-set questions × four models ({snapshot['answer_count']} answers). "
           f"Page types: {snapshot['page_type_counts']}. Question types: {snapshot['question_type_counts']}.", "",
           f"EGI could-not-find gate: {snapshot['egi_gate']['later_egi_could_not_find_n']}/{snapshot['egi_gate']['later_egi_success_n']} "
           f"({100*snapshot['egi_gate']['later_egi_could_not_find_share']:.1f}%) after the first 42 pilot questions. "
           f"Pre-run upper estimate: {snapshot['estimated_tokens']['total_upper']:,} tokens.", "",
           "Judges: primary GPT-6 Sol and secondary Qwen3.7 Plus. The pilot v2 keypoint and abstention rules were reused. "
           "Citation support groups each answer's statements by cited URL set; "
           "the evidence is all matching KB rows in chunk order up to 30,000 characters. Longer pages use bge-base-en-v1.5 selected chunks and neighbours. "
           "The primary judge's abstention decision controls citation applicability for both judges. "
           "Citation support averages over answers with cited statements; citation completeness averages over answers with statements needing attribution. "
           "All rates use Wilson 95% intervals and all means use bootstrap 95% intervals when n ≥ 10.", ""]
    metrics=(*METRICS,"false_abstention_rate","overall_abstention_rate","partly_answers_rate")
    for grouping,title in (("model","By model"),("type","By question type"),("page_type","By page type")):
        lines.extend([f"## {title}",""])
        for judge in ("primary","secondary"):
            rr=[]
            for row in summary:
                if row["grouping"]!=grouping or row["judge"]!=judge:continue
                rr.append({"Model":row["model"],"Group":row["group"],"n":row["n"],
                           **{LABELS[m]:cell(row,m) for m in metrics}})
            lines.extend([f"### {judge.title()} judge","",markdown_table(rr,["Model","Group","n",*[LABELS[m] for m in metrics]]),""])
    lines.extend(["## Paired model comparisons","",
                  "Positive difference means the first model scored higher. Correctness uses the paired Newcombe interval and exact McNemar test; "
                  "other metrics use paired bootstrap intervals and Wilcoxon signed-rank tests. P values have Holm adjustment across the six pairs within each metric and judge.",""])
    cr=[]
    for x in comp:
        ci=x["ci95"]
        cr.append({"Judge":x["judge"],"Metric":x["metric"],"Pair":x["model_a"]+" − "+x["model_b"],
                   "n":x["n_pairs"],"Difference":f"{x['mean_paired_difference']:+.3f}"+
                   (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""),
                   "p":f"{x['p']:.4g}","Holm p":f"{x['p_holm']:.4g}"})
    lines.extend([markdown_table(cr,["Judge","Metric","Pair","n","Difference","p","Holm p"]),""])
    ar=[]
    for x in agree:
        ar.append({"Metric":x["metric"],"n":x["n_pairs"],"Exact":f"{x['exact_agreement']:.3f}" if x["exact_agreement"] is not None else "—",
                   "Exact 95% CI":f"[{x['exact_agreement_ci95'][0]:.3f}, {x['exact_agreement_ci95'][1]:.3f}]" if x["exact_agreement_ci95"] else "—",
                   "Kappa":f"{x['cohen_kappa']:.3f}" if x["cohen_kappa"] is not None else "—",
                   "Weighted kappa":f"{x['quadratic_weighted_kappa']:.3f}" if x["quadratic_weighted_kappa"] is not None else "—",
                   "Spearman":f"{x['spearman']:.3f}" if x["spearman"] is not None else "—",
                   "Within 1":f"{x['agreement_within_one_point']:.3f}" if x["agreement_within_one_point"] is not None else "—",
                   "MAD":f"{x['mean_absolute_difference']:.3f}" if x["mean_absolute_difference"] is not None else "—"})
    lines.extend(["## Judge agreement","",markdown_table(ar,["Metric","n","Exact","Exact 95% CI","Kappa","Weighted kappa","Spearman","Within 1","MAD"]),""])
    br=[]
    for x in bias:
        ci=x["ci95"]
        br.append({"Metric":x["metric"],"Model":x["model"],"n":x["n"],
                   "Primary − secondary":f"{x['primary_minus_secondary']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else f"{x['primary_minus_secondary']:+.3f}"})
    lines.extend(["## Family-bias check","",markdown_table(br,["Metric","Model","n","Primary − secondary"]),""])
    outside_counts=Counter(x["model"] for x in outside)
    outside_unique={m:len({x["cited_url"] for x in outside if x["model"]==m}) for m in MODELS}
    lines.extend(["## Cited URLs outside the KB","",
                  markdown_table([{"Model":m,"Instances":outside_counts[m],"Unique URLs":outside_unique[m]} for m in MODELS],
                                 ["Model","Instances","Unique URLs"]),""])
    ac=Counter(x["model"] for x in alternatives)
    sample_meta=json.loads((OUT/"human_validation/human_validation_sample.json").read_text())
    alt_rechecks=usage.get("pilot_alternative_recheck_count",0)
    alt_intro=(f"Alternative-answer cases after rechecking {alt_rechecks} pilot answers against pilot v2 correctness: "
               if alt_rechecks else "Alternative-answer cases: ")
    lines.extend(["## Alternative answers and human validation","",
                  alt_intro+", ".join(f"{m} {ac[m]}" for m in MODELS)+". "
                  f"The blind human-validation sample contains {len(sample)} distinct questions (10 answers per model); "
                  f"page types: {sample_meta['page_type_counts']}. "
                  "All pilot question IDs were excluded. Item-level judge labels are excluded from this report.",""])
    rechecks=usage.get("pilot_alternative_recheck_count",0)
    lines.extend(["## Judge calls and tokens","",
                  f"The counts below include the full run and {rechecks} later primary-judge pilot alternative rechecks. "
                  "Earlier pilot v2 judgments reused here are shown separately; failed requests without returned usage are excluded.",""])
    new_totals=usage["new_full_run_calls_and_tokens"]
    total_calls=sum(new_totals.get(role,{}).get("calls",0) for role in ("judge_primary","judge_secondary"))
    total_tokens=sum(new_totals.get(role,{}).get("input_tokens",0)+new_totals.get(role,{}).get("output_tokens",0)
                     for role in ("judge_primary","judge_secondary"))
    lines.extend([f"Full-run and pilot-alternative-recheck total: {total_calls:,} returned calls and {total_tokens:,} tokens; "
                  f"pre-run upper estimate for the original full run {snapshot['estimated_tokens']['total_upper']:,} tokens.",""])
    ur=[]
    for role in ("judge_primary","judge_secondary"):
        v=usage["new_full_run_calls_and_tokens"].get(role,{})
        ur.append({"Judge":role,"New calls":v.get("calls",0),"New input tokens":v.get("input_tokens",0),
                   "New output tokens":v.get("output_tokens",0),
                   "Prior pilot v2 calls":usage["pilot_v2_total_calls_and_tokens"][role]["calls"],
                   "Prior input tokens":usage["pilot_v2_total_calls_and_tokens"][role]["input_tokens"],
                   "Prior output tokens":usage["pilot_v2_total_calls_and_tokens"][role]["output_tokens"]})
    lines.extend([markdown_table(ur,["Judge","New calls","New input tokens","New output tokens",
                                     "Prior pilot v2 calls","Prior input tokens","Prior output tokens"]),""])
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT/"main_table.md").write_text("\n".join(lines[0:lines.index("## Paired model comparisons")]),encoding="utf-8")


def main():
    snapshot=json.loads((OUT/"input_snapshot.json").read_text())
    items=[json.loads((ITEMS/(name+".json")).read_text()) for name in snapshot["answer_ids"]]
    if len(items)!=snapshot["answer_count"] or any(x.get("rule_version")!=snapshot["rule_version"] for x in items):
        raise ValueError("Item set incomplete or rule version mismatch")
    summary=summarise(items)
    comp=comparisons(items)
    agree=agreement(items)
    b=bias(items)
    outside=outside_urls(items)
    sample=validation(items)
    alt=alternatives(items,sample)
    u=usage(items)
    report(snapshot,summary,comp,agree,b,outside,alt,u,sample)
    print("Report and workbook data complete",len(items),len(sample),len(alt))


if __name__=="__main__":main()
