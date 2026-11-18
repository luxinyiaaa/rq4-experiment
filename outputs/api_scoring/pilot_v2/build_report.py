"""Write the self-contained pilot-v2 Markdown report from frozen scoring files."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
OLD = P.parent / "pilot"


def read_csv(name):
    with (P / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def cell(row, name, percent=False):
    value = row.get(name)
    if not value:
        return "—"
    n = row.get(name + "_n") or row["n"]
    rendered = f"{float(value)*100:.1f}%" if percent else f"{float(value):.3f}"
    return f"{rendered} (n={n})"


def table(rows, group):
    subset = [r for r in rows if r["grouping"] == group and r["judge"] == "primary"]
    out = ["| Model | Group | n | Correct | Complete | Relevance | Citation support | Citation completeness | False abstain | Overall abstain | Partly answers |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in subset:
        out.append("| " + " | ".join([r["model"], r["group"], r["n"],
            cell(r,"answer_correctness",True),cell(r,"completeness"),cell(r,"response_relevance"),
            cell(r,"citation_support"),cell(r,"citation_completeness"),
            cell(r,"false_abstention_rate",True),cell(r,"overall_abstention_rate",True),
            cell(r,"partly_answers_rate",True)]) + " |")
    return out


def main():
    summary = read_csv("summary.csv")
    old_summary = list(csv.DictReader((OLD / "summary.csv").open(newline="",encoding="utf-8")))
    oldmap = {(r["grouping"],r["model"],r["group"],r["judge"]):r for r in old_summary}
    changes = read_csv("correctness_changes.csv")
    urls = read_csv("citation_url_audit.csv")
    sample = json.loads((P / "citation_disagreements.json").read_text(encoding="utf-8"))
    usage = json.loads((P / "usage.json").read_text(encoding="utf-8"))["by_judge"]
    lines = ["# API answer scoring pilot v2", "",
      "The same frozen 42 main-set questions × four models (168 answers) were scored. The original pilot and its inputs were read only. The Environmental Expert API was not called.", "",
      "Metrics 1 and 4 were rejudged for every answer by both judges using RAGChecker non-joint keypoint checks. The keypoint prompt added these exact rules:","",
      "> Names that differ only in word form, abbreviation or word order refer to the same entity (for example 'Atmosphere Thematic Centre' and 'Atmospheric Thematic Centre').", "",
      "> A keypoint that corrects a false premise ('X, not Y') is entailed if the answer says that the premise is wrong, or states X in a way that directly contradicts Y. It is not entailed if the answer only answers the rest of the question without stating anything that contradicts Y.", "",
      "The RAGChecker prompt was also arranged so the answer alone supplies evidence; the question supplies context. This prevents a false premise in the question from being treated as evidence. The same arrangement was used for both judges.","",
      "All tables below show the primary judge. Each score cell includes its applicable n. The secondary judge and 95% intervals are in `summary.csv`; intervals are omitted when n < 10. All main-set facts are in the KB, so false- and overall-abstention rates share the same denominator. Citation support excludes abstentions and uncited answers; citation completeness excludes abstentions and answers with no statements needing attribution.","",
      "## By model","",*table(summary,"model"),"","## By question type","",*table(summary,"type"),"","## By page type","",*table(summary,"page_type"),"",
      "## Changes from the original pilot","",
      "The table below compares primary-judge model-level scores. Relevance, citation completeness and abstention labels were carried forward unchanged.","",
      "| Model | Correct old → new | Complete old → new | Citation support old → new |","|---|---:|---:|---:|"]
    for r in summary:
        if r["grouping"]!="model" or r["judge"]!="primary":
            continue
        old=oldmap[("model",r["model"],"all","primary")]
        lines.append(f"| {r['model']} | {float(old['answer_correctness'])*100:.1f}% → {float(r['answer_correctness'])*100:.1f}% | {float(old['completeness']):.3f} → {float(r['completeness']):.3f} | {float(old['citation_support']):.3f} → {float(r['citation_support']):.3f} |")
    lines += ["","### Correctness-label changes","",
      f"{len(changes)} judge-specific labels changed across {len({(r['id'],r['model']) for r in changes})} answers. Old and new keypoint labels are in `correctness_changes.csv`.",""]
    for r in changes:
        lines.append(f"- **{r['id']} / {r['model']} / {r['judge']}: {r['old_correct']} → {r['new_correct']}.** {r['reason']}")
    lines += ["","### Cited URLs initially counted outside the KB","",
      "Every missing-URL instance from the original pilot is listed below with answer context. Unicode hyphens are now normalized to ASCII hyphens for KB lookup. URLs that remain outside the KB still supply no evidence.","",
      "| Answer / model / statement | Cited URL | Answer text around URL | Classification |","|---|---|---|---|"]
    for r in urls:
        context=r["answer_context"].strip()
        lines.append(f"| {r['id']} / {r['model']} / {int(r['statement_index'])+1} | {esc(r['cited_url'])} | {esc(context)} | {r['classification']} |")
    classifications=Counter(r["classification"] for r in urls)
    lines += ["",f"There were {classifications['URL normalization error (Unicode hyphen)']} normalization errors and {classifications['Real URL outside KB']} real outside-KB URL instances. After correction, citation-support scores changed for two EGI answers under each judge; the other answer scores were unchanged.","",
      "### All-chunks disagreement check","",
      "From the 56 statement-level support disagreements after URL correction, 15 were sampled with seed 42 across models (EGI 4, OpenAI 4, GPT-5.6 Sol 4, GPT-6 Astra 3). The primary judge rechecked them with all chunks of their cited URL(s), capped at 30,000 characters in total. The workbook contains the statements, URLs, three selected chunks per URL, both original labels and reasons, recheck labels and reasons, and an empty `My label` column.","",
      f"The primary support label changed in **{sum(x['changed_from_primary'] for x in sample)}/15** rechecks: {sum(not x['primary_label'] and x['all_chunks_label'] for x in sample)} unsupported → supported and {sum(x['primary_label'] and not x['all_chunks_label'] for x in sample)} supported → unsupported. These diagnostic all-chunks labels were not substituted into metric 7, which remains defined by the three selected chunks per URL.","",
      "## Judge agreement","",
      "| Metric | Pairs | Exact agreement | Kappa | Weighted kappa | Spearman | Mean absolute difference |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in read_csv("judge_agreement.csv"):
        fmt=lambda k: f"{float(r[k]):.3f}" if r.get(k) else "—"
        lines.append(f"| {r['metric']} | {r['n_pairs']} | {fmt('exact_agreement')} | {fmt('cohen_kappa')} | {fmt('quadratic_weighted_kappa')} | {fmt('spearman')} | {fmt('mean_absolute_difference')} |")
    lines += ["","## Family-bias check","",
      "Primary minus secondary judge; a positive value means the primary judge gave a higher score.","",
      "| Model | n | Correctness | Completeness | Relevance |","|---|---:|---:|---:|---:|"]
    for r in read_csv("family_bias.csv"):
        lines.append(f"| {r['model']} | {r['n']} | {float(r['answer_correctness_primary_minus_secondary']):+.3f} | {float(r['completeness_primary_minus_secondary']):+.3f} | {float(r['response_relevance_primary_minus_secondary']):+.3f} |")
    lines += ["","## Judge calls and tokens","",
      "Recorded usage includes the initial v2 keypoint pass, the prompt-separation rerun, citation corrections, explanations and all-chunks rechecks. Calls that failed before returning a response are not included.","",
      "| Judge | Calls | Input tokens | Output tokens |","|---|---:|---:|---:|"]
    for role in ("judge_primary","judge_secondary"):
        u=usage[role]
        lines.append(f"| {role} | {u['calls']:,} | {u['input_tokens']:,} | {u['output_tokens']:,} |")
    lines += ["","The frozen answer pairs remain listed in the original pilot's `input_snapshot.json`. The 42 pilot question IDs remain excluded from the later human-validation sample.",""]
    (P / "report.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
