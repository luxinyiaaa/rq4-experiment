"""Review low relevance and relevance among non-abstaining answers from cached labels."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "outputs/api_scoring/full"))
from final_analysis import abstention, write_csv

BASE = ROOT / "outputs/api_scoring/full"
OUT = BASE / "final"
S1 = ROOT / "outputs/metrics_v2"
MODELS = ("egi", "openai", "openai_gpt56_sol", "openai_gpt6_astra")
CONDITIONS = ("C1", "C2", "C3", "C4")


def bootstrap(values, seed=42):
    if len(values) < 10:
        return None, None
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = data[rng.integers(0, len(data), (10000, len(data)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def main():
    snapshot = json.loads((BASE / "input_snapshot.json").read_text())
    api = [json.loads((BASE / "items" / (key + ".json")).read_text()) for key in snapshot["answer_ids"]]
    s1 = [json.loads(line) for line in (S1 / "items/judgements.jsonl").open(encoding="utf-8")]
    questions = {x["id"]: x["question"] for x in (json.loads(line) for line in
        (ROOT / "outputs/final/questions_final.jsonl").open(encoding="utf-8"))}
    if len(api) != 376 or len(s1) != 176:
        raise ValueError("Unexpected scored answer count")
    low = []
    for experiment, records in (("API", api), ("S1", s1)):
        for item in records:
            p = item["metrics"]["response_relevance"]["primary"]
            s = item["metrics"]["response_relevance"]["secondary"]
            if min(p["score"], s["score"]) > 3:
                continue
            low.append({"experiment": experiment, "answer_id": item["id"],
                "model_or_condition": item["model"] if experiment == "API" else item["condition"],
                "question": item["question"] if experiment == "API" else questions[item["id"]],
                "full_answer_text": item["answer"],
                "primary_score": p["score"], "primary_reason": p["reason"],
                "secondary_score": s["score"], "secondary_reason": s["reason"],
                "primary_abstention_label": abstention(item["primary_abstention_label"])})
    (OUT / "relevance_low_scores_data.json").write_text(json.dumps(low, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for experiment, records, groups, field in (("API", api, MODELS, "model"), ("S1", s1, CONDITIONS, "condition")):
        for group in groups:
            subset = [x for x in records if x[field] == group and
                abstention(x["primary_abstention_label"]) in ("answers", "partly answers")]
            for judge in ("primary", "secondary"):
                values = [x["metrics"]["response_relevance"][judge]["score"] for x in subset]
                lo, hi = bootstrap(values)
                rows.append({"experiment": experiment, "model_or_condition": group, "judge": judge,
                    "n": len(values), "mean_relevance": float(np.mean(values)) if values else None,
                    "ci95_lower": lo, "ci95_upper": hi})
    write_csv(OUT / "relevance_nonabstaining.csv", rows)
    low_counts = Counter((x["experiment"], x["model_or_condition"]) for x in low)
    def table(rows, headers):
        return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
            "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |" for row in rows)
    counts = [{"experiment": exp, "model_or_condition": group, "n_low": low_counts[exp, group]}
        for exp, groups in (("API", MODELS), ("S1", CONDITIONS)) for group in groups]
    formatted = [{"experiment": r["experiment"], "model_or_condition": r["model_or_condition"],
        "judge": r["judge"], "n": r["n"], "mean_relevance": f"{r['mean_relevance']:.3f}" if r["mean_relevance"] is not None else "—",
        "95% CI": f"[{r['ci95_lower']:.3f}, {r['ci95_upper']:.3f}]" if r["ci95_lower"] is not None else "—"}
        for r in rows]
    path = OUT / "final_results.md"
    content = path.read_text(encoding="utf-8")
    content = content.split("\n## Low relevance scores\n")[0]
    content += ("\n## Low relevance scores\n\n" + table(counts, ["experiment", "model_or_condition", "n_low"]) +
        f"\n\nAn answer is listed when either judge scored relevance at 3 or below; {len(low)} answers meet that threshold. The item-level list is in `relevance_low_scores.xlsx`.\n\n" +
        "## Relevance among non-abstaining answers\n\n" +
        table(formatted, ["experiment", "model_or_condition", "judge", "n", "mean_relevance", "95% CI"]) +
        "\n\nThese means include answers and partly answers according to the primary abstention label. Intervals are 95% percentile bootstrap intervals over answers (10,000 draws, seed 42), omitted when n < 10.\n")
    path.write_text(content, encoding="utf-8")
    print(json.dumps({"low_total": len(low), "api_low": sum(x["experiment"] == "API" for x in low),
        "s1_low": sum(x["experiment"] == "S1" for x in low), "nonabst_rows": len(rows)}))


if __name__ == "__main__":
    main()
