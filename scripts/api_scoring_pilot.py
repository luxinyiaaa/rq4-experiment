"""Score the complete common-success API-answer pilot with S1 metrics v2 judges."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from scipy.stats import spearmanr
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from rq4.llm import _load_role
from scripts import metrics_v2 as s1
from scripts.build_index import load_index
from scripts.analyze_metrics_v2 import _kappa, wilson

OUT = ROOT / "outputs/api_scoring/pilot"
ITEMS = OUT / "items"
ANSWERS = ROOT / "outputs/api_answers/full/answers.jsonl"
INVALID = ROOT / "outputs/api_answers/full/answers_invalid_after_quota.jsonl"
QUESTIONS = ROOT / "outputs/rawpage_questions/question_set_v3/questions_main.jsonl"
RAW = ROOT / "data/raw/site_pages_rows.csv"
PROMPTS = ROOT / "prompts/judge_metrics"
EMBED_LOCK = threading.Lock()
INPUT_SNAPSHOT_ROWS = 2752


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_url(url):
    value = str(url).strip()
    if not re.match(r"^[a-z]+://", value, re.I):
        value = "https://" + value
    parsed = urlsplit(value)
    host = parsed.netloc.casefold().removeprefix("www.")
    return (host + parsed.path.rstrip("/")).casefold()


def get_inputs():
    if not INVALID.exists():
        raise FileNotFoundError("Required invalid-answer quarantine file is missing")
    questions = {q["id"]: q for q in read_jsonl(QUESTIONS)}
    source_rows = read_jsonl(ANSWERS)
    if len(source_rows) < INPUT_SNAPSHOT_ROWS:
        raise ValueError("API answers are shorter than the pilot's starting snapshot")
    successes = [a for a in source_rows[:INPUT_SNAPSHOT_ROWS] if a.get("set") == "main" and
                 a.get("http_status") == 200 and str(a.get("answer", "")).strip()]
    grouped = defaultdict(dict)
    for a in successes:
        model = a["model_name"]
        if model in grouped[a["id"]]:
            raise ValueError("Duplicate successful answer pair")
        grouped[a["id"]][model] = a
    expected = {"openai", "openai_gpt56_sol", "openai_gpt6_astra", "egi"}
    common = sorted(qid for qid, models in grouped.items() if set(models) == expected)
    if len(common) != 42:
        raise ValueError(f"Expected 42 common main questions, found {len(common)}")
    if any(qid not in questions for qid in common):
        raise ValueError("Common answer absent from question set")
    return questions, [grouped[qid][model] for qid in common for model in sorted(expected)]


class Judges:
    def __init__(self):
        load_dotenv(ROOT / ".env", override=False)
        primary = _load_role("judge_primary", ROOT / "configs/models.yaml")
        secondary = _load_role("judge_secondary", ROOT / "configs/models.yaml")
        key = os.getenv(primary["api_key_env"])
        if not key:
            raise RuntimeError("OpenRouter credential is unavailable")
        self.settings = {"judge_primary": primary, "judge_secondary": {**secondary,
                         "base_url": primary["base_url"], "api_key_env": primary["api_key_env"],
                         "model": "qwen/" + secondary["model"],
                         "thinking_parameters": {"reasoning": {"effort": "none"}}}}
        self.client = OpenAI(api_key=key, base_url=primary["base_url"], timeout=75, max_retries=1)
        self.lock = threading.Lock()
        self.calls = Counter()
        self.tokens = defaultdict(Counter)

    def call(self, role, messages, *, json_mode=False, temperature=0, max_tokens=2048, **kwargs):
        token_limit = max(max_tokens, 128) if not json_mode else max(max_tokens, 4096 if role == "judge_secondary" else 2048)
        args = {"model": self.settings[role]["model"], "messages": messages,
                "temperature": temperature, "max_tokens": token_limit}
        extra_body = kwargs.get("extra_body")
        if extra_body:
            args["extra_body"] = extra_body
        if json_mode:
            args["response_format"] = {"type": "json_object"}
        for attempt in range(2):
            response = self.client.chat.completions.create(**args)
            usage = response.usage
            with self.lock:
                self.calls[role] += 1
                if usage:
                    self.tokens[role]["input"] += usage.prompt_tokens or 0
                    self.tokens[role]["output"] += usage.completion_tokens or 0
                with (OUT / "judge_calls.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"role": role, "input_tokens": usage.prompt_tokens or 0 if usage else 0,
                                        "output_tokens": usage.completion_tokens or 0 if usage else 0}) + "\n")
            value = response.choices[0].message.content or ""
            if not json_mode:
                break
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    break
            except json.JSONDecodeError:
                pass
            if attempt == 1:
                raise ValueError(f"{role} returned invalid JSON after one retry")
            args["messages"] = [*messages, {"role": "user", "content": "Return one complete valid JSON object only."}]
            args["max_tokens"] = max(args["max_tokens"], 8192)
        if not json_mode and value.strip() not in s1.LABELS:
            print(f"Unexpected {role} keypoint label: {value[:120]!r}", flush=True)
        result = {"text": value, "usage": {"prompt_tokens": usage.prompt_tokens or 0,
                  "completion_tokens": usage.completion_tokens or 0} if usage else {},
                  "api_model": response.model, "cache_hit": False}
        if json_mode:
            result["json"] = parsed
        return result

    def prompt(self, role, name, payload, system_extra=""):
        system = (PROMPTS / (name + ".md")).read_text(encoding="utf-8") + system_extra
        return self.call(role, [{"role": "system", "content": system},
                                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                         json_mode=True)["json"]


def raw_chunks(index):
    by_id = {str(row["chunk_id"]): i for i, row in enumerate(index.chunk_records)}
    grouped = defaultdict(list)
    with RAW.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nurl = norm_url(row["url"])
            chunk_id = str(row["id"])
            idx = by_id.get(chunk_id)
            vec = index.chunk_embeddings[idx] if idx is not None and index.chunk_records[idx]["text"] == row["content"] else None
            grouped[nurl].append({"id": chunk_id, "url": row["url"], "text": row["content"],
                                  "vector": np.asarray(vec, dtype=np.float32) if vec is not None else None})
    return grouped


def evidence(chunks, index, statement, cited_urls):
    with EMBED_LOCK:
        query = index._query_embedding(statement)
    selected = []
    missing = []
    for url in cited_urls:
        options = chunks.get(norm_url(url), [])
        if not options:
            missing.append(url)
            continue
        unencoded = [x for x in options if x["vector"] is None]
        if unencoded:
            with EMBED_LOCK:
                vectors = index.encoder.encode([x["text"] for x in unencoded], batch_size=64,
                                               normalize_embeddings=True, convert_to_numpy=True,
                                               show_progress_bar=False)
            for x, vector in zip(unencoded, vectors):
                x["vector"] = np.asarray(vector, dtype=np.float32)
        scores = np.array([float(x["vector"] @ query) for x in options])
        for i in np.argsort(-scores, kind="stable")[:3]:
            x = options[int(i)]
            selected.append({"cited_url": url, "chunk_id": x["id"], "text": x["text"],
                             "similarity": float(scores[int(i)])})
    return selected, missing


DECOMP_EXTRA = """
For this API-answer task, return each sentence or list item as a statement with
`statement` and `cited_urls` (a list of complete URLs). Recognise inline URLs,
markdown links, URLs in 【】, and a `Source:` line directly after a statement
or list item as belonging to that statement. Do not assign a source line to
unrelated statements. Preserve every answer statement once. Return JSON
`{"statements":[{"statement":"...","cited_urls":["..."]}]}`.
"""

ALT_PROMPT = """Given the question, reference answer, keypoints and API answer,
decide whether an answer labelled incorrect nevertheless gives a different answer
to the same question that could also be valid. Distinguish a genuinely valid
alternative from a wrong, incomplete, evasive, or merely differently worded
answer. Return JSON {"alternative_answer":"yes|no","reason":"<one sentence>"}."""


def score_one(a, q, judges, chunks, index):
    answer = a["answer"]
    item = {"id": a["id"], "model": a["model_name"], "type": q["type"],
            "page_type": q["page_type"], "source_url": a["source_url"],
            "question": q["question"], "reference_answer": q["reference_answer"],
            "keypoints": q["keypoints"], "answer": answer, "metrics": {}, "judgements": []}
    primary_label = judges.prompt("judge_primary", "appropriate_abstention",
                                   {"question": q["question"], "answer": answer},
                                   "\nAsking the user for more information instead of answering counts as abstains.")
    if primary_label.get("label") not in s1.ABSTENTION_LABELS:
        raise ValueError("Invalid primary abstention label")
    item["primary_abstention_label"] = primary_label["label"]
    for role, name in (("judge_primary", "primary"), ("judge_secondary", "secondary")):
        labels = s1._ragchecker_run(role, {"id": a["id"], "condition": "API", "answer": answer,
                                            "passages": []}, q, "response2answer",
                                    settings=judges.settings[role], mock_llm=judges.call)
        values = [str(x) for x in labels["values"]]
        summary = s1.keypoint_summary(q["keypoints"], values)
        item["metrics"].setdefault("answer_correctness", {})[name] = summary["correct"]
        item["metrics"].setdefault("completeness", {})[name] = summary["completeness"]
        item["judgements"].append({"metric": "answer_correctness_completeness", "judge": name,
                                    "labels": values, "trace": labels["trace"]})
        relevance = judges.prompt(role, "response_relevance", {"question": q["question"], "answer": answer})
        if relevance.get("score") not in range(1, 6):
            raise ValueError("Invalid relevance score")
        item["metrics"].setdefault("response_relevance", {})[name] = relevance
        abstention = primary_label if name == "primary" else judges.prompt(role,
            "appropriate_abstention", {"question": q["question"], "answer": answer},
            "\nAsking the user for more information instead of answering counts as abstains.")
        if abstention.get("label") not in s1.ABSTENTION_LABELS:
            raise ValueError("Invalid abstention label")
        item["metrics"].setdefault("appropriate_abstention", {})[name] = abstention
        item["judgements"].append({"metric": "response_relevance", "judge": name, "output": relevance})
        item["judgements"].append({"metric": "appropriate_abstention", "judge": name, "output": abstention})

    if primary_label["label"] != "abstains":
        decomp_system = (PROMPTS / "statement_decomposition.md").read_text(encoding="utf-8").split("## Output")[0] + DECOMP_EXTRA
        decomp = judges.call("judge_primary", [{"role": "system", "content": decomp_system},
            {"role": "user", "content": json.dumps({"answer": answer}, ensure_ascii=False)}],
            json_mode=True)["json"]
        statements = decomp.get("statements")
        if not isinstance(statements, list):
            raise ValueError("Invalid statement decomposition")
        item["statements"] = statements
        citation_inputs = []
        for i, statement in enumerate(statements):
            urls = list(dict.fromkeys(str(x) for x in statement.get("cited_urls", []) if str(x).strip()))
            if urls:
                selected, missing = evidence(chunks, index, statement["statement"], urls)
                citation_inputs.append((i, statement, urls, selected, missing))
        for name, role in (("primary", "judge_primary"), ("secondary", "judge_secondary")):
            def score_citation(entry):
                i, statement, urls, selected, missing = entry
                if selected:
                    support = judges.prompt(role, "citation_support", {"statement": statement["statement"],
                        "cited_passages": [{"passage_id": x["chunk_id"], "text": x["text"]} for x in selected]})
                    entailed = support.get("entailed")
                    if type(entailed) is not bool:
                        raise ValueError("Invalid citation support label")
                else:
                    support, entailed = {"reason": "No cited URL is in the KB"}, False
                return {"statement_index": i, "cited_urls": urls, "missing_kb_urls": missing,
                        "selected_chunks": selected, "entailed": entailed, "reason": support.get("reason")}
            with ThreadPoolExecutor(max_workers=3) as pool:
                support_rows = list(pool.map(score_citation, citation_inputs))
            item["metrics"].setdefault("citation_support", {})[name] = {
                "score": sum(x["entailed"] for x in support_rows) / len(support_rows) if support_rows else None,
                "cited_statements": support_rows, "cited_url_not_in_kb": sum(len(x["missing_kb_urls"]) for x in support_rows)}
            attribution = judges.prompt(role, "citation_completeness", {"statements": [
                {"statement_id": i, "statement": x["statement"], "cited_urls": x.get("cited_urls", [])}
                for i, x in enumerate(statements)]})
            rows = attribution.get("statements", attribution.get("results"))
            if not isinstance(rows, list) or len(rows) != len(statements):
                raise ValueError("Invalid citation completeness rows")
            complete = []
            for i, row in enumerate(rows):
                if type(row.get("needs_attribution")) is not bool:
                    raise ValueError("Invalid needs_attribution")
                complete.append({"statement_index": i, "needs_attribution": row["needs_attribution"],
                                 "has_cited_url": bool(statements[i].get("cited_urls")), "reason": row.get("reason")})
            need = [x for x in complete if x["needs_attribution"]]
            item["metrics"].setdefault("citation_completeness", {})[name] = {
                "score": sum(x["has_cited_url"] for x in need) / len(need) if need else None,
                "statements": complete}
        item["judgements"].append({"metric": "statement_decomposition", "judge": "primary", "output": decomp})
    else:
        for metric in ("citation_support", "citation_completeness"):
            item["metrics"][metric] = {name: {"score": None, "status": "excluded"} for name in ("primary", "secondary")}

    if primary_label["label"] != "abstains" and not item["metrics"]["answer_correctness"]["primary"]:
        alt = judges.call("judge_primary", [{"role": "system", "content": ALT_PROMPT},
            {"role": "user", "content": json.dumps({"question": q["question"],
            "reference_answer": q["reference_answer"], "keypoints": q["keypoints"], "answer": answer},
            ensure_ascii=False)}], json_mode=True)["json"]
        if alt.get("alternative_answer") not in {"yes", "no"}:
            raise ValueError("Invalid alternative answer label")
        item["alternative_answer"] = alt
    item["usage_logged"] = True
    return item


def value(item, metric, judge):
    x = item["metrics"][metric][judge]
    if metric in ("answer_correctness", "completeness"):
        return x
    if metric == "response_relevance":
        return x["score"]
    if metric == "appropriate_abstention":
        return x["label"]
    return x["score"]


def report(items, judges):
    metrics = ("answer_correctness", "completeness", "response_relevance", "citation_support", "citation_completeness")
    rows = []
    for grouping in ("model", "type", "page_type"):
        keys = sorted({(x["model"], x[grouping]) for x in items}) if grouping != "model" else sorted({(x["model"], "all") for x in items})
        for model, group in keys:
            subset = [x for x in items if x["model"] == model and (grouping == "model" or x[grouping] == group)]
            for judge in ("primary", "secondary"):
                row = {"grouping": grouping, "model": model, "group": group, "judge": judge, "n": len(subset)}
                for metric in metrics:
                    vals = [value(x, metric, judge) for x in subset]
                    vals = [float(v) for v in vals if v is not None]
                    row[metric + "_n"] = len(vals)
                    row[metric] = sum(vals) / len(vals) if vals else None
                    if len(vals) >= 10:
                        if metric == "answer_correctness":
                            row[metric + "_ci95"] = wilson(int(sum(vals)), len(vals))
                        else:
                            rng = np.random.default_rng(42)
                            arr = np.asarray(vals)
                            draws = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(axis=1)
                            row[metric + "_ci95"] = tuple(float(y) for y in np.quantile(draws, [0.025, 0.975]))
                labels = [value(x, "appropriate_abstention", judge) for x in subset]
                row["cited_url_not_in_kb"] = sum(x["metrics"]["citation_support"][judge].get("cited_url_not_in_kb", 0)
                                                  for x in subset)
                for label, key in (("abstains", "false_abstention_rate"), ("abstains", "overall_abstention_rate"),
                                   ("partly_answers", "partly_answers_rate")):
                    row[key] = sum(x == label for x in labels) / len(labels)
                    row[key + "_n"] = len(labels)
                    if len(labels) >= 10:
                        row[key + "_ci95"] = wilson(sum(x == label for x in labels), len(labels))
                rows.append(row)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
    agree = []
    for metric in (*metrics, "appropriate_abstention"):
        pairs = [(value(x, metric, "primary"), value(x, metric, "secondary")) for x in items]
        pairs = [(p, s) for p, s in pairs if p is not None and s is not None]
        pvals, svals = [p for p, _ in pairs], [s for _, s in pairs]
        agree.append({"metric": metric, "n_pairs": len(pairs), "exact_agreement":
            sum(p == s for p, s in pairs) / len(pairs) if pairs else None,
            "cohen_kappa": _kappa(pvals, svals) if metric in ("answer_correctness", "appropriate_abstention") and pairs else None,
            "quadratic_weighted_kappa": _kappa(pvals, svals, weighted=True) if metric == "response_relevance" and pairs else None,
            "spearman": float(spearmanr(pvals, svals).statistic) if metric != "appropriate_abstention" and len(pairs) > 1 and len(set(pvals)) > 1 and len(set(svals)) > 1 else None,
            "agreement_within_one_point": sum(abs(float(p)-float(s)) <= 1 for p,s in pairs)/len(pairs) if metric == "response_relevance" and pairs else None,
            "mean_absolute_difference": sum(abs(float(p)-float(s)) for p, s in pairs)/len(pairs) if pairs and metric != "appropriate_abstention" else None})
    with (OUT / "judge_agreement.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agree[0]); writer.writeheader(); writer.writerows(agree)
    bias = []
    for model in sorted({x["model"] for x in items}):
        subset = [x for x in items if x["model"] == model]
        row = {"model": model, "n": len(subset)}
        for metric in ("answer_correctness", "completeness", "response_relevance"):
            row[metric + "_primary_minus_secondary"] = sum(float(value(x, metric, "primary")) -
                float(value(x, metric, "secondary")) for x in subset) / len(subset)
        bias.append(row)
    with (OUT / "family_bias.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bias[0]); writer.writeheader(); writer.writerows(bias)
    alternatives = [{"id": x["id"], "model": x["model"], "question": x["question"],
                     "answer": x["answer"], "reason": x["alternative_answer"].get("reason")}
                    for x in items if x.get("alternative_answer", {}).get("alternative_answer") == "yes"]
    (OUT / "alternative_answer_cases.json").write_text(json.dumps(alternatives, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = {role: {field: sum(x.get("judge_usage", {}).get(role, {}).get(field, 0)
                                for x in items if not x.get("usage_logged"))
                     for field in ("calls", "input_tokens", "output_tokens")}
              for role in ("judge_primary", "judge_secondary")}
    log = OUT / "judge_calls.jsonl"
    if log.exists():
        for event in read_jsonl(log):
            role = event["role"]
            totals[role]["calls"] += 1
            totals[role]["input_tokens"] += event["input_tokens"]
            totals[role]["output_tokens"] += event["output_tokens"]
    usage = {"pilot_answers": len(items), "by_judge": totals,
             "estimate_1200_answers": {role: {field: round(n*1200/len(items)) for field, n in values.items()}
                 for role, values in totals.items()}}
    (OUT / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")


def main():
    questions, answers = get_inputs()
    ITEMS.mkdir(parents=True, exist_ok=True)
    snapshot = {"source_rows_at_start": INPUT_SNAPSHOT_ROWS, "question_count": len({a["id"] for a in answers}),
                "answer_count": len(answers), "answer_pairs": [[a["id"], a["model_name"]] for a in answers]}
    (OUT / "input_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    judges = Judges()
    index = load_index()
    chunks = raw_chunks(index)
    items = []
    def process(entry):
        n, a = entry
        path = ITEMS / f"{a['id']}__{a['model_name']}.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                item = score_one(a, questions[a["id"]], judges, chunks, index)
            except (ValueError, TimeoutError) as exc:
                print(f"Retrying item {n} after {type(exc).__name__}", flush=True)
                item = score_one(a, questions[a["id"]], judges, chunks, index)
            path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Scored {n}/{len(answers)}", flush=True)
        return item
    with ThreadPoolExecutor(max_workers=6) as pool:
        items = list(pool.map(process, enumerate(answers, 1)))
    report(items, judges)
    print("Pilot complete", flush=True)


if __name__ == "__main__":
    main()
