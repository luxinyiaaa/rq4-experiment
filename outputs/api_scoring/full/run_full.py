"""Resumable full API-answer scoring. All writes stay under this directory."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import api_scoring_pilot as base
from scripts import metrics_v2 as s1
from scripts.build_index import load_index
from outputs.api_scoring.pilot_v2 import run_pilot_v2 as v2

OUT = Path(__file__).resolve().parent
ITEMS = OUT / "items"
ANSWERS = ROOT / "outputs/api_answers/full/answers.jsonl"
QUESTIONS = ROOT / "outputs/rawpage_questions/question_set_v3/questions_main.jsonl"
PILOT = ROOT / "outputs/api_scoring/pilot_v2/items"
EXCLUDE = ROOT / "outputs/api_scoring/pilot/pilot_question_ids_exclude_from_human_validation.json"
MODELS = ("egi", "openai", "openai_gpt56_sol", "openai_gpt6_astra")
RULE_VERSION = "pilot-v2-keypoints_grouped-all-url-chunks-30000-v1"
WRITE_LOCK = threading.Lock()


def jwrite(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def cohort():
    questions = {q["id"]: q for q in base.read_jsonl(QUESTIONS)}
    rows = base.read_jsonl(ANSWERS)
    valid = [r for r in rows if r.get("set") == "main" and r.get("model_name") in MODELS
             and r.get("http_status") == 200 and not r.get("error") and str(r.get("answer", "")).strip()]
    grouped = defaultdict(dict)
    for r in valid:
        if r["model_name"] in grouped[r["id"]]:
            raise ValueError("Duplicate successful answer pair")
        grouped[r["id"]][r["model_name"]] = r
    ids = sorted(k for k, g in grouped.items() if set(g) == set(MODELS) and k in questions)
    old = set(json.loads(EXCLUDE.read_text(encoding="utf-8"))["question_ids"])
    later = [r for r in valid if r["model_name"] == "egi" and r["id"] not in old]
    nofind = [r for r in later if re.search(
        r"(?:wasn.t able|unable)\s+to\s+(?:locate|find)|(?:couldn.t|cannot|can.t)\s+(?:locate|find)|not seeing a document|"
        r"don.t have (?:any|enough) information", r["answer"], re.I)]
    gate = {"later_egi_success_n": len(later), "later_egi_could_not_find_n": len(nofind),
            "later_egi_could_not_find_share": len(nofind) / len(later) if later else 0,
            "later_egi_could_not_find_answer_ids": [r["id"] for r in nofind]}
    return questions, [grouped[k][m] for k in ids for m in MODELS], ids, gate


def preflight(questions, answers, ids, gate):
    if gate["later_egi_could_not_find_share"] > .5:
        raise RuntimeError("EGI could-not-find gate exceeded 50%")
    pilot_usage = json.loads((ROOT / "outputs/api_scoring/pilot_v2/usage.json").read_text())
    pilot_items = [json.loads(p.read_text()) for p in PILOT.glob("*.json")]
    groups = sum(len({tuple(sorted(s.get("cited_urls", []))) for s in x.get("statements", [])
                      if s.get("cited_urls")}) for x in pilot_items)
    old_pairs = {(x["id"], x["model"]) for x in pilot_items}
    reused = sum((a["id"], a["model_name"]) in old_pairs for a in answers)
    new = len(answers) - reused
    # Pilot v2 usage includes two keypoint passes and diagnostic reruns. This deliberately
    # overestimates new non-citation work, then budgets 30k characters per grouped call.
    pilot_tokens = sum(x["input_tokens"] + x["output_tokens"] for x in pilot_usage["by_judge"].values())
    noncitation = math.ceil(pilot_tokens * new / len(pilot_items))
    estimated_groups = math.ceil(groups / len(pilot_items) * len(answers))
    citation_budget = estimated_groups * 2 * 8000  # 30k characters, conservatively 8k tokens.
    estimate = noncitation + citation_budget
    result = {"rule_version": RULE_VERSION, "source_answer_rows": len(base.read_jsonl(ANSWERS)),
              "question_count": len(ids), "answer_count": len(answers), "question_ids": ids,
              "answer_ids": [f"{a['id']}__{a['model_name']}" for a in answers],
              "page_type_counts": dict(Counter(questions[k]["page_type"] for k in ids)),
              "question_type_counts": dict(Counter(questions[k]["type"] for k in ids)),
              "egi_gate": gate, "reusable_pilot_v2_answers": reused, "new_answers": new,
              "estimated_calls": {"noncitation_new_upper": math.ceil(sum(x["calls"] for x in pilot_usage["by_judge"].values()) * new / len(pilot_items)),
                                  "grouped_citation_both_judges": estimated_groups * 2},
              "estimated_tokens": {"noncitation_new_upper": noncitation,
                                   "grouped_citation_upper": citation_budget, "total_upper": estimate}}
    OUT.mkdir(parents=True, exist_ok=True)
    jwrite(OUT / "input_snapshot.json", result)
    if estimate > 30_000_000:
        raise RuntimeError(f"Estimated {estimate} tokens exceeds 30 million")
    return result


def raw_pages():
    grouped = defaultdict(list)
    with (ROOT / "data/raw/site_pages_rows.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[v2.norm_url(row["url"])].append({"id": str(row["id"]),
                "number": int(row.get("chunk_number") or 0), "text": row["content"], "url": row["url"]})
    for rows in grouped.values():
        rows.sort(key=lambda x: x["number"])
    return grouped


def clean_urls(urls):
    cleaned = []
    for u in urls:
        u = str(u).strip().strip("`<>【】[](){}\"'").rstrip(".,;:!?")
        if u and u not in cleaned:
            cleaned.append(u)
    return cleaned


class Evidence:
    def __init__(self, pages):
        self.pages = pages
        self.index = None
        self.vector_by_id = None
        self.lock = threading.Lock()

    def query(self, text):
        with self.lock:
            if self.index is None:
                self.index = load_index()
                self.vector_by_id = {str(r["chunk_id"]): self.index.chunk_embeddings[i]
                                     for i, r in enumerate(self.index.chunk_records)}
            return self.index._query_embedding(text)

    def get(self, urls, statements):
        found, missing = [], []
        for url in urls:
            rows = self.pages.get(v2.norm_url(url))
            (found if rows else missing).append((url, rows))
        allrows = [(url, r) for url, rows in found for r in rows]
        if not allrows:
            return [], [u for u, _ in missing]
        # Reserve a little space for URL and chunk identifiers. For short pages, preserve
        # every row and the original chunk order as requested.
        if sum(len(r["text"]) + 80 for _, r in allrows) <= 30000:
            selected = allrows
        else:
            query = self.query("\n".join(statements))
            ranked = []
            for url, rows in found:
                for pos, row in enumerate(rows):
                    vec = self.vector_by_id.get(row["id"])
                    score = float(np.asarray(vec, dtype=np.float32) @ query) if vec is not None else -1.0
                    ranked.append((score, url, pos))
            ranked.sort(reverse=True)
            chosen = set()
            # Add the best chunk and its neighbours; keep all selected chunks in page order.
            for _, url, pos in ranked:
                rows = self.pages[v2.norm_url(url)]
                proposed = {(url, j) for j in (pos - 1, pos, pos + 1) if 0 <= j < len(rows)}
                size = sum(len(self.pages[v2.norm_url(u)][j]["text"]) + 80 for u, j in chosen | proposed)
                if size <= 30000:
                    chosen |= proposed
            selected = [(url, row) for url, rows in found for pos, row in enumerate(rows)
                        if (url, pos) in chosen]
            if not selected:
                best = ranked[0]
                selected = [(best[1], self.pages[v2.norm_url(best[1])][best[2]])]
        passages, remaining = [], 30000
        for url, row in selected:
            prefix = f"Cited URL: {url}\nChunk {row['number']} ({row['id']}):\n"
            value = (prefix + row["text"])[:remaining]
            if not value:
                break
            passages.append({"passage_id": row["id"], "cited_url": url,
                             "chunk_number": row["number"], "text": value})
            remaining -= len(value)
        return passages, [u for u, _ in missing]


GROUP_PROMPT = (ROOT / "prompts/judge_metrics/citation_support.md").read_text(encoding="utf-8") + """
For this call, several statements share the same set of cited URLs. The evidence below
contains only KB rows for those URLs. A cited URL absent from the KB supplies no evidence.
Judge each statement separately. Return JSON {"statements":[{"statement_id":0,
"entailed":true,"reason":"one sentence"}, ...]}. Include every statement_id once.
"""


class LimitedJudges(base.Judges):
    def __init__(self):
        base.OUT = OUT
        super().__init__()
        self.semaphores = {role: threading.Semaphore(8) for role in self.settings}

    def call(self, role, messages, **kwargs):
        with self.semaphores[role]:
            for attempt in range(4):
                try:
                    return super().call(role, messages, **kwargs)
                except Exception as exc:
                    if attempt == 3:
                        raise
                    time.sleep(min(2 ** attempt * 3, 20))


def score_grouped_citations(item, judges, evidence):
    statements = item.get("statements", [])
    groups = defaultdict(list)
    for i, s in enumerate(statements):
        urls = clean_urls(s.get("cited_urls", []))
        s["cited_urls"] = urls
        if urls:
            groups[tuple(sorted(urls))].append(i)
    for judge, role in (("primary", "judge_primary"), ("secondary", "judge_secondary")):
        results = []
        for urls, indices in groups.items():
            passages, missing = evidence.get(urls, [statements[i]["statement"] for i in indices])
            if passages:
                payload = {"statements": [{"statement_id": i, "statement": statements[i]["statement"]}
                                          for i in indices], "cited_urls": list(urls),
                           "cited_passages": passages}
                response = judges.call(role, [{"role": "system", "content": GROUP_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    json_mode=True, max_tokens=4096)["json"]
                labels = response.get("statements")
                if not isinstance(labels, list) or {x.get("statement_id") for x in labels} != set(indices):
                    raise ValueError("Invalid grouped citation response")
                lookup = {x["statement_id"]: x for x in labels}
            else:
                lookup = {i: {"entailed": False, "reason": "No cited URL is in the KB."} for i in indices}
            for i in indices:
                val = lookup[i]
                if type(val.get("entailed")) is not bool:
                    raise ValueError("Invalid grouped citation label")
                results.append({"statement_index": i, "cited_urls": list(urls),
                    "missing_kb_urls": missing, "selected_chunk_ids": [p["passage_id"] for p in passages],
                    "evidence_chars": sum(len(p["text"]) for p in passages),
                    "entailed": val["entailed"], "reason": val.get("reason", "")})
        results.sort(key=lambda x: x["statement_index"])
        item["metrics"].setdefault("citation_support", {})[judge] = {
            "score": sum(x["entailed"] for x in results) / len(results) if results else None,
            "cited_statements": results,
            "cited_url_not_in_kb": sum(len(x["missing_kb_urls"]) for x in results)}


def score_new(item, q, judges):
    answer = item["answer"]
    primary = judges.prompt("judge_primary", "appropriate_abstention",
        {"question": q["question"], "answer": answer},
        "\nAsking the user for more information instead of answering counts as abstains.")
    if primary.get("label") not in s1.ABSTENTION_LABELS:
        raise ValueError("Invalid abstention label")
    item["primary_abstention_label"] = primary["label"]
    for role, judge in (("judge_primary", "primary"), ("judge_secondary", "secondary")):
        labels = s1._ragchecker_run(role, {"id": item["id"], "condition": "API", "answer": answer,
            "passages": []}, q, "response2answer", settings=judges.settings[role],
            mock_llm=lambda r, m, **kw: v2.keypoint_call(judges, r, m, answer, q["question"], **kw))
        values = [str(x) for x in labels["values"]]
        summary = s1.keypoint_summary(q["keypoints"], values)
        item["metrics"].setdefault("answer_correctness", {})[judge] = summary["correct"]
        item["metrics"].setdefault("completeness", {})[judge] = summary["completeness"]
        item["judgements"].append({"metric": "answer_correctness_completeness", "judge": judge,
                                   "labels": values, "trace": labels["trace"]})
        relevance = judges.prompt(role, "response_relevance", {"question": q["question"], "answer": answer})
        if relevance.get("score") not in range(1, 6):
            raise ValueError("Invalid relevance label")
        abst = primary if judge == "primary" else judges.prompt(role, "appropriate_abstention",
            {"question": q["question"], "answer": answer},
            "\nAsking the user for more information instead of answering counts as abstains.")
        if abst.get("label") not in s1.ABSTENTION_LABELS:
            raise ValueError("Invalid secondary abstention label")
        item["metrics"].setdefault("response_relevance", {})[judge] = relevance
        item["metrics"].setdefault("appropriate_abstention", {})[judge] = abst
    if primary["label"] != "abstains":
        decomp_system = (base.PROMPTS / "statement_decomposition.md").read_text().split("## Output")[0] + base.DECOMP_EXTRA
        decomp = judges.call("judge_primary", [{"role": "system", "content": decomp_system},
            {"role": "user", "content": json.dumps({"answer": answer}, ensure_ascii=False)}], json_mode=True)["json"]
        item["statements"] = decomp.get("statements")
        if not isinstance(item["statements"], list):
            raise ValueError("Invalid statement decomposition")
        for role, judge in (("judge_primary", "primary"), ("judge_secondary", "secondary")):
            attr = judges.prompt(role, "citation_completeness", {"statements": [
                {"statement_id": i, "statement": x["statement"], "cited_urls": clean_urls(x.get("cited_urls", []))}
                for i, x in enumerate(item["statements"])]})
            rows = attr.get("statements", attr.get("results"))
            if not isinstance(rows, list) or len(rows) != len(item["statements"]):
                raise ValueError("Invalid citation completeness rows")
            complete = []
            for i, row in enumerate(rows):
                if type(row.get("needs_attribution")) is not bool:
                    raise ValueError("Invalid needs_attribution label")
                complete.append({"statement_index": i, "needs_attribution": row["needs_attribution"],
                    "has_cited_url": bool(clean_urls(item["statements"][i].get("cited_urls", []))),
                    "reason": row.get("reason", "")})
            need = [x for x in complete if x["needs_attribution"]]
            item["metrics"].setdefault("citation_completeness", {})[judge] = {
                "score": sum(x["has_cited_url"] for x in need) / len(need) if need else None,
                "statements": complete}
        if not item["metrics"]["answer_correctness"]["primary"]:
            alt = judges.call("judge_primary", [{"role": "system", "content": base.ALT_PROMPT},
                {"role": "user", "content": json.dumps({"question": q["question"],
                 "reference_answer": q["reference_answer"], "keypoints": q["keypoints"],
                 "answer": answer}, ensure_ascii=False)}], json_mode=True)["json"]
            if alt.get("alternative_answer") not in {"yes", "no"}:
                raise ValueError("Invalid alternative_answer label")
            item["alternative_answer"] = alt
    else:
        item["statements"] = []
        item["metrics"]["citation_completeness"] = {
            x: {"score": None, "status": "excluded"} for x in ("primary", "secondary")}
    return item


def main():
    questions, answers, ids, gate = cohort()
    snapshot = preflight(questions, answers, ids, gate)
    print("Preflight:", len(ids), "questions,", len(answers), "answers; EGI no-find",
          gate["later_egi_could_not_find_n"], "/", gate["later_egi_success_n"],
          "; estimated upper tokens", snapshot["estimated_tokens"]["total_upper"], flush=True)
    ITEMS.mkdir(parents=True, exist_ok=True)
    pages = raw_pages()
    evidence = Evidence(pages)
    judges = LimitedJudges()
    def process(a):
        path = ITEMS / f"{a['id']}__{a['model_name']}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("rule_version") == RULE_VERSION and saved.get("answer") == a["answer"]:
                return saved, "cached"
        pilot = PILOT / path.name
        if pilot.exists():
            item = json.loads(pilot.read_text())
            if item["answer"] != a["answer"]:
                raise ValueError("Pilot answer differs from current answer")
            item["reused_from_pilot_v2"] = True
        else:
            q = questions[a["id"]]
            item = {"id": a["id"], "model": a["model_name"], "type": q["type"],
                    "page_type": q["page_type"], "source_url": a["source_url"],
                    "question": q["question"], "reference_answer": q["reference_answer"],
                    "keypoints": q["keypoints"], "answer": a["answer"], "metrics": {},
                    "judgements": []}
            score_new(item, q, judges)
        score_grouped_citations(item, judges, evidence)
        item["rule_version"] = RULE_VERSION
        with WRITE_LOCK:
            jwrite(path, item)
        return item, "scored"
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(process, a): a for a in answers}
        for f in as_completed(futures):
            a = futures[f]
            try:
                item, status = f.result()
                results.append(item)
                print(len(results), "/", len(answers), status, a["model_name"], a["id"], flush=True)
            except Exception as exc:
                print("FAILED", a["model_name"], a["id"], type(exc).__name__, str(exc)[:200], flush=True)
    if len(results) != len(answers):
        raise RuntimeError(f"Only {len(results)}/{len(answers)} items scored; rerun to resume")
    print("All items scored", len(results), flush=True)


if __name__ == "__main__":
    main()
