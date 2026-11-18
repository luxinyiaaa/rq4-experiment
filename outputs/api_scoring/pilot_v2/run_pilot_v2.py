"""Re-score the frozen API pilot under the two keypoint clarifications."""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import api_scoring_pilot as base
from scripts import metrics_v2 as s1
from scripts.build_index import load_index
from refchecker.checker.checker_prompts import SUBSENTENCE_CLAIM_CHECKING_PROMPT

OLD = ROOT / "outputs/api_scoring/pilot"
OUT = ROOT / "outputs/api_scoring/pilot_v2"
ITEMS = OUT / "items"
KEYPOINT_RULES = (
    "Names that differ only in word form, abbreviation or word order refer to the same entity "
    "(for example 'Atmosphere Thematic Centre' and 'Atmospheric Thematic Centre').\n"
    "A keypoint that corrects a false premise ('X, not Y') is entailed if the answer says that "
    "the premise is wrong, or states X in a way that directly contradicts Y. It is not entailed "
    "if the answer only answers the rest of the question without stating anything that contradicts Y."
)
HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})
ITEM_LOCK = threading.Lock()
KEYPOINT_PROMPT_VERSION = "v2-separated-answer-evidence"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_url(url):
    value = str(url).strip().translate(HYPHENS)
    if not re.match(r"^[a-z]+://", value, re.I):
        value = "https://" + value
    parsed = urlsplit(value)
    return (parsed.netloc.casefold().removeprefix("www.") + parsed.path.rstrip("/")).casefold()


def old_items():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OLD / "items").glob("*.json"))]
    frozen = json.loads((OLD / "input_snapshot.json").read_text(encoding="utf-8"))
    pairs = {(x["id"], x["model"]) for x in rows}
    if len(rows) != 168 or pairs != {tuple(x) for x in frozen["answer_pairs"]}:
        raise ValueError("Existing pilot does not match its frozen 168-answer manifest")
    return rows


def item_path(row):
    return ITEMS / f"{row['id']}__{row['model']}.json"


def save_item(row):
    path = item_path(row)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


def keypoint_call(judges, role, messages, answer, question, **kwargs):
    original = str(messages[-1]["content"])
    match = re.search(r"### Claim:\n(.*?)\n\nYour answer", original, flags=re.S)
    if not match:
        raise ValueError("Could not locate RAGChecker keypoint in non-joint prompt")
    prompt = SUBSENTENCE_CLAIM_CHECKING_PROMPT.format(reference=answer, claim=match.group(1))
    prompt += ("\n\nQuestion context (not evidence):\n" + question +
               "\n\nAdditional keypoint rules:\n" + KEYPOINT_RULES +
               "\nOnly the answer text under Reference counts as evidence; "
               "the question may contain a false premise.")
    return judges.call(role, [{"role": "user", "content": prompt}], **kwargs)


def score_keypoints(old, judges):
    path = item_path(old)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("v2_status", {}).get("keypoints") and existing["v2_status"].get("keypoint_prompt_version") == KEYPOINT_PROMPT_VERSION:
            return existing
    row = deepcopy(old)
    row.pop("judge_usage", None)
    row["usage_logged"] = True
    row["v2_status"] = {"keypoints": False, "citations": False}
    question = {"question": old["question"], "reference_answer": old["reference_answer"],
                "keypoints": old["keypoints"], "type": old["type"]}
    for role, judge in (("judge_primary", "primary"), ("judge_secondary", "secondary")):
        result = s1._ragchecker_run(role, {"id": old["id"], "condition": "API",
            "answer": old["answer"], "passages": []}, question, "response2answer",
            settings=judges.settings[role], mock_llm=lambda r, m, **kw: keypoint_call(judges, r, m,
                old["answer"], old["question"], **kw))
        labels = [str(x) for x in result["values"]]
        summary = s1.keypoint_summary(old["keypoints"], labels)
        row["metrics"]["answer_correctness"][judge] = summary["correct"]
        row["metrics"]["completeness"][judge] = summary["completeness"]
        row["judgements"] = [j for j in row["judgements"] if not
            (j.get("metric") == "answer_correctness_completeness" and j.get("judge") == judge)]
        row["judgements"].append({"metric": "answer_correctness_completeness", "judge": judge,
                                   "labels": labels, "trace": result["trace"],
                                   "prompt_additions": KEYPOINT_RULES})
    row["v2_status"]["keypoints"] = True
    row["v2_status"]["keypoint_prompt_version"] = KEYPOINT_PROMPT_VERSION
    with ITEM_LOCK:
        save_item(row)
    return row


def old_label(row, judge):
    return next(j["labels"] for j in row["judgements"] if
                j.get("metric") == "answer_correctness_completeness" and j.get("judge") == judge)


def audit_missing_urls(originals, raw_urls):
    records = []
    for old in originals:
        statements = old.get("statements", [])
        for entry in old["metrics"]["citation_support"]["primary"].get("cited_statements", []):
            idx = entry["statement_index"]
            statement = statements[idx]["statement"]
            for url in entry.get("missing_kb_urls", []):
                fixed = norm_url(url)
                classification = "URL normalization error (Unicode hyphen)" if fixed in raw_urls else "Real URL outside KB"
                pos = old["answer"].find(url)
                context = old["answer"][max(0, pos-80):pos+len(url)+80] if pos >= 0 else statement
                records.append({"id": old["id"], "model": old["model"], "statement_index": idx,
                    "cited_url": url, "answer_context": context, "classification": classification,
                    "normalized_kb_url": raw_urls.get(fixed, ""), "statement": statement})
    with (OUT / "citation_url_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    return records


def fix_citations(row, judges, chunks, index):
    if row.get("v2_status", {}).get("citations"):
        return row
    statements = row.get("statements", [])
    for judge, role in (("primary", "judge_primary"), ("secondary", "judge_secondary")):
        metric = row["metrics"]["citation_support"][judge]
        scored = []
        for entry in metric.get("cited_statements", []):
            urls = entry["cited_urls"]
            selected, missing = base.evidence(chunks, index, statements[entry["statement_index"]]["statement"], urls)
            corrected = set(entry.get("missing_kb_urls", [])) - set(missing)
            if corrected:
                answer = judges.prompt(role, "citation_support", {"statement":
                    statements[entry["statement_index"]]["statement"], "cited_passages": [
                    {"passage_id": x["chunk_id"], "text": x["text"]} for x in selected]})
                entailed = answer.get("entailed")
                if type(entailed) is not bool:
                    raise ValueError("Invalid revised citation support label")
                entry = {**entry, "entailed": entailed, "reason": answer.get("reason"),
                         "selected_chunks": selected, "missing_kb_urls": missing,
                         "v2_reason": "Unicode-hyphen URL matched to KB; rejudged with selected chunks."}
            scored.append(entry)
        metric["cited_statements"] = scored
        metric["cited_url_not_in_kb"] = sum(len(x["missing_kb_urls"]) for x in scored)
        metric["score"] = sum(x["entailed"] for x in scored)/len(scored) if scored else None
    row["v2_status"]["citations"] = True
    with ITEM_LOCK:
        save_item(row)
    return row


def changes(originals, revised, judges):
    oldmap = {(x["id"], x["model"]): x for x in originals}
    rows = []
    for item in revised:
        old = oldmap[(item["id"], item["model"])]
        for judge, role in (("primary", "judge_primary"), ("secondary", "judge_secondary")):
            previous = old["metrics"]["answer_correctness"][judge]
            current = item["metrics"]["answer_correctness"][judge]
            if previous == current:
                continue
            before = old_label(old, judge)
            after = old_label(item, judge)
            changed_points = [{"keypoint": p["text"], "old": a, "new": b}
                              for p,a,b in zip(item["keypoints"],before,after) if a!=b]
            explanation = judges.call(role, [{"role": "system", "content":
                "Explain why the revised keypoint labels changed the correctness decision. "
                "Use the two added keypoint rules where applicable. Give one concise reason. "
                "Return JSON {\"reason\":\"...\"}."},
                {"role": "user", "content": json.dumps({"question": item["question"],
                    "answer": item["answer"], "changed_keypoints": changed_points,
                    "old_correct": previous, "new_correct": current,
                    "rules": KEYPOINT_RULES}, ensure_ascii=False)}], json_mode=True)["json"]
            rows.append({"id": item["id"], "model": item["model"], "judge": judge,
                "old_correct": previous, "new_correct": current,
                "changed_keypoints": json.dumps(changed_points, ensure_ascii=False),
                "reason": explanation.get("reason", "")})
    with (OUT / "correctness_changes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","model","judge","old_correct","new_correct","changed_keypoints","reason"])
        writer.writeheader(); writer.writerows(rows)
    return rows


def full_page_text(cited_urls, raw_pages):
    sections = []
    remaining = 30000
    for url in cited_urls:
        for chunk in raw_pages.get(norm_url(url), []):
            if remaining <= 0:
                break
            text = chunk["content"]
            value = text[:remaining]
            sections.append(value)
            remaining -= len(value)
    return "\n\n".join(sections)[:30000]


def disagreement_rows(items):
    rows = []
    for item in items:
        p = {x["statement_index"]: x for x in item["metrics"]["citation_support"]["primary"].get("cited_statements", [])}
        s = {x["statement_index"]: x for x in item["metrics"]["citation_support"]["secondary"].get("cited_statements", [])}
        for idx in sorted(p.keys() & s.keys()):
            if p[idx]["entailed"] != s[idx]["entailed"]:
                rows.append({"id": item["id"], "model": item["model"], "statement_index": idx,
                    "statement": item["statements"][idx]["statement"], "cited_urls": p[idx]["cited_urls"],
                    "selected_chunks": p[idx]["selected_chunks"],
                    "primary_label": p[idx]["entailed"], "primary_reason": p[idx]["reason"],
                    "secondary_label": s[idx]["entailed"], "secondary_reason": s[idx]["reason"]})
    return rows


def select_disagreements(rows):
    rng = random.Random(42)
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    for group in groups.values():
        group.sort(key=lambda x: (x["id"],x["statement_index"]))
        rng.shuffle(group)
    selected = []
    while len(selected) < 15:
        progressed = False
        for model in sorted(groups):
            if groups[model] and len(selected)<15:
                selected.append(groups[model].pop())
                progressed = True
        if not progressed:
            raise ValueError("Fewer than 15 citation disagreements")
    return selected


def rejudge_all_chunks(selected, raw_pages, judges):
    path = OUT / "all_chunks_rechecks.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for row in selected:
        key = f"{row['id']}__{row['model']}__{row['statement_index']}"
        if key not in saved or saved[key].get("evidence_chars", 0) > 30000:
            text = full_page_text(row["cited_urls"], raw_pages)
            result = judges.prompt("judge_primary", "citation_support", {"statement": row["statement"],
                "cited_passages": [{"passage_id": "ALL_CHUNKS", "text": text}]})
            if type(result.get("entailed")) is not bool:
                raise ValueError("Invalid all-chunks citation label")
            saved[key] = {"label": result["entailed"], "reason": result.get("reason", ""),
                          "evidence_chars": len(text)}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        row["all_chunks_label"] = saved[key]["label"]
        row["all_chunks_reason"] = saved[key]["reason"]
        row["changed_from_primary"] = row["all_chunks_label"] != row["primary_label"]
        row["evidence_chars"] = saved[key]["evidence_chars"]
    (OUT / "citation_disagreements.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
    return selected


def raw_page_map():
    grouped = defaultdict(list)
    with (ROOT / "data/raw/site_pages_rows.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[norm_url(row["url"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda x: int(x.get("chunk_number") or 0))
    return grouped


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ITEMS.mkdir(parents=True,exist_ok=True)
    originals = old_items()
    raw_pages = raw_page_map()
    raw_urls = {u: pages[0]["url"] for u,pages in raw_pages.items()}
    audit_missing_urls(originals,raw_urls)
    base.OUT = OUT
    base.norm_url = norm_url
    judges = base.Judges()
    with ThreadPoolExecutor(max_workers=6) as pool:
        revised = list(pool.map(lambda row: score_keypoints(row, judges), originals))
    print("Keypoints scored",len(revised),flush=True)
    index = load_index()
    chunks = base.raw_chunks(index)
    with ThreadPoolExecutor(max_workers=4) as pool:
        revised = list(pool.map(lambda row: fix_citations(row,judges,chunks,index),revised))
    print("Citations corrected",len(revised),flush=True)
    changed = changes(originals,revised,judges)
    disagreements = disagreement_rows(revised)
    sample = select_disagreements(disagreements)
    rejudge_all_chunks(sample,raw_pages,judges)
    base.report(revised,judges)
    print("Pilot v2 complete",len(changed),"correctness changes",len(disagreements),
          "statement disagreements",flush=True)


if __name__ == "__main__":
    main()
