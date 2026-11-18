"""Complete retrieval-only and aggregate RQ4 supplemental analyses."""
from __future__ import annotations

import csv
import json
import math
import re
from html import escape
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

import pandas as pd

from scripts.build_index import load_index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def exact_mcnemar(original: list[bool], rewritten: list[bool]) -> tuple[int, int, float]:
    b = sum(a and not z for a, z in zip(original, rewritten))
    c = sum((not a) and z for a, z in zip(original, rewritten))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return b, c, min(1.0, 2.0 * tail)


def numeric_score(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_inline_xlsx(path: Path) -> list[list[str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(si.itertext()) for si in root.findall("m:si", ns)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    matrix: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        cells: dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            ref = re.match(r"([A-Z]+)", str(cell.attrib.get("r", "")))
            if not ref:
                continue
            column = 0
            for char in ref.group(1):
                column = column * 26 + ord(char) - 64
            if cell.attrib.get("t") == "inlineStr":
                inline = cell.find("m:is", ns)
                value = "".join(inline.itertext()) if inline is not None else ""
            else:
                raw = cell.findtext("m:v", default="", namespaces=ns)
                value = shared[int(raw)] if cell.attrib.get("t") == "s" and raw else raw
            cells[column - 1] = value
        matrix.append([cells.get(i, "") for i in range(max(cells, default=-1) + 1)])
    return matrix


def column_name(number: int) -> str:
    name = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def write_minimal_xlsx(headers: list[str], records: list[list[str]], path: Path, sheet_name: str) -> None:
    rows = [headers] + records
    sheet_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(values, start=1):
            address = f"{column_name(column_number)}{row_number}"
            cells.append(f'<c r="{address}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    worksheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData><autoFilter ref="A1:{column_name(len(headers))}{len(rows)}"/></worksheet>')
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>')
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        archive.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


def paraphrase_retrieval() -> list[dict[str, Any]]:
    path = OUT / "paraphrase_control.xlsx"
    matrix = read_inline_xlsx(path)
    if not matrix:
        raise ValueError("Paraphrase workbook is empty")
    headers = matrix[0]
    col = {str(name): i + 1 for i, name in enumerate(headers) if name is not None}
    required = {"id", "original_question", "source_worded_question", "human_decision", "human_note"}
    if not required.issubset(col):
        raise ValueError(f"Paraphrase workbook is missing columns: {sorted(required - set(col))}")
    if "answer_in_question" not in col:
        col["answer_in_question"] = len(headers) + 1
        headers.append("answer_in_question")

    rows: dict[str, dict[str, Any]] = {}
    for row_number, raw_row in enumerate(matrix[1:], start=2):
        raw_row.extend([""] * (len(headers) - len(raw_row)))
        qid = str(raw_row[col["id"] - 1] or "").strip()
        if not qid:
            continue
        decision = "edit" if qid == "W009" else "accept"
        note = str(raw_row[col["human_note"] - 1] or "")
        source_q = str(raw_row[col["source_worded_question"] - 1] or "")
        if qid == "W009":
            match = re.search(r"Replace source_worded_question with:\s*(.+?)\s*$", note, re.S)
            if not match:
                raise ValueError("W009 human_note does not contain the requested replacement question")
            source_q = match.group(1).strip()
            if not source_q:
                raise ValueError("W009 replacement question is empty")
            raw_row[col["source_worded_question"] - 1] = source_q
        raw_row[col["human_decision"] - 1] = decision
        answer_in_question = qid == "W017"
        raw_row[col["answer_in_question"] - 1] = str(answer_in_question)
        rows[qid] = {
            "row_number": row_number,
            "source_question": source_q,
            "answer_in_question": answer_in_question,
            "human_decision": decision,
        }
    if len(rows) != 14:
        raise ValueError(f"Expected 14 paraphrase rows, found {len(rows)}")
    write_minimal_xlsx(headers, matrix[1:], path, "Paraphrase control")

    questions = {row["id"]: row for row in read_jsonl(OUT / "questions_final.jsonl")}
    splits = {row["id"]: str(row["quote_split"]).lower() == "true" for row in csv.DictReader((OUT / "quote_split_diagnostic.csv").open(encoding="utf-8", newline=""))}
    index = load_index(ROOT / "data" / "processed" / "index")
    results: list[dict[str, Any]] = []
    for qid, edited in rows.items():
        q = questions[qid]
        gold_chunks = {str(chunk) for chunk in q.get("gold_chunk_ids", [])}
        gold_url = str(q.get("url", ""))
        for variant, query in (("original", str(q.get("question", ""))), ("source_worded", edited["source_question"])):
            hits = index.retrieve(query, k=10)
            chunk_ranks = [i for i, hit in enumerate(hits, 1) if str(hit["chunk_id"]) in gold_chunks]
            page_ranks = [i for i, hit in enumerate(hits, 1) if str(hit["url"]) == gold_url]
            results.append({
                "id": qid,
                "variant": variant,
                "quote_split": splits.get(qid, False),
                "answer_in_question": edited["answer_in_question"],
                "query": query,
                "top10_chunk_ids": [str(hit["chunk_id"]) for hit in hits],
                "top10_urls": [str(hit["url"]) for hit in hits],
                "gold_chunk_hit": bool(chunk_ranks),
                "gold_chunk_rank": min(chunk_ranks) if chunk_ranks else None,
                "gold_page_hit": bool(page_ranks),
                "gold_page_rank": min(page_ranks) if page_ranks else None,
            })
    write_jsonl(OUT / "paraphrase_retrieval_results.jsonl", results)

    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_id[row["id"]][row["variant"]] = row
    summary_rows = []
    group_specs = [("all_14", list(by_id)), ("excluding_W017", [qid for qid in by_id if qid != "W017"])]
    for split in (False, True):
        group_specs.append((f"quote_split_{str(split).lower()}", [qid for qid in by_id if splits.get(qid, False) == split]))
    for group, qids in group_specs:
        for measure in ("gold_chunk_hit", "gold_page_hit"):
            orig = [bool(by_id[qid]["original"][measure]) for qid in qids]
            reword = [bool(by_id[qid]["source_worded"][measure]) for qid in qids]
            b, c, p = exact_mcnemar(orig, reword)
            summary_rows.append({
                "group": group, "measure": measure, "n": len(qids),
                "original_hits": sum(orig), "source_worded_hits": sum(reword),
                "original_hit_source_miss": b, "original_miss_source_hit": c,
                "exact_mcnemar_p": p,
            })
    pd.DataFrame(summary_rows).to_csv(OUT / "paraphrase_retrieval_summary.csv", index=False)
    return summary_rows


def manual_check_and_disagreements() -> tuple[Counter[str], int, Counter[str]]:
    score_path = OUT / "scores_final_revised.jsonl"
    scores = read_jsonl(score_path)
    rule_counts: Counter[str] = Counter()
    flagged = 0
    directions: Counter[str] = Counter()
    for score in scores:
        rag = score.get("ragchecker") or {}
        original = rag.get("correctness")
        revised = score.get("correctness_v2")
        refusal = (score.get("abstention") or {}).get("label")
        flask = (score.get("flask") or {}).get("score")
        recall = numeric_score(score.get("context_recall"))
        rules = []
        if isinstance(original, bool) and isinstance(flask, int) and ((original and flask <= 3) or (not original and flask >= 4)):
            rules.append("correctness_flask_disagreement")
        if refusal == "abstains" and original is True:
            rules.append("abstains_but_ragchecker_correct")
        if score.get("condition") == "C4" and refusal == "answers" and recall is not None and math.isclose(recall, 0.0):
            rules.append("C4_answers_context_recall_zero")
        if isinstance(original, bool) and isinstance(revised, bool) and original != revised:
            rules.append("correctness_v2_disagreement")
            directions[f"{str(original).lower()}_to_{str(revised).lower()}"] += 1
        score["needs_manual_check_rules"] = rules
        score["needs_manual_check"] = bool(rules)
        rule_counts.update(rules)
        flagged += bool(rules)
    write_jsonl(score_path, scores)
    write_jsonl(OUT / "scores_final.jsonl", scores)

    needs = [{
        "id": s["id"], "condition": s["condition"], "type": s.get("type"),
        "rules": "; ".join(s["needs_manual_check_rules"]),
        "question": s.get("question", ""), "answer": s.get("answer", ""),
        "correctness": (s.get("ragchecker") or {}).get("correctness"),
        "correctness_v2": s.get("correctness_v2"),
        "flask": (s.get("flask") or {}).get("score"),
        "abstention": (s.get("abstention") or {}).get("label"),
        "context_recall": s.get("context_recall"),
    } for s in scores if s["needs_manual_check"]]
    pd.DataFrame(needs).to_csv(OUT / "needs_manual_check_revised.csv", index=False)
    pd.DataFrame(needs).to_csv(OUT / "needs_manual_check_final.csv", index=False)

    a2r = {(r["id"], r["condition"]): r for r in read_jsonl(OUT / "answer2response_checks.jsonl")}
    disagreements = []
    for score in scores:
        if not (isinstance((score.get("ragchecker") or {}).get("correctness"), bool)
                and isinstance(score.get("correctness_v2"), bool)
                and (score.get("ragchecker") or {}).get("correctness") != score.get("correctness_v2")):
            continue
        row = a2r[(score["id"], score["condition"])]
        labels = list(row.get("answer2response_labels", []))
        claims = list(row.get("answer_claims", []))
        label_traces = [trace for trace in row.get("trace", []) if trace.get("call_type") == "label"]
        if len(labels) != len(claims) or len(label_traces) != len(labels):
            raise ValueError(f"Cannot align answer2response evidence for {score['id']} {score['condition']}")
        evidence = [
            {"answer_statement": claim, "judge_raw_return": trace.get("raw_response")}
            for claim, label, trace in zip(claims, labels, label_traces)
            if str(label).casefold() == "contradiction"
        ]
        disagreements.append({
            "id": score["id"], "condition": score["condition"],
            "original_correctness": (score.get("ragchecker") or {}).get("correctness"),
            "correctness_v2": score.get("correctness_v2"),
            "extra_contradiction": score.get("extra_contradiction"),
            "contradiction_count": len(evidence),
            "contradiction_evidence": json.dumps(evidence, ensure_ascii=False),
        })
    pd.DataFrame(disagreements).to_csv(OUT / "correctness_disagreements.csv", index=False)
    return rule_counts, flagged, directions


def type_quote_split_table() -> pd.DataFrame:
    questions = {q["id"]: q for q in read_jsonl(OUT / "questions_final.jsonl")}
    split_map = {
        row["id"]: str(row["quote_split"]).lower() == "true"
        for row in csv.DictReader((OUT / "quote_split_diagnostic.csv").open(encoding="utf-8", newline=""))
    }
    c4 = [s for s in read_jsonl(OUT / "scores_final_revised.jsonl") if s["condition"] == "C4"]
    cells: dict[tuple[str, bool], list[bool]] = defaultdict(list)
    for score in c4:
        q = questions[score["id"]]
        cells[(str(q.get("type", "")), split_map.get(score["id"], False))].append(bool((score.get("ragchecker") or {}).get("correctness")))
    rows = []
    for (qtype, split), values in sorted(cells.items()):
        rows.append({"type": qtype, "quote_split": split, "n": len(values), "correct": sum(values), "accuracy": sum(values) / len(values) if values else None})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "type_quote_split_c4_correctness.csv", index=False)
    return frame


def main() -> None:
    paraphrase_rows = paraphrase_retrieval()
    rule_counts, unique_flagged, directions = manual_check_and_disagreements()
    type_split = type_quote_split_table()
    print("Paraphrase retrieval (retrieval only):")
    print(pd.DataFrame(paraphrase_rows).to_string(index=False))
    print("\nNeeds-manual-check counts:")
    for rule in (
        "correctness_flask_disagreement", "abstains_but_ragchecker_correct",
        "C4_answers_context_recall_zero", "correctness_v2_disagreement",
    ):
        print(f"{rule}: {rule_counts[rule]}")
    print(f"Unique flagged answers: {unique_flagged}")
    print("\nCorrectness v2 discrepancy directions:")
    for direction, count in sorted(directions.items()):
        print(f"{direction}: {count}")
    print("\nQuestion type x quote_split x C4 original RAGChecker correctness:")
    print(type_split.to_string(index=False))
    print("\nUpdated English outputs: paraphrase_control.xlsx, paraphrase_retrieval_results.jsonl, paraphrase_retrieval_summary.csv, needs_manual_check_revised.csv, needs_manual_check_final.csv, correctness_disagreements.csv, type_quote_split_c4_correctness.csv")


if __name__ == "__main__":
    main()
