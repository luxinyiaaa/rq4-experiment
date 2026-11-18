"""Check scored items and aggregate/review outputs without modifying inputs."""
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
snapshot = json.loads((OUT / "input_snapshot.json").read_text())
items = [json.loads((OUT / "items" / (aid + ".json")).read_text()) for aid in snapshot["answer_ids"]]
assert len(items) == snapshot["answer_count"]
assert len({(x["id"], x["model"]) for x in items}) == len(items)
assert all(x["rule_version"] == snapshot["rule_version"] for x in items)
for item in items:
    assert item["metrics"]["answer_correctness"].keys() >= {"primary", "secondary"}
    assert item["metrics"]["completeness"].keys() >= {"primary", "secondary"}
    for judge in ("primary", "secondary"):
        score = item["metrics"]["citation_support"][judge]
        rows = score.get("cited_statements", [])
        assert all(x["evidence_chars"] <= 30000 for x in rows)
        assert all(type(x["entailed"]) is bool for x in rows)
        assert len({x["statement_index"] for x in rows}) == len(rows)
        expected = sum(x["entailed"] for x in rows) / len(rows) if rows else None
        assert score["score"] == expected
        assert score["cited_url_not_in_kb"] == sum(len(x["missing_kb_urls"]) for x in rows)
sample = json.loads((OUT / "human_validation" / "human_validation_sample.json").read_text())
assert len(sample["question_ids"]) == len(set(sample["question_ids"])) == 40
assert set(sample["assigned_models"].values()) == {10}
key = list(csv.DictReader((OUT / "human_validation" / "human_validation_key.csv").open()))
assert len(key) == 40 and len({x["sample_id"] for x in key}) == 40
for name in ("human_validation/human_validation.xlsx", "alternative_answers.xlsx",
             "human_validation/alternative_answers_in_sample.xlsx"):
    with zipfile.ZipFile(OUT / name) as z:
        assert z.testzip() is None
        assert "xl/workbook.xml" in z.namelist()
report = (OUT / "report.md").read_text()
assert not re.search(r"(?:api[_-]?key|sk-[A-Za-z0-9]{8,}|openrouter\.ai/api)", report, re.I)
print("Verified", len(items), "items and", len(key), "blind samples")
