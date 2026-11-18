from __future__ import annotations

import pytest

from scripts.check_questions import page_usage_violations, partner_gold_title_violations, page_text_catalog, template_answer_check
from scripts.generate_questions import _user_message, duplicate_title_note
from scripts.prepare_main_round2 import apply_edited_quotes


def test_edited_quotes_validate_verbatim_and_recompute_gold_and_partner_ids():
    question = {
        "id": "W001",
        "text": "Alpha fact. Beta fact.",
        "chunk_ids": ["c1", "c2"],
        "partner": {"text": "Partner fact.", "chunk_ids": ["p1"]},
    }
    apply_edited_quotes(
        question,
        "[passage] Beta fact.\n[partner] Partner fact.",
        {"c1": "Alpha fact. ", "c2": "Beta fact.", "p1": "Partner fact."},
    )
    assert question["gold_chunk_ids"] == ["c2"]
    assert question["partner_chunk_ids"] == ["p1"]
    assert all(quote["matched"] for quote in question["supporting_quotes"])


def test_edited_quote_rejects_nonverbatim_line_without_partial_mutation():
    question = {"id": "W002", "text": "A fact.", "chunk_ids": ["c1"], "gold_chunk_ids": ["old"]}
    with pytest.raises(ValueError, match="not an exact substring"):
        apply_edited_quotes(question, "[passage] A fact!", {"c1": "A fact."})
    assert question["gold_chunk_ids"] == ["old"]


def test_page_usage_and_partner_title_rules_report_only_cross_question_violations():
    rows = [
        {"id": "Q1", "url": "gold-1", "partner": {"url": "partner-1"}},
        {"id": "Q2", "url": "gold-2", "partner": {"url": "partner-1"}},
    ]
    assert page_usage_violations(rows) == {"partner-1": [
        {"question_id": "Q1", "role": "partner"},
        {"question_id": "Q2", "role": "partner"},
    ]}
    assert partner_gold_title_violations(rows, {
        "gold-1": "A", "gold-2": "B", "partner-1": "A",
    }) == [{
        "question_id": "Q2", "partner_url": "partner-1", "title": "A", "other_gold_id": "Q1",
    }]


def test_duplicate_title_note_is_inserted_into_teacher_user_message():
    item = {
        "type": "comparison", "url": "gold", "text": "Gold text",
        "partner": {"url": "partner", "text": "Partner text"},
    }
    titles = {"gold": "Shared title", "partner": "Partner title"}
    by_title = {"Shared title": ["gold", "other"]}
    note = duplicate_title_note(item, titles, by_title)
    assert note.startswith("NOTE: Another record in the knowledge base has the same title as the passage.")
    message = _user_message({**item, "duplicate_title_note": note})
    assert note in message
    assert "other does not share" in message


def test_template_answer_check_normalizes_whitespace_and_excludes_own_pages():
    question = {
        "id": "Q1", "url": "gold", "partner": {"url": "partner"},
        "supporting_quotes": [
            {"source": "passage", "text": "A complete\n factual statement."},
        ],
    }
    result = template_answer_check(question, {
        "gold": "A complete factual statement.",
        "partner": "A complete factual statement.",
        "copy-1": "prefix A complete\n factual statement. suffix",
        "copy-2": "A complete   factual statement.",
    })
    assert result["status"] == "checked"
    assert result["flagged"] is True
    assert result["matching_quotes"][0]["urls"] == ["copy-1", "copy-2"]


def test_template_answer_check_requires_two_other_pages():
    result = template_answer_check({
        "url": "gold", "supporting_quotes": [{"text": "Unique evidence."}],
    }, {"gold": "Unique evidence.", "copy-1": "Unique evidence."})
    assert result["flagged"] is False


def test_page_text_catalog_joins_chunks_in_numeric_order_without_added_spaces():
    pages = page_text_catalog([
        {"url": "u", "chunk_id": "12", "text": "world"},
        {"url": "u", "chunk_id": "2", "text": "hello "},
    ])
    assert pages["u"] == "hello world"
