"""Send the selected S1 questions to each Environmental Expert API model once."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "outputs" / "final" / "questions_final.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "expert_api" / "smoke"
SUMMARY_PATH = OUTPUT_DIR / "smoke_summary.md"
QUESTION_IDS = ("W047", "W029", "W068")
MODEL_NAMES = ("openai", "openai_gpt56_sol", "openai_gpt6_astra", "egi")
TIMEOUT_SECONDS = 300


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE entries without printing or exposing their values."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_questions() -> dict[str, dict[str, Any]]:
    questions = {
        item["id"]: item
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
        if item.get("id") in QUESTION_IDS
    }
    missing = [item_id for item_id in QUESTION_IDS if item_id not in questions]
    if missing:
        raise ValueError(f"Missing requested question IDs: {', '.join(missing)}")
    return questions


def json_shape(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def schema_lines(value: Any, path: str = "$", depth: int = 0) -> list[str]:
    """Describe JSON keys recursively, including object/list contents."""
    if depth > 8:
        return []
    lines: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lines.append(f"- `{child_path}`: {json_shape(child)}")
            lines.extend(schema_lines(child, child_path, depth + 1))
    elif isinstance(value, list):
        if not value:
            return lines
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            lines.append(f"- `{child_path}`: {json_shape(child)}")
            lines.extend(schema_lines(child, child_path, depth + 1))
    return lines


def flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for child in value.values() for text in flatten_strings(child)]
    if isinstance(value, list):
        return [text for child in value for text in flatten_strings(child)]
    return []


def find_values(value: Any, key_pattern: re.Pattern[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key_pattern.search(str(key)):
                found.append(child)
            found.extend(find_values(child, key_pattern))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_values(child, key_pattern))
    return found


def extract_answer(payload: Any) -> str:
    if isinstance(payload, dict):
        answer = payload.get("answer", "")
        return answer if isinstance(answer, str) else ""
    return ""


def contains_citation_marker(texts: list[str]) -> bool:
    patterns = (
        re.compile(r"\[[0-9]+\]"),
        re.compile(r"【[^】]+】"),
        re.compile(r"\((?:source|citation|ref)\s*[:#]", re.I),
    )
    return any(pattern.search(text) for text in texts for pattern in patterns)


def summarize_content(payload: Any) -> dict[str, Any]:
    key_names = [str(key).lower() for key in all_keys(payload)]
    strings = flatten_strings(payload)
    answer = extract_answer(payload)
    url_values = find_values(payload, re.compile(r"url|source", re.I))
    passage_values = find_values(payload, re.compile(r"passage|chunk|retriev|context|evidence", re.I))
    model_values = find_values(payload, re.compile(r"^model(_name)?$", re.I))
    non_answer_texts = [text for text in strings if text.strip() != answer.strip()]
    return {
        "retrieved_passages_or_chunks": any(re.search(r"passage|chunk|retriev|context|evidence", key) for key in key_names),
        "source_urls": bool(url_values) or any(text.startswith(("http://", "https://")) for text in strings),
        "citation_markers": contains_citation_marker(strings),
        "model_name_present": bool(model_values),
        "model_names": model_values,
        "other_content_beyond_answer_text": bool(non_answer_texts) or len(payload) > 1 if isinstance(payload, dict) else bool(non_answer_texts),
        "answer_text": answer,
    }


def all_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key, child in value.items()] + [key for child in value.values() for key in all_keys(child)]
    if isinstance(value, list):
        return [key for child in value for key in all_keys(child)]
    return []


def redact_endpoint(message: str, endpoint: str) -> str:
    return message.replace(endpoint, "[endpoint redacted]") if endpoint else message


def call_api(endpoint: str, query: str, model_name: str) -> tuple[int | None, float, Any, str | None]:
    start = time.monotonic()
    try:
        response = requests.post(
            endpoint,
            json={"query": query, "model_name": model_name},
            timeout=TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - start
        try:
            payload = response.json()
        except requests.exceptions.JSONDecodeError:
            payload = {"raw_response_text": response.text}
        if response.ok:
            return response.status_code, elapsed, payload, None
        return response.status_code, elapsed, payload, f"HTTP {response.status_code}: {response.reason}"
    except requests.RequestException as error:
        elapsed = time.monotonic() - start
        # Do not serialize exception text: requests exceptions can include the request URL.
        detail = "timeout" if isinstance(error, requests.Timeout) else "connection or request error"
        return None, elapsed, {"error": type(error).__name__}, f"{type(error).__name__}: {detail}"


def markdown_cell(value: Any) -> str:
    text = str(value or "")
    return text.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def run() -> None:
    env_values = load_env_file(ROOT / ".env")
    endpoint = env_values.get("ENVRI_EXPERT_URL", "")
    if not endpoint.strip():
        raise SystemExit("ENVRI_EXPERT_URL is missing from .env; no API calls were made.")

    questions = load_questions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    for item_id in QUESTION_IDS:
        question = questions[item_id]
        for model_name in MODEL_NAMES:
            status, elapsed, payload, error = call_api(endpoint, question["question"], model_name)
            response_path = OUTPUT_DIR / f"{item_id}_{model_name}.json"
            response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            details = summarize_content(payload)
            calls.append({
                "id": item_id,
                "model": model_name,
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "status": status,
                "elapsed": elapsed,
                "error": error,
                "path": response_path.name,
                "schema": schema_lines(payload),
                **details,
            })

    success_count = sum(call["status"] is not None and 200 <= call["status"] < 300 and not call["error"] for call in calls)
    failure_count = len(calls) - success_count

    lines = [
        "# Environmental Expert API Smoke Test",
        "",
        f"Successful calls: {success_count} of {len(calls)}.",
        f"Failed calls: {failure_count} of {len(calls)}.",
        "",
        "Each request was sent once, sequentially, with a 300-second timeout. Endpoint and credential values are intentionally omitted.",
        "",
        "## Calls",
        "",
        "| Question | Model requested | HTTP status | Response time (s) | Error | Raw response |",
        "|---|---|---:|---:|---|---|",
    ]
    for call in calls:
        lines.append(
            f"| {call['id']} | {call['model']} | {call['status'] if call['status'] is not None else 'N/A'} | "
            f"{call['elapsed']:.2f} | {markdown_cell(call['error'] or '')} | `{call['path']}` |"
        )
    for call in calls:
        lines.extend(["", f"### {call['id']} — {call['model']}", ""])
        lines.append("JSON structure:")
        lines.extend(call["schema"] or ["- No JSON keys (response was not an object or list)."])
        lines.extend([
            "",
            "Response contents:",
            f"- Retrieved passages or chunks: {'yes' if call['retrieved_passages_or_chunks'] else 'no'}",
            f"- Source URLs: {'yes' if call['source_urls'] else 'no'}",
            f"- Citation markers: {'yes' if call['citation_markers'] else 'no'}",
            f"- Model name in response: {markdown_cell(json.dumps(call['model_names'], ensure_ascii=False)) if call['model_name_present'] else 'no'}",
            f"- Other content beyond answer text: {'yes' if call['other_content_beyond_answer_text'] else 'no'}",
        ])
    lines.extend(["", "## Questions and Answers", ""])
    for item_id in QUESTION_IDS:
        question = questions[item_id]
        lines.extend([
            f"### {item_id}",
            "",
            f"**Question:** {markdown_cell(question['question'])}",
            "",
            f"**Reference answer:** {markdown_cell(question['reference_answer'])}",
            "",
        ])
        for call in (c for c in calls if c["id"] == item_id):
            lines.extend([f"**{call['model']} answer:** {markdown_cell(call['answer_text']) or '(no answer text field found)'}", ""])

    SUMMARY_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if failure_count:
        print(f"Smoke run finished with {success_count} successful and {failure_count} failed calls. Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    else:
        print(f"All {success_count} sequential calls succeeded. Summary: {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
