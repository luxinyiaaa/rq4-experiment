"""Fetch selected raw-page pilot URLs through headless Chromium."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE_CSV = ROOT / "outputs/rawpage_questions/pilot/pilot_pages.csv"
OUT = ROOT / "data/raw_web/pilot_v2"
TARGETS = {"RP04", "RP05", "RP07", "RP08"}


def main() -> None:
    with PAGE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = {row["page_id"]: row for row in csv.DictReader(stream) if row["page_id"] in TARGETS}
    if set(rows) != TARGETS:
        raise ValueError("The pilot page report is missing one or more requested page IDs")
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for page_id in sorted(TARGETS):
            source_url = rows[page_id]["url"]
            page = browser.new_page(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
            ))
            main_responses = []
            page.on("response", lambda response: main_responses.append(response) if response.request.is_navigation_request() else None)
            error = None
            response_status = None
            try:
                response = page.goto(source_url, wait_until="networkidle", timeout=60000)
                response_status = response.status if response is not None else None
            except PlaywrightTimeoutError:
                error = "networkidle_timeout_60000ms"
                if main_responses:
                    response_status = main_responses[-1].status
            except Exception as exc:
                error = type(exc).__name__
                if main_responses:
                    response_status = main_responses[-1].status
            try:
                rendered_html = page.content()
            except Exception:
                rendered_html = ""
            (OUT / f"{page_id}_browser.html").write_text(rendered_html, encoding="utf-8")
            metadata = {
                "page_id": page_id,
                "original_url": source_url,
                "final_url": page.url,
                "http_status": response_status,
                "fetch_time_utc": datetime.now(timezone.utc).isoformat(),
                "wait_until": "networkidle",
                "timeout_ms": 60000,
                "error": error,
            }
            (OUT / f"{page_id}_browser.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
