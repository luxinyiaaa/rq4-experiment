"""Construct reproducible S2 evidence bundles from the revised catalogues."""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from rq4.llm import call_llm
from scripts.build_kb_index import OUT as INDEX_DIR, parse_time_interval, read_jsonl, write_jsonl
from scripts.sample_windows import page_windows

PROMPT = ROOT / "prompts/judge_link_screen.md"
PAGES_PATH = ROOT / "data/processed/pages.csv"
CHUNKS_PATH = ROOT / "data/processed/chunks_clean.csv"
S1_PATH = ROOT / "outputs/final/questions_final.jsonl"
TYPE_ORDER = ("set_all", "multi_hop", "comparison_multi", "ambiguity", "unanswerable")
RELEVANT_FIELD = re.compile(r"country|geographical|area|holding centre|station|variable|parameter|instrument|ecosystem|time period|start time|stop time|depth|height|coverages?|latitude|longitude|region", re.I)
EXCLUDED_FIELD = re.compile(r"identifier|\bid\b|licen[cs]e|file.?name|file.?size|version|\bpid\b|\bdoi\b", re.I)
CLASSIFICATION_FIRST = re.compile(r"country|geographical area|data holding centre|parameters?|station|network", re.I)
RELATED_FIELDS = ("geographical area", "parameter category", "parameters", "data holding centre", "part of", "programme")
COMPARABLE_FIELD = re.compile(r"time|period|start|stop|end|year|depth|height|elevation|area|coverage", re.I)
WINDOW_CACHE: dict[str, list[dict[str, Any]]] = {}
WINDOW_OPTIONS: dict[str, list[dict[str, Any]]] = {}


def _window(frame: pd.DataFrame, needle: str = "") -> dict[str, Any] | None:
    url = str(frame.iloc[0].get("url", ""))
    if url not in WINDOW_OPTIONS: WINDOW_OPTIONS[url] = page_windows(frame)
    if url not in WINDOW_CACHE: WINDOW_CACHE[url] = WINDOW_OPTIONS[url]
    opts = WINDOW_CACHE[url]
    if needle:
        found = [w for w in opts if needle.casefold() in w["text"].casefold()]
        if found: return found[0]
    return opts[0] if opts else None


def _field_window(frame: pd.DataFrame, field: str, value: str) -> dict[str, Any] | None:
    url = str(frame.iloc[0].get("url", ""))
    if url not in WINDOW_OPTIONS: WINDOW_OPTIONS[url] = page_windows(frame)
    field_re = re.escape(re.sub(r"\s+", " ", field)).replace(r"\ ", r"\s+")
    value_re = re.escape(re.sub(r"\s+", " ", value)).replace(r"\ ", r"\s+")
    adjacent = re.compile(rf"(?is){field_re}(?:\s*\([^)]*\))?\s*(?:\||:)?\s*{value_re}")
    matches = [w for w in WINDOW_OPTIONS[url] if adjacent.search(w["text"])]
    return matches[0] if matches else None


def _year_overlap(interval: dict[str, Any], start_year: int, end_year: int | None = None) -> bool:
    start, end = interval.get("start_year"), interval.get("end_year")
    wanted_end = end_year if end_year is not None else start_year
    if start is None:
        return False
    return start <= wanted_end and (end is None or end >= start_year)


def _record_interval(page: dict[str, Any], field: str) -> dict[str, Any]:
    return parse_time_interval(str(page.get("fields", {}).get(field, "")))


def _comparison_value_in_any_title(values: list[str], titles: list[str]) -> bool:
    normalized_titles = [str(title).casefold() for title in titles]
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        tokens = [text.casefold()]
        tokens.extend(re.findall(r"\b(?:18|19|20|21)\d{2}\b", text))
        tokens.extend(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?", text))
        for title in normalized_titles:
            if any(token and token.casefold() in title for token in tokens):
                return True
    return False


def _parse_comparison_value(field: str, value: str) -> dict[str, Any] | None:
    text = str(value).strip()
    if not text or not COMPARABLE_FIELD.search(field):
        return None
    if re.search(r"time|period|start|stop|end|year|coverage", field, re.I):
        interval = parse_time_interval(text)
        if interval.get("start_year") is not None or interval.get("end_year") is not None:
            return {"kind": "interval", **interval}
        return None
    number = re.search(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?", text)
    if not number:
        return None
    try:
        numeric = float(number.group(0).replace(",", "."))
    except ValueError:
        return None
    unit = text[number.end():].strip().casefold()
    return {"kind": "numeric", "value": numeric, "unit": unit}


def _shared_comparison_relation(pages: list[dict[str, Any]]) -> tuple[str, str] | None:
    allowed = ("geographical area", "parameter category", "parameters", "data holding centre", "holding institution", "programme", "part of")
    fields = set.intersection(*(set(page.get("fields", {})) for page in pages)) if pages else set()
    for field in sorted(fields, key=str.casefold):
        if not any(alias in field.casefold() for alias in allowed):
            continue
        values = [re.sub(r"\s+", " ", str(page.get("fields", {}).get(field, "")).strip()).casefold() for page in pages]
        if not all(values):
            continue
        if all(value == values[0] for value in values[1:]):
            return field, str(pages[0].get("fields", {}).get(field, "")).strip()
        if "geographical area" in field.casefold() and all(a in b or b in a for a, b in itertools.combinations(values, 2)):
            return field, str(pages[0].get("fields", {}).get(field, "")).strip()
    return None


def _page_field_map(page_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["url"]: row for row in page_rows}


def _claim_window(frame: pd.DataFrame, quote: str) -> dict[str, Any] | None:
    url = str(frame.iloc[0].get("url", ""))
    if url not in WINDOW_OPTIONS: WINDOW_OPTIONS[url] = page_windows(frame)
    for option in WINDOW_OPTIONS[url]:
        if quote and quote in option["text"]:
            return option
    return None


def _target_window(frame: pd.DataFrame, entity: str) -> dict[str, Any] | None:
    url = str(frame.iloc[0].get("url", ""))
    if url not in WINDOW_OPTIONS: WINDOW_OPTIONS[url] = page_windows(frame)
    options = WINDOW_OPTIONS[url]
    if not options: return None
    def score(option: dict[str, Any]) -> tuple[int, int, int]:
        text = option["text"]
        fact_score = len(re.findall(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b|\b(?:located|based) in\b|\binstrument\b|\bmember\b", text, re.I))
        entity_score = int(entity.casefold() in text.casefold())
        prose = len(re.findall(r"\b[A-Za-z]{3,}\b", re.sub(r"https?://\S+|!\[[^]]*\]\([^)]*\)", "", text)))
        return entity_score, fact_score, prose
    return max(options, key=score)


def _target_windows(frame: pd.DataFrame, entity: str, limit: int = 2) -> list[dict[str, Any]]:
    url = str(frame.iloc[0].get("url", ""))
    if url not in WINDOW_OPTIONS: WINDOW_OPTIONS[url] = page_windows(frame)
    options = [w for w in WINDOW_OPTIONS[url] if entity.casefold() in w["text"].casefold()
               and not re.search(r"latest programme news|conference|event logistics", w["text"], re.I)]
    if not options:
        options = WINDOW_OPTIONS[url]
    def score(option: dict[str, Any]) -> tuple[int, int, int]:
        text = option["text"]
        facts = len(re.findall(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\b|\b(?:located|based) in\b|\binstrument\b|\bmember\b|\bwill have\b|\bby\s+20\d{2}\b", text, re.I))
        salient = len(re.findall(r"\b(?:goal|advance|tool|fund|benefit|cost|maturity|robustness|located|member|instrument)\w*\b", text, re.I))
        return facts, salient, len(re.findall(r"[A-Za-z]{3,}", text))
    ordered = sorted(options, key=score, reverse=True)
    selected=[]; seen=set()
    for option in ordered:
        signature=re.sub(r"\s+"," ",option["text"]).casefold()
        if signature in seen: continue
        seen.add(signature); selected.append(option)
        if len(selected)>=limit: break
    return selected


def _s1_map() -> dict[str, list[str]]:
    out = {}
    for q in read_jsonl(S1_PATH):
        for p in (q, q.get("partner")):
            if p and p.get("url"): out.setdefault(p["url"], []).append(q.get("question", ""))
    return out


def _page_record(page: dict[str, Any], window: dict[str, Any], s1: dict[str, list[str]]) -> dict[str, Any]:
    return {"url": page["url"], "title": page.get("record_name") or page["title"], "page_type": page["page_type"],
            "url_prefix": page.get("url_prefix", ""),
            "chunk_ids": [str(x) for x in window["chunk_ids"]], "text": window["text"],
            "length_chars": len(window["text"]), "s1_questions": s1.get(page["url"], [])}


def _note(kind: str, cat: dict[str, Any], pages: list[dict[str, Any]], field: str = "", value: str = "", values: tuple[str, str] = ("", "")) -> str:
    def clean(value_text: Any) -> str:
        text = str(value_text or "")
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"https?://\S+", "", text, flags=re.I)
        text = re.split(r"\s+(?=(?:Country|Geographical area|Data holding centre|Station|Network|Parameters?|Instruments?|Time period|Start time|Stop time|Depth|Height)\s*[|:])", text, maxsplit=1, flags=re.I)[0]
        return re.sub(r"\s+", " ", text).strip(" |:;,.\t\r\n")
    members = cat["members"]
    if kind == "set_all":
        value = clean(value)
        exact = [m["title"] for m in members if clean(cat.get("_fields", {}).get(m["url"], {}).get(field, "")).casefold() == value.casefold()]
        return (f'CATALOGUE NOTE: The knowledge base contains {len(members)} records of the catalogue "{cat["name"]}". '
                f"Exactly these {len(exact)} records have {field} = {value}: " + "; ".join(exact) + ". No other record of this catalogue has this value.")
    if kind == "unanswerable":
        v1, v2 = map(clean, values)
        field = clean(field)
        second_field = clean(cat.get("_second_field", ""))
        detail = []
        for m in members:
            f = cat.get("_fields", {}).get(m["url"], {})
            if field in f and v1.casefold() in clean(f[field]).casefold(): detail.append(f'{m["title"]} (with {field} = {clean(f[field])})')
            elif second_field in f and v2.casefold() in clean(f[second_field]).casefold(): detail.append(f'{m["title"]} (with {second_field} = {clean(f[second_field])})')
        return (f'CATALOGUE NOTE: The knowledge base contains {len(members)} records of the catalogue "{cat["name"]}". '
                f"No record has both {field} = {v1} and {second_field} = {v2}. The closest records are: " + "; ".join(detail[:4]) + ".")
    return ""


def _bundle(kind: str, task: str, pages: list[dict[str, Any]], s1: dict[str, list[str]], cat: dict[str, Any] | None = None,
            group_note: str = "", **extra: Any) -> dict[str, Any]:
    for i, page in enumerate(pages, 1): page["evidence_id"] = f"E{i}"
    seen = []
    for p in pages:
        seen.extend(s1.get(p["url"], []))
    note_lines = [f"- {q}" for q in dict.fromkeys(seen) if q]
    s1_note = "ALREADY ASKED: Another question set already asks the following about these pages. Do not ask these facts again:\n" + "\n".join(note_lines) if note_lines else ""
    return {"type": kind, "user_task": task, "pages": pages,
            "catalogue_id": cat.get("catalogue_id") if cat else None,
            "catalogue_name": cat.get("name") if cat else None,
            "catalogue_size": len(cat["members"]) if cat else 0,
            "group_note": group_note, "s1_note": s1_note,
            "difficulty": {"page_count": len(pages), "evidence_segment_count": len(pages),
                           "catalogue_size": len(cat["members"]) if cat else 0,
                           "has_duplicate_or_similar_records": any(p.get("duplicate_title") for p in pages)}, **extra}


def _screen_link(row: dict[str, Any], *, mock: bool, output_dir: Path) -> dict[str, Any]:
    if mock:
        return {"link_id": row["link_id"], "passed": True, "reason": "Mock-only validation; relation and target fact fields are present."}
    prompt = PROMPT.read_text(encoding="utf-8").split("## User template")[0].strip()
    user = (f"Entity: {row['entity']}\nA title: {row['source_title']}\nA excerpt: {row['source_quote']}\n"
            f"B title: {row['target_title']}\nB excerpt: {row['target_excerpt']}\n"
            "Can one plausible researcher question require a specific fact from A and a specific fact from B? Return JSON.")
    result = call_llm("judge", [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
        json_mode=True, temperature=0.0, prompt_sha256=__import__("hashlib").sha256(prompt.encode()).hexdigest(),
        cache_dir=output_dir / "cache/llm", log_dir=output_dir / "logs")
    item = result.get("json") or {}
    return {"link_id": row["link_id"], "passed": item.get("can_construct") is True or item.get("valid_link") is True,
            "reason": item.get("reason") or item.get("reasoning", ""), "judge_response": item, "model": result.get("model")}


def build_bundles(seed: int = 42, max_per_type: int = 9, *, mock: bool = True, output_dir: Path | None = None,
                  main3: bool = False) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    output_dir = output_dir or (ROOT / "outputs/s2/pilot2")
    pages = read_jsonl(INDEX_DIR / "indexed_pages.jsonl"); cats = read_jsonl(INDEX_DIR / "catalogues.jsonl")
    field_rows = read_jsonl(INDEX_DIR / "catalogue_fields.jsonl")
    links = read_jsonl(INDEX_DIR / "entity_links.jsonl"); conflicts = read_jsonl(INDEX_DIR / "conflict_candidates.jsonl")
    claim_pages = read_jsonl(INDEX_DIR / "page_claims.jsonl")
    claim_bridges = read_jsonl(INDEX_DIR / "claim_bridges.jsonl")
    chunks_df = pd.read_csv(CHUNKS_PATH, dtype=str, keep_default_na=False)
    frames = {url: f.sort_values("chunk_number", kind="stable") for url, f in chunks_df.groupby("url", sort=False)}
    page_by_url = {p["url"]: p for p in pages}; s1 = _s1_map(); rng = random.Random(seed)
    field_by_url = _page_field_map(field_rows)
    for cat in cats: cat["_fields"] = {m["url"]: page_by_url.get(m["url"], {}).get("fields", {}) for m in cat["members"]}
    candidates: dict[str, list[dict[str, Any]]] = {t: [] for t in TYPE_ORDER}
    for cat in cats:
        members = [page_by_url[m["url"]] for m in cat["members"] if m["url"] in page_by_url]
        data_fields = [f for f in cat["common_fields"] if RELEVANT_FIELD.search(f) and not EXCLUDED_FIELD.search(f)]
        data_fields.sort(key=lambda f: (0 if CLASSIFICATION_FIRST.search(f) else 1, f.casefold()))
        # set_all uses a data-selection attribute with exactly three to eight matches.
        for field in data_fields:
            if re.search(r"time|temporal|coverage|period", field, re.I):
                temporal_rows=[]
                for p in members:
                    value=str(p.get("fields",{}).get(field,"")).strip()
                    years=re.findall(r"\b(?:18|19|20|21)\d{2}\b",value)
                    if value and years and not any(year in p.get("title","") for year in years):
                        temporal_rows.append({"value":value,"page":p})
                for triple in itertools.combinations(temporal_rows,3):
                    if len({x["value"].casefold() for x in triple})<3:
                        continue
                    relation=_shared_comparison_relation([row["page"] for row in triple])
                    if not relation: continue
                    parsed_values = [_parse_comparison_value(field, row["value"]) for row in triple]
                    if not all(parsed_values): continue
                    titles = [row["page"].get("title", "") for row in triple]
                    if _comparison_value_in_any_title([row["value"] for row in triple], titles): continue
                    ev=[]
                    for item in triple:
                        w=_window(frames[item["page"]["url"]],field)
                        if not w: ev=[]; break
                        ev.append(_page_record(item["page"],w,s1))
                    if ev:
                        candidates["comparison_multi"].append(_bundle("comparison_multi","compare",ev,s1,cat,"",
                            target_field=field,record_values={item["page"]["title"]:item["value"] for item in triple},
                            related_context={relation[0]:relation[1]},candidate_source="catalogue_field"))
                        break
                continue
            groups: dict[str, list[dict[str, Any]]] = {}
            for p in members:
                v = p["fields"].get(field, "").strip()
                if v and len(v) < 140: groups.setdefault(v.casefold(), []).append(p)
            for key, matches in groups.items():
                if not 3 <= len(matches) <= 8: continue
                value = matches[0]["fields"][field]
                global_urls = {row["url"] for row in field_rows
                               if str(row.get("fields", {}).get(field, "")).casefold().strip() == value.casefold().strip()}
                if global_urls != {p["url"] for p in matches}:
                    continue
                evid = []
                for p in matches:
                    w = _field_window(frames[p["url"]], field, value)
                    if not w: evid = []; break
                    evid.append(_page_record(p, w, s1))
                if evid:
                    catnote = _note("set_all", cat, evid, field, value)
                    candidates["set_all"].append(_bundle("set_all", "find_data", evid, s1, cat, catnote,
                        target_field=field, target_value=value,
                        expected_items=[x["title"] for x in evid], field_match_urls=[p["url"] for p in matches]))
        # Temporal set conditions use interval overlap, never literal field equality.
        for field in [f for f in data_fields if re.search(r"time|temporal|coverage|period", f, re.I)]:
            intervals = {p["url"]: _record_interval(p, field) for p in members}
            if any(interval.get("start_year") is None or interval.get("end_year") is None for interval in intervals.values()):
                continue
            start_years = [v["start_year"] for v in intervals.values() if v.get("start_year") is not None]
            if not start_years:
                continue
            global_intervals = {row["url"]: (row, parse_time_interval(str(row.get("fields", {}).get(field, ""))))
                                for row in field_rows if row.get("fields", {}).get(field)}
            for year in range(max(1900, min(start_years)), dt.date.today().year + 1):
                matches = [p for p in members if _year_overlap(intervals[p["url"]], year)]
                if not 3 <= len(matches) <= 8:
                    continue
                global_urls = {url for url, (_, interval) in global_intervals.items() if _year_overlap(interval, year)}
                match_urls = {p["url"] for p in matches}
                if global_urls != match_urls:
                    continue
                evid = []
                for p in matches:
                    w = _window(frames[p["url"]], field)
                    if not w:
                        evid = []; break
                    evid.append(_page_record(p, w, s1))
                if evid:
                    note = (f'CATALOGUE NOTE: The records listed below belong to "{cat["name"]}". '
                            f'They are exactly the records whose temporal coverage overlaps calendar year {year}: '
                            + "; ".join(p["title"] for p in evid) + ".")
                    candidates["set_all"].append(_bundle("set_all", "find_data", evid, s1, cat, note,
                        target_field=field, target_value=f"covers calendar year {year}", time_condition={"start_year":year,"end_year":year},
                        expected_items=[x["title"] for x in evid], field_match_urls=[p["url"] for p in matches],
                        candidate_source="catalogue_field"))
        # Comparison values are exact and mutually different.
        related_groups: dict[tuple[str, str], set[str]] = set()
        for relation_field in RELATED_FIELDS:
            values = Counter(str(p.get("fields", {}).get(relation_field, "")).strip().casefold() for p in members)
            for value, count in values.items():
                if value and count >= 3:
                    related_groups.add((relation_field, value))
        for field in data_fields:
            if not COMPARABLE_FIELD.search(field):
                continue
            vals: dict[str, dict[str, Any]] = {}
            for p in members:
                value = p["fields"].get(field, "").strip()
                parsed_value = _parse_comparison_value(field, value)
                if not parsed_value or value.casefold() in p.get("title", "").casefold():
                    continue
                if value and len(value) < 140: vals.setdefault(value.casefold(), {"value": value, "parsed_value": parsed_value, "page": p})
            if len(vals) < 3: continue
            chosen = None
            shared_relation = None
            for triple in itertools.combinations(vals.values(), 3):
                relation = _shared_comparison_relation([row["page"] for row in triple])
                if relation:
                    chosen=list(triple); shared_relation=relation; break
            if not chosen:
                continue
            ev = []
            for item in chosen:
                p, v = item["page"], item["value"]
                w = _field_window(frames[p["url"]], field, v)
                if not w: ev = []; break
                ev.append(_page_record(p, w, s1))
            if ev:
                candidates["comparison_multi"].append(_bundle("comparison_multi", "compare", ev, s1, cat,
                    record_values={p["title"]: chosen[i]["value"] for i,p in enumerate(ev)}, target_field=field,
                    related_context={shared_relation[0]:shared_relation[1]},
                    candidate_source="catalogue_field"))
        # Unanswerable field combinations: each value occurs, never on the same page.
        for fi, f1 in enumerate(data_fields):
            for f2 in data_fields[fi+1:]:
                if re.search(r"time|temporal|coverage|period", f1 + " " + f2, re.I):
                    continue
                vals1 = list(dict.fromkeys(p["fields"].get(f1, "") for p in members if p["fields"].get(f1)))
                vals2 = list(dict.fromkeys(p["fields"].get(f2, "") for p in members if p["fields"].get(f2)))
                if len(vals1) < 2 or len(vals2) < 2: continue
                for v1 in vals1[:3]:
                    for v2 in vals2[:3]:
                        if any(p["fields"].get(f1, "").casefold() == v1.casefold() and p["fields"].get(f2, "").casefold() == v2.casefold() for p in members): continue
                        left = next((p for p in members if p["fields"].get(f1, "").casefold() == v1.casefold()), None)
                        right = next((p for p in members if p["fields"].get(f2, "").casefold() == v2.casefold()), None)
                        if not left or not right: continue
                        ev = []
                        for p, needle in ((left, v1), (right, v2)):
                            w = _field_window(frames[p["url"]], f1 if p is left else f2, needle)
                            if not w: ev = []; break
                            ev.append(_page_record(p, w, s1))
                        if ev:
                            cat["_second_field"] = f2
                            note = _note("unanswerable", cat, ev, f1, values=(v1,v2))
                            candidates["unanswerable"].append(_bundle("unanswerable", "find_data", ev, s1, cat, note,
                                target_field=f1, second_field=f2, target_values=[v1,v2], candidate_source="catalogue_field"))
        # Temporal absence uses interval overlap and checks the full indexed KB before candidacy.
        temporal_fields = [field for field in data_fields if re.search(r"time|temporal|coverage|period", field, re.I)]
        category_fields = [field for field in data_fields if CLASSIFICATION_FIRST.search(field) and field not in temporal_fields]
        for temporal_field in temporal_fields:
            starts = [_record_interval(p, temporal_field).get("start_year") for p in members]
            starts = [year for year in starts if year is not None]
            if not starts:
                continue
            global_rows = [row for row in field_rows if row.get("fields", {}).get(temporal_field)]
            for category_field in category_fields:
                category_values = list(dict.fromkeys(p["fields"].get(category_field, "").strip()
                    for p in members if p["fields"].get(category_field, "").strip()))
                for category_value in category_values:
                    holding_rows = [p for p in members if p["fields"].get(category_field, "").strip().casefold() == category_value.casefold()]
                    if not holding_rows:
                        continue
                    for year in range(max(1900, min(starts)), dt.date.today().year + 1):
                        cat_matches = [p for p in holding_rows if _year_overlap(_record_interval(p, temporal_field), year)]
                        global_matches = [row for row in global_rows
                            if str(row.get("fields", {}).get(category_field, "")).strip().casefold() == category_value.casefold()
                            and _year_overlap(parse_time_interval(str(row.get("fields", {}).get(temporal_field, ""))), year)]
                        year_exists = any(_year_overlap(_record_interval(p, temporal_field), year) for p in members)
                        if cat_matches or global_matches or not year_exists:
                            continue
                        left = holding_rows[0]
                        right = next((p for p in members if p["url"] != left["url"] and _year_overlap(_record_interval(p, temporal_field), year)), None)
                        if not right:
                            continue
                        ev=[]
                        for p in (left,right):
                            w=_window(frames[p["url"]], temporal_field)
                            if not w: ev=[]; break
                            ev.append(_page_record(p,w,s1))
                        if ev:
                            note=(f'CATALOGUE NOTE: In "{cat["name"]}", no record held by {category_value} has temporal coverage overlapping calendar year {year}. '
                                  f'The closest records are {left["title"]} ({category_field}: {category_value}; {temporal_field}: {left["fields"].get(temporal_field, "unspecified")}) and '
                                  f'{right["title"]} ({temporal_field}: {right["fields"].get(temporal_field, "unspecified")}).')
                            candidates["unanswerable"].append(_bundle("unanswerable","find_data",ev,s1,cat,note,
                                target_field=category_field,second_field=temporal_field,target_values=[category_value,year],
                                time_condition={"start_year":year,"end_year":year,"category_field":category_field,"category_value":category_value},
                                candidate_source="catalogue_field"))
        # Same record name, different data-relevant field values.
        byname: dict[str,list[dict[str,Any]]] = {}
        for p in members:
            record_name = str(p.get("record_name", "")).strip()
            if p.get("name_truncated") or not record_name: continue
            byname.setdefault(record_name.casefold(), []).append(p)
        for name, same in byname.items():
            for i, a in enumerate(same):
                for b in same[i+1:]:
                    diffs = [(f,a["fields"][f],b["fields"][f]) for f in set(a["fields"]) & set(b["fields"])
                             if RELEVANT_FIELD.search(f) and not EXCLUDED_FIELD.search(f) and a["fields"][f].casefold()!=b["fields"][f].casefold()]
                    if not diffs: continue
                    field, va, vb = diffs[0]; ev=[]
                    for p,val in ((a,va),(b,vb)):
                        w=_window(frames[p["url"]],val) or _window(frames[p["url"]],field)
                        if not w: ev=[]; break
                        ev.append(_page_record(p,w,s1))
                    if ev:
                        note=f'AMBIGUITY NOTE: The knowledge base contains {len(same)} different records that share the name "{a["record_name"]}": {a["record_name"]} ({field}: {va}); {b["record_name"]} ({field}: {vb}).'
                        candidates["ambiguity"].append(_bundle("ambiguity","dataset_details",ev,s1,cat,note,
                            ambiguity_kind="same_name", shared_name=a["record_name"], target_field=field,
                            record_values=[{"title":a["title"],"value":va,"source":"E1"},{"title":b["title"],"value":vb,"source":"E2"}],
                            candidate_source="same_name_records"))
    # Claim bridges provide the preferred cross-page factual packages.
    claim_by_url = {row["url"]: row.get("claims", []) for row in claim_pages}
    used_claim_sources: set[tuple[str, str, str, str]] = set()
    for bridge in claim_bridges:
        fact_a, fact_b = bridge.get("fact_a", {}), bridge.get("fact_b", {})
        source_page, target_page = page_by_url.get(fact_a.get("url")), page_by_url.get(fact_b.get("url"))
        if not source_page or not target_page or source_page["url"] == target_page["url"]:
            continue
        key = (source_page["url"], str(fact_a.get("chunk_id")), target_page["url"], str(fact_b.get("chunk_id")))
        if key in used_claim_sources:
            continue
        used_claim_sources.add(key)
        windows = [_claim_window(frames[source_page["url"]], str(fact_a.get("quote", ""))),
                   _claim_window(frames[target_page["url"]], str(fact_b.get("quote", "")))]
        if not all(windows):
            continue
        ev = [_page_record(source_page, windows[0], s1), _page_record(target_page, windows[1], s1)]
        candidates["multi_hop"].append(_bundle("multi_hop", "background", ev, s1, None, "",
            entity=bridge.get("entity", ""), source_quote=fact_a.get("quote", ""), fact_a=fact_a, fact_b=fact_b,
            bridge_id=bridge.get("bridge_id"), candidate_source="cross_page_fact",
            catalogue_id=f"FACT:{source_page.get('url_prefix', '')}", catalogue_name=f"Claim bridge at {source_page.get('url_prefix', '')}"))
    # Claim-based comparison sets: same data-selection property, distinct explicit values,
    # and a shared place, organisation, parameter family, network, or programme.
    claim_rows = []
    for page_row in claim_pages:
        page = page_by_url.get(page_row["url"])
        if not page:
            continue
        for claim in page_row.get("claims", []):
            prop, value = str(claim.get("property", "")).strip(), str(claim.get("value", "")).strip()
            if not prop or not value or not RELEVANT_FIELD.search(prop) or EXCLUDED_FIELD.search(prop) or not _parse_comparison_value(prop, value):
                continue
            if value.casefold() in str(page.get("title", "")).casefold():
                continue
            claim_rows.append((page, claim, prop, value))
    by_property: dict[str, list[tuple[dict[str, Any], dict[str, Any], str, str]]] = defaultdict(list)
    for row in claim_rows:
        by_property[row[2].casefold()].append(row)
    for property_name, rows in by_property.items():
        distinct_pages: dict[str, tuple[dict[str, Any], dict[str, Any], str, str]] = {}
        for row in rows:
            distinct_pages.setdefault(row[0]["url"], row)
        rows = list(distinct_pages.values())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                for k in range(j + 1, len(rows)):
                    chosen = [rows[i], rows[j], rows[k]]
                    values = [x[3].casefold() for x in chosen]
                    if len(set(values)) != 3:
                        continue
                    titles = [str(row[0].get("title", "")) for row in chosen]
                    if _comparison_value_in_any_title(values, titles):
                        continue
                    relation = next(((field, str(chosen[0][0].get("fields", {}).get(field, "")).strip())
                        for field in RELATED_FIELDS
                        if chosen[0][0].get("fields", {}).get(field)
                        and all(str(row[0].get("fields", {}).get(field, "")).strip().casefold() == str(chosen[0][0]["fields"][field]).strip().casefold() for row in chosen)), None)
                    if not relation:
                        continue
                    ev=[]
                    for page, claim, prop, value in chosen:
                        window=_claim_window(frames[page["url"]],str(claim.get("quote", "")))
                        if not window: ev=[]; break
                        ev.append(_page_record(page,window,s1))
                    if ev:
                        candidates["comparison_multi"].append(_bundle("comparison_multi","compare",ev,s1,None,"",
                            target_field=chosen[0][2],record_values={chosen[m][0]["title"]:chosen[m][3] for m in range(3)},
                            related_context={relation[0]:relation[1]},claim_facts=[x[1] for x in chosen],candidate_source="cross_page_fact",
                            catalogue_id=f"FACT:{relation[0]}:{relation[1]}",catalogue_name=f"Claim comparison: {relation[0]} {relation[1]}"))
                        break
    # Claim sets are admitted only when repeated facts have a checkable, exact subject/property/value.
    fact_sets: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for page_row in claim_pages:
        page = page_by_url.get(page_row["url"])
        if not page: continue
        for claim in page_row.get("claims", []):
            subject=str(claim.get("subject", "")).strip(); prop=str(claim.get("property", "")).strip(); value=str(claim.get("value", "")).strip()
            if subject and prop and value and RELEVANT_FIELD.search(prop) and not EXCLUDED_FIELD.search(prop):
                fact_sets[(subject.casefold(),prop.casefold(),value.casefold())].append((page,claim))
    for (subject, prop, value), rows in fact_sets.items():
        unique={row[0]["url"]:row for row in rows}
        if not 3 <= len(unique) <= 8: continue
        selected=list(unique.values())
        ev=[]
        for page,claim in selected:
            window=_claim_window(frames[page["url"]],str(claim.get("quote", "")))
            if not window: ev=[]; break
            ev.append(_page_record(page,window,s1))
        if ev:
            note=f"CATALOGUE NOTE: These records are independently reported to share {selected[0][1].get('subject', subject)} / {selected[0][1].get('property', prop)} / {selected[0][1].get('value', value)}. The complete list must be checked against retrieval across the full knowledge base."
            candidates["set_all"].append(_bundle("set_all","find_data",ev,s1,None,note,
                expected_items=[p["title"] for p in ev],field_match_urls=[p["url"] for p in ev],
                target_field=selected[0][1].get("property",prop),target_value=selected[0][1].get("value",value),
                claim_facts=[x[1] for x in selected],candidate_source="cross_page_fact",
                catalogue_id=f"FACTSET:{subject}:{prop}:{value}",catalogue_name=f"Claim set: {subject}"))
    # Strict entity links are screened by judge before they become bundles.
    for conflict in conflicts:
        if conflict.get("kind") != "ambiguity" or not conflict.get("object"):
            continue
        left=page_by_url.get(conflict.get("url_a")); right=page_by_url.get(conflict.get("url_b"))
        if not left or not right: continue
        field=conflict.get("attribute",""); va=conflict.get("value_a",""); vb=conflict.get("value_b","")
        ev=[]
        for p,val in ((left,va),(right,vb)):
            w=_field_window(frames[p["url"]],field,val)
            if not w: ev=[]; break
            ev.append(_page_record(p,w,s1))
        if ev:
            note=(f'AMBIGUITY NOTE: The knowledge base contains records that share the name "{conflict["object"]}": '
                  f'{left["title"]} ({field}: {va}); {right["title"]} ({field}: {vb}).')
            candidates["ambiguity"].append(_bundle("ambiguity","dataset_details",ev,s1,None,note,
                ambiguity_kind="same_name",shared_name=conflict["object"],target_field=field,
                record_values=[{"title":left["title"],"value":va,"source":"E1"},{"title":right["title"],"value":vb,"source":"E2"}], conflict_candidate=conflict,
                catalogue_id=f'NAME:{left.get("url_prefix", "")}', candidate_source="same_name_records"))

    link_results=[]
    for link in links:
        source=page_by_url.get(link["source_url"]); target=page_by_url.get(link["target_url"])
        if not source or not target: continue
        tw="".join(frames[target["url"]]["content"].tolist())
        if not re.search(r"\b(?:19|20)\d{2}\b|\b\d+\b|instrument|located|based in|member",tw,re.I): continue
        target_title=(target.get("record_name") or target["title"]).casefold()
        entity=link["entity"].casefold()
        if entity not in target_title: continue
        if not re.search(rf"(?i)\b(?:produced by|operated by|hosted by|part of|coordinated by|provided by|maintained by|managed by|run by)\b[^.\n]{{0,180}}{re.escape(entity)}",link["source_quote"]): continue
        target_excerpt = _target_window(frames[target["url"]], link["entity"])
        link["target_excerpt"] = target_excerpt["text"] if target_excerpt else ""
        if isinstance(link["target_excerpt"],dict): link["target_excerpt"]=link["target_excerpt"]["text"]
        link_results.append(_screen_link(link,mock=mock,output_dir=output_dir))
    screened={r["link_id"] for r in link_results if r["passed"]}
    result_by_id={r["link_id"]:r for r in link_results}
    for link in links:
        if link.get("link_id") not in screened: continue
        source=page_by_url[link["source_url"]]; target=page_by_url[link["target_url"]]
        sw=_window(frames[source["url"]],link["source_quote"]); target_windows=_target_windows(frames[target["url"]],link["entity"],limit=2)
        if not sw or not target_windows: continue
        for variant_no,tw in enumerate(target_windows,1):
            ev=[_page_record(source,sw,s1),_page_record(target,tw,s1)]
            candidates["multi_hop"].append(_bundle("multi_hop","background",ev,s1,None,"",
                entity=link["entity"],source_quote=link["source_quote"],link_screen=result_by_id[link["link_id"]],
                target_fact_variant=variant_no,
                catalogue_id=f'LINK:{source.get("url_prefix", "")}', catalogue_name=f'Explicit links at {source.get("url_prefix", "")}'))
    # Verified S1 conflicts only; same-title ambiguity is represented separately above.
    for conflict in conflicts:
        if conflict.get("kind")!="conflict" or not conflict.get("url_a") or not conflict.get("url_b"): continue
        left=page_by_url.get(conflict["url_a"]); right=page_by_url.get(conflict["url_b"])
        if not left or not right: continue
        attribute = str(conflict.get("attribute", ""))
        if (not left.get("record_name") or left.get("record_name", "").casefold() != right.get("record_name", "").casefold()
                or attribute not in left.get("fields", {}) or attribute not in right.get("fields", {})):
            continue
        value_a = left["fields"][attribute]; value_b = right["fields"][attribute]
        if value_a.casefold() == value_b.casefold(): continue
        ev=[]
        for p,val in ((left,value_a),(right,value_b)):
            w=_field_window(frames[p["url"]], attribute, val)
            if not w: ev=[]; break
            ev.append(_page_record(p,w,s1))
        if ev:
            note=(f'CONFLICT NOTE: These pages describe the same {left["record_name"]} and give different values for '
                  f'{attribute}: {value_a} ({left["title"]}), {value_b} ({right["title"]}).')
            candidates["ambiguity"].append(_bundle("ambiguity","dataset_details",ev,s1,None,note,
                ambiguity_kind="confirmed_conflict", conflict_candidate={**conflict,"object":left["record_name"],"attribute":attribute,
                    "value_a":value_a,"value_b":value_b,"quote_a":left.get("field_quotes",{}).get(attribute,""),
                    "quote_b":right.get("field_quotes",{}).get(attribute,""),"url_a":left["url"],"url_b":right["url"]},
                candidate_source="s1_confirmed_conflict"))
    # De-duplicate under the formal per-type, catalogue/holder-site, and condition-field limits.
    counts={}; selected=[]; raw_counts={k:len(v) for k,v in candidates.items()}
    raw_by_source={kind:dict(Counter(row.get("candidate_source","catalogue_field") for row in rows)) for kind,rows in candidates.items()}
    target_limits = {"ambiguity": 6, "unanswerable": 12, "set_all": 12, "comparison_multi": 9, "multi_hop": 4}
    for kind in TYPE_ORDER:
        if main3 and kind in {"ambiguity", "multi_hop"}:
            continue
        rng.shuffle(candidates[kind]); seen=set(); cat_holder_counts=Counter(); field_counts=Counter(); cat_field_counts=Counter(); source_seen=set()
        ordered=sorted(candidates[kind],key=lambda row:(0 if row.get("candidate_source")=="cross_page_fact" else 1))
        for row in ordered:
            urls=tuple((p["url"],tuple(p.get("chunk_ids",[]))) for p in row["pages"]); key=(urls,row.get("target_field"),row.get("target_value"),row.get("ambiguity_kind"),row.get("entity"),row.get("target_fact_variant"))
            if key in seen: continue
            seen.add(key)
            source_key=(kind,row.get("candidate_source","catalogue_field"),row.get("bridge_id"),row.get("catalogue_id"),row.get("target_field"),str(row.get("target_value")),urls)
            if source_key in source_seen: continue
            catkey=row.get("catalogue_id") or row.get("pages", [{}])[0].get("url_prefix") or "no_catalogue"
            if main3 and "icos object catalogue" in str(row.get("catalogue_name", "")).casefold():
                continue
            first_fields = row.get("pages", [{}])[0].get("fields", {}) if row.get("pages") else {}
            holder = next((str(first_fields.get(name, "")).strip() for name in first_fields
                           if re.search(r"data holding centre|holding institution|data provider|station|site name", name, re.I)
                           and first_fields.get(name)), "")
            if not holder and row.get("pages"):
                holder = urlsplit(row["pages"][0].get("url", "")).netloc.casefold()
            diversity_key = (catkey, holder.casefold())
            fieldkey = str(row.get("target_field", "")).casefold().strip()
            if kind == "unanswerable" and row.get("second_field"):
                fieldkey = " + ".join(sorted((fieldkey, str(row["second_field"]).casefold().strip())))
            if main3 and kind == "set_all":
                if not CLASSIFICATION_FIRST.search(fieldkey) or row.get("time_condition"):
                    continue
                triple_key = (str(catkey), fieldkey, str(row.get("target_value", "")).casefold().strip())
                if cat_field_counts[triple_key] >= 1 or field_counts[fieldkey] >= 3:
                    continue
            if main3 and kind == "unanswerable":
                page_rows = [page_by_url.get(p.get("url"), {}) for p in row.get("pages", [])]
                has_shared_context = False
                for rel_field in RELATED_FIELDS:
                    vals = [str(page.get("fields", {}).get(rel_field, "")).strip().casefold() for page in page_rows]
                    if len(vals) >= 2 and all(vals) and (all(v == vals[0] for v in vals[1:]) or
                            ("geographical area" in rel_field and all(a in b or b in a for a, b in itertools.combinations(vals, 2)))):
                        has_shared_context = True
                        break
                if not has_shared_context:
                    continue
            if cat_holder_counts[diversity_key] >= 3: continue
            if fieldkey and field_counts[fieldkey] >= 2 and kind != "set_all": continue
            if kind == "set_all" and fieldkey and cat_field_counts[(catkey, fieldkey)] >= 1: continue
            main3_limits = {"set_all": 8, "unanswerable": 10, "comparison_multi": 99}
            limit = min(max_per_type, main3_limits.get(kind, target_limits.get(kind, max_per_type)))
            if counts.get(kind,0)>=limit: break
            row["bundle_id"]=f"S2-{kind.upper()}-{counts.get(kind,0)+1:03d}"
            selected.append(row); counts[kind]=counts.get(kind,0)+1; cat_holder_counts[diversity_key]+=1
            if fieldkey: field_counts[fieldkey]+=1
            if kind == "set_all" and fieldkey:
                if main3:
                    cat_field_counts[(str(catkey), fieldkey, str(row.get("target_value", "")).casefold().strip())] += 1
                else:
                    cat_field_counts[(catkey, fieldkey)] += 1
            source_seen.add(source_key)
    write_jsonl(INDEX_DIR/"entity_link_screening.jsonl",link_results)
    return selected, raw_counts, {"raw_by_source":raw_by_source,"selected_by_type":dict(Counter(r["type"] for r in selected)),
                                 "selected_by_source":{kind:dict(Counter(r.get("candidate_source","catalogue_field") for r in selected if r["type"]==kind)) for kind in TYPE_ORDER},
                                 "indexed_before_static_screen":len(links),
                                 "eligible_for_judge_screen":len(link_results),
                                 "judge_passed":len(screened)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--max-per-type",type=int,default=9)
    parser.add_argument("--out",type=Path,default=INDEX_DIR/"bundles_s2.jsonl"); parser.add_argument("--mock",action="store_true"); parser.add_argument("--main3",action="store_true")
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/s2/pilot2"); args=parser.parse_args()
    if not args.mock: load_dotenv(ROOT/".env",override=True)
    rows,raw,links=build_bundles(args.seed,args.max_per_type,mock=args.mock,output_dir=args.output_root,main3=args.main3)
    write_jsonl(args.out,rows)
    summary={"selected_bundles":len(rows),"by_type":dict(Counter(r["type"] for r in rows)),
             "raw_candidates":raw,"entity_links":links,"output":str(args.out),"mock":args.mock}
    summary_path=args.out.with_name("bundle_summary_s2.json")
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
