"""Build the revised S2 catalogue, entity-link, and conflict indexes."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sample_windows import page_windows

PAGES = ROOT / "data/processed/pages.csv"
CHUNKS = ROOT / "data/processed/chunks_clean.csv"
S1_QUESTIONS = ROOT / "outputs/final/questions_final.jsonl"
OUT = ROOT / "data/processed/s2"
FIELD_LABELS = {"country", "geographical area", "data holding centre", "station", "time period",
                "start time", "stop time", "variables", "parameters", "instruments", "ecosystem type",
                "data set name", "data affiliation", "part of", "station name", "latitude", "longitude",
                "sampling height", "depth", "temporal coverage", "spatial coverage", "observed variables",
                "research infrastructure", "network", "project", "centre", "instrument", "ecosystem", "data level",
                "documentation", "licence", "license", "file made by", "production time", "format", "encoding",
                "submitted by", "publication time", "publication date", "temporal coverages", "data affiliation",
                "downloads", "keywords", "description", "site reference", "organization", "uri",
                "definition", "preferred label", "broader concept", "narrower concept", "network name",
                "observing network", "programme", "data centre", "discipline", "temporal coverages",
                "temporal coverage", "start year", "end year", "coordinate system", "spatial resolution"}
FIELD_LABELS.update({"pref label", "preflabel", "alt label", "altlabel", "notation", "status", "concept status", "skos definition",
                     "within vocab", "alternative labels", "date", "identifier", "note", "has current version", "related",
                     "creator", "modified", "version info", "register manager", "register owner", "description"})
BAD_SUFFIX = {"from", "of", "in", "at", "to", "by", "for", "with", "on", "near", "off", "and"}
BAD_FIELDS = {"id", "identifier", "uri", "url", "license", "licence", "file", "filename", "file name",
              "file size", "version", "pid", "doi", "date", "title", "name"}
RELATIONS = r"produced by|operated by|hosted by|part of|coordinated by|provided by|maintained by|managed by|run by"
GENERIC_ENTITIES = {"research infrastructure", "data", "project", "network", "infrastructure", "service"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _window(frame: pd.DataFrame) -> dict[str, Any] | None:
    candidates = page_windows(frame)
    return candidates[0] if candidates else None


def _normalize_label(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = re.sub(r"\s+", " ", re.sub(r"[*_`#]", "", value)).strip(" :|-–—").casefold()
    value = re.sub(r"\s*\([^)]*\)$", "", value)
    return value


def _field_is_data_relevant(field: str) -> bool:
    return bool(re.search(r"country|geographical|area|holding centre|station|variable|parameter|instrument|ecosystem|time period|start time|stop time|temporal coverages?|depth|height|coverage|latitude|longitude|region|site reference", field, re.I))


def _norm_value(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"\s+", " ", value).strip(" .;,").casefold()


def _field_quote(text: str, field: str, value: str) -> str:
    field_re = re.escape(field).replace(r"\ ", r"\s+")
    value_re = re.escape(value).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?is){field_re}(?:\s*\([^)]*\))?\s*(?:\||:)?\s*{value_re}")
    match = pattern.search(text)
    return match.group(0).strip() if match else value


def extract_fields(text: str) -> dict[str, str]:
    """Extract labelled values from common plain-text and Markdown layouts."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    result: dict[str, str] = {}
    for i, raw in enumerate(lines):
        line = raw.strip(" |\t")
        if not line:
            continue
        markdown = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|?$", raw)
        bold = re.match(r"^\*\*([^*]+?):\*\*\s*(.*)$", raw)
        if markdown:
            left, right = markdown.group(1), markdown.group(2)
        elif bold:
            left, right = bold.group(1), bold.group(2)
        elif "|" in raw:
            left, right = raw.split("|", 1)
            label = _normalize_label(left)
            value = re.sub(r"\s+", " ", right).strip(" |\t")
            if not value and i + 1 < len(lines):
                value = lines[i + 1].strip(" |\t")
        else:
            label = _normalize_label(line)
            if label not in FIELD_LABELS:
                label = next((candidate for candidate in sorted(FIELD_LABELS, key=len, reverse=True)
                              if label.endswith(candidate)), label)
            value = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip(" |\t")
                if nxt and "|" not in nxt and len(nxt) < 220:
                    value = nxt
            if label.startswith("**"):
                label = _normalize_label(label)
        if markdown or bold:
            label = _normalize_label(left)
            value = re.sub(r"\s+", " ", right).strip(" |\t")
        label = _normalize_label(label)
        if (label in FIELD_LABELS or label.startswith(("variable", "parameter", "instrument", "ecosystem type", "start time", "stop time", "sampling height"))) and value:
            value = re.sub(r"\s+", " ", value).strip(" |")
            if value and value not in {"-", "—", "N/A"}:
                result.setdefault(label, value)
    return result


def parse_time_interval(value: str) -> dict[str, int | None | bool]:
    """Parse broad temporal coverage into inclusive years and an open-end flag."""
    text = re.sub(r"\s+", " ", str(value)).strip().casefold()
    years = [int(x) for x in re.findall(r"\b(?:18|19|20|21)\d{2}\b", text)]
    start = end = None
    if years:
        start = years[0]
        end = years[-1] if len(years) > 1 else years[0]
        # A two-digit range endpoint inherits the century of its first year.
        short_range = re.search(r"\b((?:18|19|20|21)\d{2})\s*[-–]\s*(\d{2})\b", text)
        if short_range:
            century = int(short_range.group(1)) // 100 * 100
            end = century + int(short_range.group(2))
            if end < start:
                end += 100
    open_end = bool(re.search(r"ongoing|onwards|since|present|current|to date|\bto\s*$|[-–]\s*$", text))
    if open_end:
        end = None
    return {"start_year": start, "end_year": end, "open_end": open_end}


def record_name(text: str) -> tuple[str, bool, str]:
    h1 = next((re.sub(r"^#\s+", "", x).strip(" #") for x in text.splitlines() if re.match(r"^#\s+", x)), "")
    fields = extract_fields(text)
    named = (fields.get("data set name") or fields.get("preferred label") or fields.get("pref label") or "").lstrip(" #")
    h2 = next((re.sub(r"^##\s+", "", x).strip(" #") for x in text.splitlines() if re.match(r"^##\s+", x)), "")
    heading = h2 if h1.casefold() in {"concept", "vocabulary", "record", "dataset"} and h2 else h1
    value = named if len(named.split()) >= 3 else heading
    words = re.findall(r"[\w'-]+", value)
    truncated = bool(value and (not words or len(words) < 3 or words[-1].casefold() in BAD_SUFFIX))
    source = "data_set_name" if value == fields.get("data set name", "").lstrip(" #") and named else ("preferred_label" if value == named and named else ("h2" if value == h2 else "h1"))
    return value, truncated, source


def _url_prefix(url: str) -> str:
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    host = p.netloc.casefold()
    if "vocab.nerc.ac.uk" in host and len(parts) >= 3 and parts[0] == "collection":
        return f"{host}/{'/'.join(parts[:3])}"
    elif "dar.elter-ri.eu" in host and parts and parts[0] == "external-datasets":
        return f"{host}/external-datasets"
    elif "goosocean.org" in host and "global-ocean-observing-networks" in parts:
        return f"{host}/{'/'.join(parts[:parts.index('global-ocean-observing-networks') + 1])}"
    # ICOS object hashes and EDMED report numbers are variable terminal segments.
    if parts and (re.fullmatch(r"[A-Za-z0-9_-]{12,}", parts[-1]) or parts[-1].isdigit()):
        parts = parts[:-1]
    return f"{p.netloc.casefold()}/{'/'.join(parts[:2])}"


def _series_template(name: str) -> str:
    value = name.casefold().strip()
    # Preserve numeric tokens in proper names such as San Rossore 2.
    value = re.sub(r"\s+(?:from|at|near|in|off)\s+[^,]+$", " {site}", value)
    value = re.sub(r"\s+\([^)]*\)$", " {variant}", value)
    return re.sub(r"\s+", " ", value)


def _make_catalogues(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        if page["record_name"]:
            templates[(page["url_prefix"], _series_template(page["record_name"]))].append(page)
        buckets[page["url_prefix"]].append(page)
    rows = []
    assigned: set[str] = set()
    for (prefix, template), members in sorted(templates.items()):
        if len(members) < 2:
            continue
        common = set(members[0]["fields"])
        for member in members[1:]: common.intersection_update(member["fields"])
        if len(common) < 3:
            continue
        rows.append(_catalogue_row(len(rows) + 1, prefix, template, members, common))
        assigned.update(p["url"] for p in members)
    for prefix, bucket in sorted(buckets.items()):
        unassigned = sorted((p for p in bucket if p["url"] not in assigned), key=lambda p: ((p.get("record_name") or p.get("title", "")).casefold(), p["url"]))
        while unassigned:
            seed = unassigned.pop(0)
            members = [seed]
            common = set(seed["fields"])
            changed = True
            while changed:
                changed = False
                for candidate in list(unassigned):
                    candidate_common = common & set(candidate["fields"])
                    if len(candidate_common) >= 3:
                        members.append(candidate)
                        unassigned.remove(candidate)
                        common = candidate_common
                        changed = True
            if len(members) < 2 or len(common) < 3:
                continue
            names = [_series_template(p["record_name"]) for p in members if p.get("record_name")]
            template = Counter(names).most_common(1)[0][0] if names else "url-prefix catalogue"
            rows.append(_catalogue_row(len(rows) + 1, prefix, template, members, common))
    return rows


def _catalogue_row(number: int, prefix: str, template: str, members: list[dict[str, Any]], common: set[str]) -> dict[str, Any]:
    host, _, path = prefix.partition("/")
    if "edmed.seadatanet.org" in host:
        catalogue_name = "EDMED report catalogue"
    elif "meta.icos-cp.eu" in host:
        catalogue_name = "ICOS object catalogue"
    else:
        catalogue_name = f"{host} {path.replace('/', ' ').replace('-', ' ')}".strip()
    distributions = {}
    coverage: dict[str, dict[str, int | float]] = {}
    all_labels = sorted({field for member in members for field in member["fields"]})
    for field in all_labels:
        present = sum(field in member["fields"] and bool(str(member["fields"][field]).strip()) for member in members)
        coverage[field] = {"present": present, "member_count": len(members),
                           "coverage_pct": round(100 * present / max(1, len(members)), 1)}
    for field in sorted(common):
        counts: dict[str, list[str]] = defaultdict(list)
        for member in members: counts[member["fields"][field]].append(member["url"])
        distributions[field] = [{"value": value, "count": len(urls), "urls": sorted(urls)}
                                for value, urls in sorted(counts.items(), key=lambda item: item[0].casefold())]
    return {"catalogue_id": f"CAT-{number:04d}", "name": catalogue_name, "title_template": template,
            "url_prefix": prefix,
            "members": [{"url": p["url"], "title": p["record_name"], "page_type": p["page_type"]} for p in members],
            "common_fields": sorted(common), "field_coverage": coverage, "value_distributions": distributions}


def _template_only(page_text: str, repeated_lines: set[str], name: str, fields: dict[str, str]) -> bool:
    cleaned = []
    for line in page_text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        lowered = line.casefold()
        if not line or line.startswith(("![](", "![", "http://", "https://")):
            continue
        if any(term in lowered for term in ("cookie settings", "privacy policy", "terms of use", "skip to content", "related links")):
            continue
        cleaned.append(line)
    meaningful = "\n".join(cleaned)
    if len(re.sub(r"\s+", "", meaningful)) < 160:
        return True
    long_lines = [line for line in cleaned if len(line) >= 28]
    repeated_ratio = sum(line in repeated_lines for line in long_lines) / max(1, len(long_lines))
    has_record_content = bool(name and len(name.split()) >= 3) or bool(fields) or bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\s*(?:m|km|mm|ha|ppm|ppb)\b", meaningful, re.I))
    return repeated_ratio >= 0.8 and not has_record_content


def _entity_links(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedicated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        heading = page["record_name"] or page["h1"]
        if not heading or page["name_truncated"]:
            continue
        aliases = {heading.casefold()}
        acr = "".join(word[0] for word in re.findall(r"[A-Za-z][A-Za-z0-9-]*", heading) if word)
        if len(acr) >= 3:
            aliases.add(acr.casefold())
        for alias in aliases:
            if alias not in GENERIC_ENTITIES and (any(c.isupper() for c in heading) or alias.isupper() or len(heading.split()) >= 2):
                dedicated[alias].append(page)
    links, seen = [], set()
    entity_aliases = sorted((alias for alias in dedicated if len(alias) >= 3 and alias not in GENERIC_ENTITIES), key=len, reverse=True)
    relation_re = re.compile(rf"(?is)([^.\n]{{0,180}}\b(?:{RELATIONS})\s+[^.\n]{{0,220}})")
    for source in pages:
        clauses = [m.group(1).strip() for m in relation_re.finditer(source["text"])]
        if not clauses:
            continue
        for clause in clauses:
            folded = clause.casefold()
            alias = next((candidate for candidate in entity_aliases if re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", folded)), None)
            if not alias:
                continue
            for target in dedicated[alias]:
                if source["url"] == target["url"]:
                    continue
                if not re.search(r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?\s*(?:km|m|mm|ha|members|stations)\b|\b(?:located|based) in\b|instrument", target["text"], re.I):
                    continue
                key = (source["url"], target["url"], alias)
                if key in seen:
                    continue
                seen.add(key)
                links.append({"source_url": source["url"], "source_title": source["record_name"] or source["h1"],
                              "target_url": target["url"], "target_title": target["record_name"] or target["h1"],
                              "entity": next((x for x in (target["record_name"], target["h1"]) if x and alias in x.casefold()), alias),
                              "source_quote": clause, "relation_explicit": True, "target_has_specific_fact": True})
    for i, row in enumerate(links, 1):
        row["link_id"] = f"ENT-{i:05d}"
    return links


def build_index(pages_path: Path = PAGES, chunks_path: Path = CHUNKS, s1_questions_path: Path = S1_QUESTIONS,
                out_dir: Path = OUT) -> dict[str, Any]:
    pages_df = pd.read_csv(pages_path, dtype=str, keep_default_na=False)
    chunks_df = pd.read_csv(chunks_path, dtype=str, keep_default_na=False)
    frames = {u: f.sort_values("chunk_number", kind="stable") for u, f in chunks_df.groupby("url", sort=False)}
    page_texts = {u: "".join(f["content"].tolist()) for u, f in frames.items()}
    line_counts: dict[str, set[str]] = defaultdict(set)
    for url, text in page_texts.items():
        for line in set(re.sub(r"\s+", " ", x).strip() for x in text.splitlines() if len(x.strip()) >= 28):
            line_counts[line].add(url)
    repeated_lines = {line for line, urls in line_counts.items() if len(urls) >= 3}
    index_pages = []
    excluded_template_only = 0
    excluded_pages: list[dict[str, str]] = []
    for row in pages_df.to_dict("records"):
        url, ptype = row["url"], row["page_type"]
        if ptype == "news_events":
            excluded_pages.append({"url": url, "page_type": ptype, "reason": "news_events page type"})
            continue
        if row.get("eligible", "").casefold() != "true":
            excluded_pages.append({"url": url, "page_type": ptype, "reason": "page is not eligible in the cleaned page manifest"})
            continue
        if url not in frames:
            excluded_pages.append({"url": url, "page_type": ptype, "reason": "no cleaned text chunks"})
            continue
        text = page_texts[url]
        name, truncated, source = record_name(text)
        h1 = next((re.sub(r"^#\s+", "", x).strip(" #") for x in text.splitlines() if re.match(r"^#\s+", x)), "")
        fields = extract_fields(text)
        if _template_only(text, repeated_lines, name, fields):
            excluded_template_only += 1
            excluded_pages.append({"url": url, "page_type": ptype, "reason": "template-only or insufficient record content"})
            continue
        index_pages.append({"url": url, "title": name or h1 or str(frames[url].iloc[0].get("title", "")),
                            "record_name": name, "name_truncated": truncated, "record_name_source": source,
                            "h1": h1, "page_type": ptype, "url_prefix": _url_prefix(url),
                            "fields": fields, "field_quotes": {k: _field_quote(text, k, v) for k, v in fields.items()},
                            "text": text, "chunk_ids": [str(x) for x in frames[url]["id"].tolist()]})
    catalogues = _make_catalogues(index_pages)
    entity_links = _entity_links(index_pages)
    # Duplicate-name ambiguity candidates require different relevant values.
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in index_pages:
        if page["record_name"] and not page["name_truncated"]:
            by_name[page["record_name"].casefold()].append(page)
    ambiguity = []
    for name, members in by_name.items():
        if len({m["url"] for m in members}) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i+1:]:
                for field in set(a["fields"]) & set(b["fields"]):
                    va, vb = a["fields"][field], b["fields"][field]
                    if field in BAD_FIELDS or not _field_is_data_relevant(field) or _norm_value(va) == _norm_value(vb):
                        continue
                    ambiguity.append({"kind": "ambiguity", "source_kind": "same_name_different_record",
                                      "object": a["record_name"], "attribute": field, "value_a": va, "value_b": vb,
                                      "quote_a": a["field_quotes"][field], "quote_b": b["field_quotes"][field],
                                      "url_a": a["url"], "url_b": b["url"]})
    # Preserve previously judge-confirmed S1 conflict findings without changing S1.
    s1 = read_jsonl(s1_questions_path)
    by_url = {p["url"]: p for p in index_pages}
    for q in s1:
        gc = q.get("global_conflict") or {}
        if not gc.get("conflict"):
            continue
        mapping = {str(x.get("passage_id")): x for x in gc.get("passage_map", [])}
        for finding in gc.get("conflicts", []):
            other = mapping.get(str(finding.get("passage_id")), finding)
            left = by_url.get(q.get("url")); right = by_url.get(other.get("url"))
            if left and right:
                ambiguity.append({"kind": "conflict", "source_kind": "s1_judge_kb_conflict",
                                  "object": q.get("question", ""), "attribute": finding.get("what_differs", ""),
                                  "value_a": q.get("reference_answer", ""), "value_b": finding.get("what_differs", ""),
                                  "quote_a": " ".join(str(x.get("text", "")) for x in q.get("supporting_quotes", []) if isinstance(x, dict)),
                                  "quote_b": "", "url_a": left["url"], "url_b": right["url"]})
    for i, row in enumerate(ambiguity, 1):
        row["conflict_id"] = f"AMB-{i:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    field_rows = [{k: p[k] for k in ("url", "title", "record_name", "name_truncated", "record_name_source", "page_type", "url_prefix", "fields", "field_quotes")} | {"time_intervals": {field: parse_time_interval(value) for field, value in p["fields"].items() if re.search(r"time|temporal|coverage|period", field, re.I)}} for p in index_pages]
    write_jsonl(out_dir / "catalogue_fields.jsonl", field_rows)
    write_jsonl(out_dir / "catalogues.jsonl", catalogues)
    write_jsonl(out_dir / "entity_links.jsonl", entity_links)
    write_jsonl(out_dir / "conflict_candidates.jsonl", ambiguity)
    write_jsonl(out_dir / "page_exclusions.jsonl", excluded_pages)
    eligible_by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_by_prefix: dict[str, list[dict[str, str]]] = defaultdict(list)
    for page in index_pages:
        eligible_by_prefix[page["url_prefix"]].append(page)
    for item in excluded_pages:
        excluded_by_prefix[_url_prefix(item["url"])].append(item)
    catalogue_prefixes = {c["url_prefix"] for c in catalogues}
    diagnostics = []
    for prefix in sorted(set(eligible_by_prefix) | set(excluded_by_prefix)):
        group = eligible_by_prefix.get(prefix, [])
        counts = Counter(field for page in group for field in page.get("fields", {}))
        common = sorted(field for field, count in counts.items() if count == len(group) and len(group) > 1)
        catalogued = prefix in catalogue_prefixes
        diagnostics.append({"url_prefix": prefix, "eligible_members": len(group),
            "excluded_member_count": len(excluded_by_prefix.get(prefix, [])),
            "excluded_reasons": dict(Counter(x["reason"] for x in excluded_by_prefix.get(prefix, []))),
            "common_fields": common, "common_field_count": len(common),
            "field_coverage": {field: {"present": count, "member_count": len(group),
                                       "coverage_pct": round(100*count/max(1,len(group)),1)} for field,count in sorted(counts.items())},
            "included_as_catalogue": catalogued,
            "exclusion_reason": "" if catalogued else ("fewer than two eligible members" if len(group)<2 else "fewer than three shared extracted field labels")})
    write_jsonl(out_dir / "directory_diagnostics.jsonl", diagnostics)
    write_jsonl(out_dir / "indexed_pages.jsonl", [{k: p[k] for k in ("url", "title", "record_name", "name_truncated", "page_type", "url_prefix", "fields", "field_quotes", "chunk_ids")} for p in index_pages])
    etc = [c for c in catalogues if "etc l2 archive" in c["title_template"].casefold()]
    metadata = {"indexed_pages": len(index_pages), "catalogues": len(catalogues), "entity_links_before_screen": len(entity_links),
                "same_name_ambiguity_candidates": sum(x["kind"] == "ambiguity" for x in ambiguity),
                "s1_judge_flagged_conflict_observations": sum(x["kind"] == "conflict" for x in ambiguity),
                "same_object_field_conflict_candidates": 0,
                "etc_l2_archive_members": len(etc[0]["members"]) if etc else 0,
                "catalogues_by_member_count": dict(Counter(str(len(x["members"])) for x in catalogues)),
                "excluded_template_only_pages": excluded_template_only,
                "excluded_pages_by_reason": dict(Counter(row["reason"] for row in excluded_pages))}
    (out_dir / "s2_index_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = ["# S2 Index Summary", "", f"- Indexed pages: {len(index_pages)}", f"- Catalogues: {len(catalogues)}",
             f"- Entity links before judge screening: {len(entity_links)}", f"- Same-name ambiguity candidates with differing data fields: {metadata['same_name_ambiguity_candidates']}",
             f"- S1 judge-flagged conflict observations carried forward: {metadata['s1_judge_flagged_conflict_observations']}",
             f"- Same-object conflicts with a shared explicit data field: {metadata['same_object_field_conflict_candidates']}",
             f"- ETC L2 ARCHIVE members: {metadata['etc_l2_archive_members']}", "",
             "## Largest catalogues", ""]
    for c in sorted(catalogues, key=lambda x: (-len(x["members"]), x["name"]))[:15]:
        top_coverage = sorted(c["field_coverage"].items(), key=lambda item: (-item[1]["coverage_pct"], item[0]))[:8]
        coverage_text = "; ".join(f"{field} {stats['present']}/{stats['member_count']} ({stats['coverage_pct']}%)" for field, stats in top_coverage)
        lines.append(f"- {c['name']} ({c['url_prefix']}): {len(c['members'])} members; common fields: {', '.join(c['common_fields'])}; field coverage: {coverage_text}")
    lines.extend(["", "## Page exclusion reasons", ""])
    for reason, count in sorted(metadata["excluded_pages_by_reason"].items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Catalogue examples", ""])
    for c in catalogues[:5]:
        lines.append("- " + json.dumps({"name":c["name"],"member_count":len(c["members"]),"common_fields":c["common_fields"],"sample_members":[m["title"] for m in c["members"][:3]]},ensure_ascii=False))
    lines.extend(["", "## Explicit entity-link examples", ""])
    for row in entity_links[:5]: lines.append("- " + json.dumps({k:row.get(k) for k in ("source_title","entity","target_title","source_quote")},ensure_ascii=False))
    if not entity_links: lines.append("- No explicit entity links found.")
    lines.extend(["", "## Same-name ambiguity examples", ""])
    for row in [x for x in ambiguity if x["kind"]=="ambiguity"][:5]:
        lines.append("- " + json.dumps({k:row.get(k) for k in ("object","attribute","value_a","value_b","url_a","url_b")},ensure_ascii=False))
    if not any(x["kind"]=="ambiguity" for x in ambiguity): lines.append("- No eligible same-name ambiguity candidates found.")
    lines.extend(["", "## S1 judge-flagged conflict observations", "", "These are carried forward for review and are not treated as same-object conflicts unless the two pages share a record name and the same explicit data field differs.", ""])
    for row in [x for x in ambiguity if x["kind"]=="conflict"][:5]:
        lines.append("- " + json.dumps({"s1_question":row.get("object"),"difference":row.get("attribute"),"url_a":row.get("url_a"),"url_b":row.get("url_b")},ensure_ascii=False))
    if not any(x["kind"]=="conflict" for x in ambiguity): lines.append("- No S1 conflict observations found.")
    (out_dir / "s2_index_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {**metadata, "pages": index_pages, "catalogues_data": catalogues, "entity_links_data": entity_links, "conflicts_data": ambiguity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=PAGES); parser.add_argument("--chunks", type=Path, default=CHUNKS)
    parser.add_argument("--s1-questions", type=Path, default=S1_QUESTIONS); parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(); result = build_index(args.pages, args.chunks, args.s1_questions, args.out_dir)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_data") and k != "pages"}, indent=2))


if __name__ == "__main__":
    main()
