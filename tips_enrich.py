#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from tips_ingest import (
    EXTRACTED,
    PROCESSED,
    RAW_DOCS,
    ROOT,
    USER_AGENT,
    Document,
    build_index,
    clean_text,
    extract_html,
    extract_pdf,
    extract_xlsx,
    safe_slug,
    sha256_file,
    split_text,
)


CR_PAGE_URL = "https://www.ecb.europa.eu/paym/target/tips/governance/html/changerequests.en.html"
CR_LIST_GLOB = "List-of-change-requests__*.html"
GENERATED_PREFIXES = ("cr-", "acr-")
ACRONYM_STOPWORDS = {
    "ALL",
    "AND",
    "AS",
    "BUSINESS",
    "CHANGE",
    "DATA",
    "DESCRIPTION",
    "DETAILS",
    "DOCUMENT",
    "DOCUMENTS",
    "ECB",
    "ENGLISH",
    "FN",
    "EUROPEAN",
    "FOR",
    "FROM",
    "FUNCTION",
    "FUNCTIONS",
    "ITEM",
    "LIST",
    "PAGE",
    "REQUEST",
    "RIGHTS",
    "SCOPE",
    "SERVICE",
    "STATUS",
    "TARGET",
    "THE",
    "TIPS",
    "USER",
    "WITH",
}


def log(message: str) -> None:
    print(f"[tips-enrich] {message}", flush=True)


def sha1_text(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def url_extension(url: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    return suffix or ".html"


def find_cr_list_path() -> Path:
    matches = sorted((RAW_DOCS / "change_requests").glob(CR_LIST_GLOB))
    if not matches:
        raise FileNotFoundError("No local ECB change request list found under data/raw/documents/change_requests")
    return matches[0]


def release_from_status(status: str) -> str:
    match = re.search(r"R(20\d{2})[._-]?(NOV|OCT|JUN|MAR)", status or "", re.I)
    if match:
        return f"R{match.group(1)}.{match.group(2).upper()}"
    match = re.search(r"Release\s+(\d+(?:\.\d+)?)", status or "", re.I)
    if match:
        return f"Release {match.group(1)}"
    return ""


def direct_title_anchor(dd) -> object | None:
    for child in dd.find_all("div", class_="title", recursive=False):
        anchor = child.find("a", href=True)
        if anchor:
            return anchor
    return None


def direct_cr_status(dd) -> str:
    for accordion in dd.find_all("div", class_="accordion", recursive=False):
        children = [child for child in accordion.children if getattr(child, "name", None) == "div"]
        for index, child in enumerate(children):
            if "content-box" not in (child.get("class") or []):
                continue
            terms = [node for node in child.find_all(["dt", "dd"], recursive=True)]
            for pos, term in enumerate(terms[:-1]):
                if term.name == "dt" and term.get_text(" ", strip=True) == "TIPS CR status":
                    return " ".join(terms[pos + 1].get_text(" ", strip=True).split())
    return ""


def annexes_for(dd: object, page_url: str) -> list[dict]:
    annexes: list[dict] = []
    for header in dd.find_all("div", class_="header"):
        title = " ".join(header.get_text(" ", strip=True).split())
        if title.lower() != "annexes":
            continue
        box = header.find_next_sibling("div", class_="content-box")
        if not box:
            continue
        for anchor in box.find_all("a", href=True):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            href = urljoin(page_url, anchor["href"])
            if text and href not in {item["url"] for item in annexes}:
                annexes.append({"title": text, "url": href})
    return annexes


def parse_change_requests() -> list[dict]:
    path = find_cr_list_path()
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    main = soup.find("main") or soup
    entries: list[dict] = []
    seen_urls: set[str] = set()
    for dt in main.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        anchor = direct_title_anchor(dd)
        if not anchor:
            continue
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if title.lower().startswith("annex "):
            continue
        match = re.search(r"\b(TIPS-\d{4}(?:-[A-Z0-9]+)?)\b", title)
        if not match:
            continue
        url = urljoin(CR_PAGE_URL, anchor["href"])
        if url in seen_urls:
            continue
        seen_urls.add(url)
        status = direct_cr_status(dd)
        date = dt.get("isoDate") or " ".join(dt.get_text(" ", strip=True).split())
        code = match.group(1)
        entries.append(
            {
                "code": code,
                "kind": code.split("-")[-1] if "-" in code else "",
                "title": title,
                "date": date,
                "status": status,
                "release": release_from_status(status),
                "published": True,
                "url": url,
                "annexes": annexes_for(dd, CR_PAGE_URL),
            }
        )
    entries.sort(key=lambda item: (item["date"], item["code"]), reverse=True)
    return entries


def local_download_path(title: str, url: str, doc_id: str) -> Path:
    suffix = url_extension(url)
    name = safe_slug(title, 95)
    return RAW_DOCS / "change_requests" / "files" / f"{name}__{doc_id}{suffix}"


def download_file(session: requests.Session, url: str, title: str, doc_id: str, force: bool) -> Path:
    target = local_download_path(title, url, doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    with session.get(url, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as fh:
            for block in response.iter_content(chunk_size=1024 * 256):
                if block:
                    fh.write(block)
        tmp.replace(target)
    return target


def extract_units(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".xlsx", ".xlsm"}:
        return extract_xlsx(path)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    try:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        text = ""
    return [{"unit_type": "file", "unit": 1, "text": text}] if text else []


def cr_structured_text(entry: dict, local_path: str = "") -> str:
    release = entry.get("release") or "No release/status assigned in the ECB list"
    annexes = entry.get("annexes") or []
    annex_text = "\n".join(f"- {item['title']} ({item['url']})" for item in annexes) or "None"
    return clean_text(
        f"""TIPS Change Request
Code: {entry['code']}
Title: {entry['title']}
Publication date on ECB page: {entry['date']}
TIPS CR status: {entry.get('status') or 'No TIPS CR status shown in the ECB list'}
Release/status bucket: {release}
Published on ECB change request page: {'yes' if entry.get('published') else 'no'}
CR source URL: {entry['url']}
Local file: {local_path}
Annexes:
{annex_text}"""
    )


def create_change_request_docs(entries: list[dict], force: bool) -> tuple[list[Document], list[dict]]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    docs: list[Document] = []
    chunks: list[dict] = []
    files_manifest: list[dict] = []

    summary_id = "cr-index"
    summary_path = PROCESSED / "change_requests.json"
    summary_doc = Document(
        id=summary_id,
        title="Structured TIPS change request catalogue",
        url=CR_PAGE_URL,
        ext=".json",
        category="change_requests",
        family="change_requests",
        status="downloaded",
        local_path=str(summary_path.relative_to(ROOT)),
        source_host="www.ecb.europa.eu",
        extracted_chars=summary_path.stat().st_size if summary_path.exists() else 0,
        extracted_units=len(entries),
        text_path=str(summary_path.relative_to(ROOT)),
        contexts=[{"h3": "Change Requests", "accordion": ["Structured catalogue"], "path": ["Change Requests"]}],
    )
    docs.append(summary_doc)

    release_counter = Counter(entry.get("release") or "No release/status shown" for entry in entries)
    summary_lines = [
        "Structured TIPS change request catalogue",
        f"Total published entries on ECB page: {len(entries)}",
        "Entries by TIPS CR status/release:",
        *[f"- {release}: {count}" for release, count in sorted(release_counter.items())],
    ]
    chunks.append(
        {
            "chunk_id": f"{summary_id}:summary:1",
            "doc_id": summary_id,
            "title": summary_doc.title,
            "category": summary_doc.category,
            "family": summary_doc.family,
            "release": "",
            "revision_status": "structured",
            "context_path": ["Change Requests", "Structured catalogue"],
            "unit_type": "change_request_catalogue",
            "unit": "summary",
            "text": "\n".join(summary_lines),
            "local_path": summary_doc.local_path,
            "source_url": CR_PAGE_URL,
        }
    )

    for entry in entries:
        doc_id = f"cr-{sha1_text(entry['url'])}"
        local_path = ""
        units: list[dict] = []
        error = ""
        try:
            downloaded = download_file(session, entry["url"], entry["title"], doc_id, force=force)
            local_path = str(downloaded.relative_to(ROOT))
            units = extract_units(downloaded)
            text = "\n\n".join(f"[{unit.get('unit_type')} {unit.get('unit')}]\n{unit.get('text', '')}" for unit in units)
            text_path = EXTRACTED / f"{doc_id}.txt"
            text_path.write_text(clean_text(f"{cr_structured_text(entry, local_path)}\n\n{text}"), encoding="utf-8")
            sha256 = sha256_file(downloaded)
            size = downloaded.stat().st_size
        except Exception as exc:
            error = repr(exc)
            text_path = EXTRACTED / f"{doc_id}.txt"
            text_path.write_text(cr_structured_text(entry, local_path), encoding="utf-8")
            sha256 = ""
            size = 0

        doc = Document(
            id=doc_id,
            title=entry["title"],
            url=entry["url"],
            ext=url_extension(entry["url"]),
            category="change_requests",
            family="change_requests",
            release=entry.get("release") or "",
            revision_status=entry.get("status") or "published_no_status",
            status="downloaded" if not error else "failed",
            local_path=local_path,
            source_host="www.ecb.europa.eu",
            error=error,
            sha256=sha256,
            size_bytes=size,
            extracted_chars=text_path.stat().st_size,
            extracted_units=max(1, len(units)),
            text_path=str(text_path.relative_to(ROOT)),
            contexts=[
                {
                    "h3": "Change Requests",
                    "accordion": [entry.get("release") or entry.get("status") or "No status", entry["code"]],
                    "path": ["Change Requests", entry.get("release") or entry.get("status") or "No status", entry["code"]],
                }
            ],
        )
        docs.append(doc)
        chunks.append(
            {
                "chunk_id": f"{doc_id}:structured:1",
                "doc_id": doc.id,
                "title": doc.title,
                "category": doc.category,
                "family": doc.family,
                "release": doc.release,
                "revision_status": doc.revision_status,
                "context_path": doc.contexts[0]["path"],
                "unit_type": "change_request",
                "unit": entry["code"],
                "text": cr_structured_text(entry, local_path),
                "local_path": doc.local_path,
                "source_url": doc.url,
            }
        )
        for unit in units:
            for index, piece in enumerate(split_text(unit.get("text", ""), max_chars=1800, overlap=220)):
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}:{unit.get('unit_type')}:{unit.get('unit')}:{index}",
                        "doc_id": doc.id,
                        "title": doc.title,
                        "category": doc.category,
                        "family": doc.family,
                        "release": doc.release,
                        "revision_status": doc.revision_status,
                        "context_path": doc.contexts[0]["path"],
                        "unit_type": unit.get("unit_type"),
                        "unit": unit.get("unit"),
                        "text": piece,
                        "local_path": doc.local_path,
                        "source_url": doc.url,
                    }
                )
        files_manifest.append({"code": entry["code"], "title": entry["title"], "local_path": local_path, "error": error})
        time.sleep(0.02)
    return docs, chunks


def load_existing_processed() -> tuple[list[dict], list[dict]]:
    docs_path = PROCESSED / "documents.json"
    chunks_path = PROCESSED / "chunks.jsonl"
    if not docs_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Run python tips_ingest.py first")
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    docs = [doc for doc in docs if not str(doc.get("id", "")).startswith(GENERATED_PREFIXES) and doc.get("id") != "cr-index"]
    chunks = [chunk for chunk in chunks if not str(chunk.get("doc_id", "")).startswith(GENERATED_PREFIXES) and chunk.get("doc_id") != "cr-index"]
    return docs, chunks


def acronym_candidates_from_text(text: str) -> list[tuple[str, str]]:
    if "List of acronyms" not in text and "List of abbreviations" not in text:
        return []
    normalized = re.sub(r"\s+", " ", text)
    start = normalized.find("List of acronyms")
    if start < 0:
        start = normalized.find("List of abbreviations")
    segment = normalized[start : start + 9000]
    token_re = re.compile(r"\b([A-Z][A-Z0-9/]{1,6})\b")
    matches = [match for match in token_re.finditer(segment) if match.group(1) not in ACRONYM_STOPWORDS]
    pairs: list[tuple[str, str]] = []
    for current, nxt in zip(matches, matches[1:]):
        acronym = current.group(1)
        definition = segment[current.end() : nxt.start()].strip(" -:;,.")
        if not definition or len(definition) > 120:
            continue
        if re.search(r"\b(Page|All rights reserved|Item Description)\b", definition):
            continue
        words = definition.split()
        if len(words) > 9 or len(words) < 1:
            continue
        if not any(char.isalpha() for char in definition):
            continue
        if len(acronym) == 2 and acronym in {"ID", "NO", "ON"}:
            continue
        pairs.append((acronym, definition))
    return pairs


def build_acronym_docs(base_docs: list[dict], base_chunks: list[dict]) -> tuple[list[Document], list[dict], list[dict]]:
    by_acronym: dict[str, list[dict]] = defaultdict(list)
    by_doc = {doc["id"]: doc for doc in base_docs}
    for chunk in base_chunks:
        text = chunk.get("text", "")
        for acronym, definition in acronym_candidates_from_text(text):
            by_acronym[acronym].append(
                {
                    "definition": definition,
                    "title": chunk.get("title"),
                    "release": chunk.get("release"),
                    "unit_type": chunk.get("unit_type"),
                    "unit": chunk.get("unit"),
                    "local_path": chunk.get("local_path"),
                    "source_url": chunk.get("source_url"),
                    "family": chunk.get("family"),
                }
            )

    manual = {
        "CR": "Change Request",
        "CRs": "Change Requests",
        "LKT": "Linked Transaction Model",
        "SDD": "Scope Defining Document",
        "SDDs": "Scope Defining Documents",
    }
    for acronym, definition in manual.items():
        by_acronym[acronym].append(
            {
                "definition": definition,
                "title": "Structured TIPS glossary",
                "release": "",
                "unit_type": "glossary",
                "unit": acronym,
                "local_path": "data\\processed\\acronyms.json",
                "source_url": CR_PAGE_URL if acronym.startswith("CR") else "",
                "family": "acronyms",
            }
        )

    entries: list[dict] = []
    for acronym, sources in sorted(by_acronym.items()):
        definitions = Counter(source["definition"] for source in sources)
        definition = manual.get(acronym, definitions.most_common(1)[0][0])
        if acronym in manual:
            normalized_sources = []
            for source in sources:
                normalized_source = dict(source)
                normalized_source["definition"] = definition
                normalized_sources.append(normalized_source)
            sources = normalized_sources
        source_items = [source for source in sources if source["definition"] == definition][:8]
        entries.append({"acronym": acronym, "definition": definition, "sources": source_items})

    acronyms_path = PROCESSED / "acronyms.json"
    acronyms_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    doc = Document(
        id="acr-index",
        title="Structured TIPS acronym dictionary",
        url=str(acronyms_path.relative_to(ROOT)),
        ext=".json",
        category="acronyms",
        family="acronyms",
        status="downloaded",
        local_path=str(acronyms_path.relative_to(ROOT)),
        source_host="local",
        sha256=sha256_file(acronyms_path),
        size_bytes=acronyms_path.stat().st_size,
        extracted_chars=acronyms_path.stat().st_size,
        extracted_units=len(entries),
        text_path=str(acronyms_path.relative_to(ROOT)),
        contexts=[{"h3": "Acronyms", "accordion": ["Structured dictionary"], "path": ["Acronyms"]}],
    )
    chunks: list[dict] = []
    for entry in entries:
        source_lines = []
        for index, source in enumerate(entry["sources"], start=1):
            source_lines.append(
                f"[{index}] {source.get('title')} {source.get('release') or ''} "
                f"{source.get('unit_type')} {source.get('unit')} - {source.get('local_path')}"
            )
        text = clean_text(
            f"""Acronym: {entry['acronym']}
Definition: {entry['definition']}
This is a structured TIPS acronym dictionary entry generated from local TIPS documentation.
Sources:
{chr(10).join(source_lines)}"""
        )
        chunks.append(
            {
                "chunk_id": f"acr-index:{entry['acronym']}",
                "doc_id": doc.id,
                "title": doc.title,
                "category": doc.category,
                "family": doc.family,
                "release": "",
                "revision_status": "structured",
                "context_path": ["Acronyms", entry["acronym"]],
                "unit_type": "acronym",
                "unit": entry["acronym"],
                "text": text,
                "local_path": doc.local_path,
                "source_url": doc.url,
            }
        )
    return [doc], chunks, entries


def enrich(force: bool = False) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    existing_docs, existing_chunks = load_existing_processed()

    cr_entries = parse_change_requests()
    (PROCESSED / "change_requests.json").write_text(json.dumps(cr_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"parsed {len(cr_entries)} ECB TIPS change requests")
    cr_docs, cr_chunks = create_change_request_docs(cr_entries, force=force)
    log(f"prepared {len(cr_docs)} structured/downloaded change request docs and {len(cr_chunks)} chunks")

    acr_docs, acr_chunks, acronym_entries = build_acronym_docs(existing_docs + [asdict(doc) for doc in cr_docs], existing_chunks + cr_chunks)
    log(f"built acronym dictionary with {len(acronym_entries)} entries")

    all_docs_payload = existing_docs + [asdict(doc) for doc in cr_docs + acr_docs]
    all_chunks = existing_chunks + cr_chunks + acr_chunks
    docs = [Document(**doc) for doc in all_docs_payload]
    build_index(docs, all_chunks)
    summary = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    summary["structured_change_requests"] = len(cr_entries)
    summary["structured_acronyms"] = len(acronym_entries)
    (PROCESSED / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add structured TIPS CR and acronym data to the local index.")
    parser.add_argument("--force", action="store_true", help="redownload existing CR files")
    args = parser.parse_args(argv)
    try:
        summary = enrich(force=args.force)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
