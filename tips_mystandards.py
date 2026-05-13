#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

import requests

from tips_ingest import (
    EXTRACTED,
    PROCESSED,
    RAW,
    ROOT,
    Document,
    build_index,
    clean_text,
    safe_slug,
    sha256_file,
    split_text,
)


VERSION = "UDFS TIPS R2026.NOV"
VERSION_SLUG = "r2026_nov"
MYSTANDARDS = "https://www2.swift.com/mystandards"
EXPORT_SCHEMA_PACKAGE = "srv/com.swift.mystandards.service.generate.Generator/exportSchemaPackage"
RAW_MS = RAW / "mystandards" / VERSION_SLUG
CLAWDBOT = shutil.which("clawdbot.cmd") or shutil.which("clawdbot") or "clawdbot"


def log(message: str) -> None:
    print(f"[tips-mystandards] {message}", flush=True)


def run_json(args: list[str]) -> Any:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(result.stdout)


def base_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"{MYSTANDARDS}/",
            "Origin": "https://www2.swift.com",
            "x-requested-with": "XMLHttpRequest",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return session


def apply_browser_cookies(session: requests.Session, cookies: list[dict]) -> requests.Session:
    xsrf = ""
    browser_cookie_header: list[str] = []
    target_host = "www2.swift.com"
    request_path = "/mystandards/api/"
    eligible_for_api: list[dict] = []
    for cookie in cookies:
        domain = cookie.get("domain") or ""
        if "swift.com" not in domain:
            continue
        session.cookies.set(
            cookie.get("name"),
            cookie.get("value"),
            domain=domain,
            path=cookie.get("path") or "/",
        )
        if cookie.get("name") == "mystandards-XSRF-TOKEN":
            xsrf = cookie.get("value") or ""
        clean_domain = domain.lstrip(".")
        cookie_path = cookie.get("path") or "/"
        root_path = cookie_path.rstrip("/") or "/"
        domain_matches = target_host == clean_domain or target_host.endswith(f".{clean_domain}")
        path_matches = request_path.startswith(root_path)
        if domain_matches and path_matches:
            eligible_for_api.append(cookie)
    if xsrf:
        session.headers["mystandards-x-xsrf-token"] = urllib.parse.unquote(xsrf)
    eligible_for_api.sort(key=lambda item: len(item.get("path") or ""), reverse=True)
    for cookie in eligible_for_api:
        browser_cookie_header.append(f"{cookie.get('name')}={cookie.get('value')}")
    if browser_cookie_header:
        session.headers["Cookie"] = "; ".join(browser_cookie_header)
    return session


def find_target_id(profile: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    tabs = run_json([CLAWDBOT, "browser", "tabs", "--browser-profile", profile, "--json"])
    for tab in tabs.get("tabs", []):
        if "mystandards" in (tab.get("url") or ""):
            return tab["targetId"]
    raise RuntimeError(f"No MyStandards tab found in browser profile {profile!r}")


def cookies_from_playwright(profile: str) -> list[dict]:
    from playwright.sync_api import sync_playwright

    status = run_json([CLAWDBOT, "browser", "status", "--browser-profile", profile, "--json"])
    cdp_url = status.get("cdpUrl") or f"http://127.0.0.1:{status.get('cdpPort')}"
    if not cdp_url or cdp_url.endswith(":None"):
        raise RuntimeError(f"Could not resolve Chrome CDP URL for browser profile {profile!r}: {status}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        cookies: list[dict] = []
        for context in browser.contexts:
            cookies.extend(context.cookies())
        return cookies


def session_from_clawdbot(profile: str, target_id: str) -> requests.Session:
    session = base_session()
    try:
        return apply_browser_cookies(session, cookies_from_playwright(profile))
    except Exception as exc:
        log(f"Playwright CDP cookies unavailable, trying clawdbot cookies: {exc}")
    try:
        cookie_json = run_json(
            [
                CLAWDBOT,
                "browser",
                "cookies",
                "--target-id",
                target_id,
                "--browser-profile",
                profile,
                "--json",
            ]
        )
        return apply_browser_cookies(session, cookie_json.get("cookies", []))
    except Exception as exc:
        raise RuntimeError(f"Could not read MyStandards cookies from profile {profile!r}: {exc}") from exc


def search_r2026_nov(session: requests.Session, page_size: int = 100) -> list[dict]:
    payload = {
        "query": "TIPS",
        "pageNbr": 1,
        "sortBy": "",
        "sortOrder": "desc",
        "pageSize": page_size,
        "filterParameters": [{"filterName": "metadata/common/version", "filters": VERSION}],
        "exactMatch": False,
        "exists": [],
        "isNull": [],
    }
    response = session.post(f"{MYSTANDARDS}/api/search/usageguideline", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    hits = data.get("hits", {}).get("hits", [])
    if total > len(hits):
        payload["pageSize"] = total
        response = session.post(f"{MYSTANDARDS}/api/search/usageguideline", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
    RAW_MS.mkdir(parents=True, exist_ok=True)
    (RAW_MS / "search_response.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return hits


def get_ug_header(session: requests.Session, ug_uri: str) -> dict | None:
    response = session.get(f"{MYSTANDARDS}/api/public/{ug_uri}/header", timeout=60)
    if response.status_code != 200:
        return None
    return response.json()


def export_schema_package(session: requests.Session, ug_uri: str) -> bytes:
    encoded_uri = urllib.parse.quote(ug_uri, safe="")
    encoded_export = urllib.parse.quote(EXPORT_SCHEMA_PACKAGE, safe="")
    url = f"{MYSTANDARDS}/api/public/export/get?exportType={encoded_export}&objectUri={encoded_uri}"
    response = session.get(url, timeout=120)
    response.raise_for_status()
    data = response.json()
    path = data.get("path")
    uuid = (data.get("params") or {}).get("uuid")
    if not path or not uuid:
        raise RuntimeError(f"Export response missing path/uuid for {ug_uri}: {data}")
    download = f"{MYSTANDARDS}/api/public/{path}?uuid={urllib.parse.quote(uuid)}"
    dl = session.get(download, timeout=180)
    dl.raise_for_status()
    if not dl.content.startswith(b"PK"):
        raise RuntimeError(f"Downloaded schema package is not a zip for {ug_uri}: {dl.content[:80]!r}")
    return dl.content


def extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    extracted: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".xsd", ".xml", ".json", ".txt"}:
                continue
            name = safe_slug(Path(info.filename).name, max_len=120)
            if not Path(name).suffix:
                name = f"{name}{suffix}"
            max_name_len = max(24, 235 - len(str(dest)) - 1)
            if len(name) > max_name_len:
                stem = Path(name).stem
                suffix = Path(name).suffix
                digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
                keep = max(8, max_name_len - len(suffix) - len(digest) - 1)
                name = f"{stem[:keep]}-{digest}{suffix}"
            out = dest / name
            counter = 2
            while out.exists():
                out = dest / f"{out.stem}-{counter}{out.suffix}"
                counter += 1
            with archive.open(info) as source, out.open("wb") as target:
                target.write(source.read())
            extracted.append(out)
    return extracted


def text_from_files(files: list[Path]) -> str:
    parts: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        text = clean_text(text)
        if text:
            rel = path.relative_to(ROOT)
            parts.append(f"FILE: {rel}\n{text}")
    return "\n\n".join(parts)


def doc_id_for(uri: str) -> str:
    return hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]


def append_to_processed(docs: list[Document], chunks: list[dict]) -> None:
    documents_path = PROCESSED / "documents.json"
    chunks_path = PROCESSED / "chunks.jsonl"
    if not documents_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Run python tips_ingest.py before adding MyStandards material")

    existing_docs = json.loads(documents_path.read_text(encoding="utf-8"))
    by_id = {doc["id"]: doc for doc in existing_docs}
    for doc in docs:
        by_id[doc.id] = doc.to_dict()
    documents_path.write_text(
        json.dumps(list(by_id.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    existing_chunks: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                existing_chunks.append(json.loads(line))
    existing_chunks = [chunk for chunk in existing_chunks if not str(chunk.get("doc_id", "")).startswith("ms-")]
    existing_chunks.extend(chunks)
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in existing_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    build_index([Document(**doc) for doc in by_id.values()], existing_chunks)


def ingest_mystandards(profile: str, target_id: str = "", force: bool = False, delay: float = 0.15) -> dict:
    resolved_target = find_target_id(profile, target_id)
    session = session_from_clawdbot(profile, resolved_target)
    hits = search_r2026_nov(session)
    log(f"found {len(hits)} MyStandards usage guidelines for {VERSION}")

    docs: list[Document] = []
    chunks: list[dict] = []
    manifest: list[dict] = []
    packages_dir = RAW_MS / "schema_packages"
    extracted_dir = RAW_MS / "extracted"
    packages_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for index, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        uri = source.get("URI")
        name = source.get("name") or uri or "usage-guideline"
        if not uri:
            continue
        doc_id = f"ms-{doc_id_for(uri)}"
        message_id = ((source.get("restrictedMessage") or {}).get("title") or "").strip()
        status = source.get("calculatedVersionStatus") or "Not specified"
        collection = (source.get("collection") or {}).get("name")
        publish_date = (source.get("collection") or {}).get("publishingDate")
        safe_name = safe_slug(name, 100)
        out_dir = extracted_dir / f"{safe_slug(name, 45)}__{doc_id}"
        zip_path = packages_dir / f"{safe_name}__{doc_id}.zip"

        try:
            if not zip_path.exists() or force:
                log(f"exporting {index}/{len(hits)}: {name}")
                zip_path.write_bytes(export_schema_package(session, uri))
            files = extract_zip(zip_path, out_dir)
            header = get_ug_header(session, uri) or {}
            metadata = {
                "search_hit": hit,
                "header": header,
                "extracted_files": [str(path.relative_to(ROOT)) for path in files],
            }
            meta_path = out_dir / "mystandards_metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            text = text_from_files(files)
            intro = "\n".join(
                [
                    f"MyStandards usage guideline: {name}",
                    f"Version: {VERSION}",
                    f"Status: {status}",
                    f"Restricted message: {message_id}",
                    f"Collection: {collection}",
                    f"Publishing date: {publish_date}",
                    f"URI: {uri}",
                ]
            )
            full_text = clean_text(f"{intro}\n\n{text}")
            text_path = EXTRACTED / f"{doc_id}.txt"
            text_path.write_text(full_text, encoding="utf-8")

            doc = Document(
                id=doc_id,
                title=f"MyStandards {VERSION} - {name}",
                url=f"{MYSTANDARDS}/#/{uri}",
                ext=".zip",
                category="mystandards_messages",
                family="mystandards_udfs",
                release="R2026.NOV",
                revision_status="work_in_progress" if "progress" in status.lower() else status.lower().replace(" ", "_"),
                status="downloaded",
                local_path=str(zip_path.relative_to(ROOT)),
                source_host="www2.swift.com",
                sha256=sha256_file(zip_path),
                size_bytes=zip_path.stat().st_size,
                extracted_chars=len(full_text),
                extracted_units=len(files),
                text_path=str(text_path.relative_to(ROOT)),
                contexts=[
                    {
                        "h3": "MyStandards UDFS TIPS R2026.NOV",
                        "accordion": [VERSION, message_id, collection],
                        "path": ["MyStandards", VERSION, message_id, collection],
                    }
                ],
            )
            docs.append(doc)

            for part_index, piece in enumerate(split_text(full_text, max_chars=1800, overlap=250)):
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}:schema:{part_index}",
                        "doc_id": doc.id,
                        "title": doc.title,
                        "category": doc.category,
                        "family": doc.family,
                        "release": doc.release,
                        "revision_status": doc.revision_status,
                        "context_path": ["MyStandards", VERSION, message_id, collection],
                        "unit_type": "schema_package",
                        "unit": message_id or name,
                        "text": piece,
                        "local_path": doc.local_path,
                        "source_url": doc.url,
                    }
                )
            manifest.append(
                {
                    "name": name,
                    "uri": uri,
                    "message_id": message_id,
                    "status": status,
                    "collection": collection,
                    "publishing_date": publish_date,
                    "zip": str(zip_path.relative_to(ROOT)),
                    "files": [str(path.relative_to(ROOT)) for path in files],
                    "chunks": sum(1 for chunk in chunks if chunk["doc_id"] == doc.id),
                }
            )
            time.sleep(delay)
        except Exception as exc:
            log(f"failed {name}: {exc}")
            manifest.append({"name": name, "uri": uri, "error": repr(exc)})

    (RAW_MS / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    append_to_processed(docs, chunks)
    summary = {
        "version": VERSION,
        "profile": profile,
        "target_id": resolved_target,
        "usage_guidelines": len(hits),
        "downloaded": len(docs),
        "chunks": len(chunks),
        "manifest": str((RAW_MS / "manifest.json").relative_to(ROOT)),
    }
    (RAW_MS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done: {summary['downloaded']} MyStandards packages, {summary['chunks']} chunks")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest MyStandards TIPS R2026.NOV schema packages.")
    parser.add_argument("--profile", default="choupo", help="clawdbot browser profile with active MyStandards session")
    parser.add_argument("--target-id", default="", help="optional MyStandards browser target id")
    parser.add_argument("--force", action="store_true", help="redownload existing schema packages")
    args = parser.parse_args(argv)
    try:
        ingest_mystandards(profile=args.profile, target_id=args.target_id, force=args.force)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
