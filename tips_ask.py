#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"
ACRONYMS_PATH = ROOT / "data" / "processed" / "acronyms.json"
CHANGE_REQUESTS_PATH = ROOT / "data" / "processed" / "change_requests.json"
CODEX = shutil.which("codex.cmd") or shutil.which("codex") or "codex"
MODEL_PRESETS = {
    "codex_high": {"label": "Codex High", "reasoning": "high"},
    "codex_fast": {"label": "Codex Fast", "reasoning": "low"},
    "local_rag": {"label": "Solo RAG", "reasoning": "none"},
}
GENERATION_CONTEXT_HITS = 24
GENERATION_RETRIEVAL_HITS = 32
MAX_CONTEXT_HITS = 56
ACRONYM_DEFINITION_OVERRIDES = {
    "UDFS": "User Detailed Functional Specifications",
    "UHB": "User Handbook",
    "LEA": "Legal Archiving",
}
CURRENCY_CODES = {"EUR", "SEK", "DKK", "NOK"}

FAMILY_HINTS = {
    "connectivity": [
        "connectivity",
        "mept",
        "message exchange",
        "network service provider",
        "nsp",
        "gosign",
        "technical requirement",
    ],
    "pricing": ["pricing", "fee", "billing", "price", "tariff", "cost"],
    "tips_udfs": ["udfs", "functional specification", "message", "schema", "xsd", "xml", "pacs", "camt"],
    "mystandards_udfs": [
        "mystandards",
        "my standards",
        "usage guideline",
        "schema",
        "schemas",
        "xsd",
        "xml",
        "r2026",
        "nov.2026",
    ],
    "mpl_udfs": ["mpl", "mobile proxy lookup", "proxy"],
    "tips_uhb": ["uhb", "user handbook", "gui", "screen", "user interface"],
    "tips_urd": ["urd", "user requirement", "requirement"],
    "participation": ["participation", "participant", "onboarding", "reachability", "psp", "non-bank"],
    "legal": ["legal", "terms", "condition", "guideline", "access policy", "hosting"],
    "release_documentation": ["release", "milestone", "r2026", "r2025", "r2024", "r2023"],
    "production_problems": ["production problem", "incident", "outage", "problem"],
    "training_and_featured_topics": ["training", "validation", "sip", "cross-currency", "ntc", "crdm"],
    "change_requests": ["change request", "change requests", "cr", "crs", "publicada", "publicadas", "published", "status"],
    "acronyms": ["acronym", "acronimo", "acronimos", "sigla", "siglas", "significa", "meaning"],
}

RELEASE_RE = re.compile(r"R(20\d{2})[._-]?(NOV|OCT|JUN|MAR)", re.I)
RELEASE_REVERSE_RE = re.compile(r"\b(NOV|OCT|JUN|MAR)[._ -]?(20\d{2})\b", re.I)
MESSAGE_RE = re.compile(r"\b(acmt|admi|camt|pacs|reda)\.(\d{3})\b", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,8}\b")
SPANISH_LANGUAGE_HINTS = {
    "como",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "dame",
    "de",
    "del",
    "el",
    "en",
    "es",
    "explica",
    "hay",
    "la",
    "las",
    "los",
    "partes",
    "que",
    "qué",
    "quien",
    "quién",
    "son",
    "transaccion",
    "transacción",
    "una",
}
ENGLISH_LANGUAGE_HINTS = {
    "are",
    "does",
    "explain",
    "how",
    "in",
    "is",
    "of",
    "parties",
    "the",
    "transaction",
    "what",
    "which",
    "who",
}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "como",
    "con",
    "cual",
    "cuales",
    "cuáles",
    "de",
    "del",
    "dame",
    "el",
    "en",
    "es",
    "explica",
    "for",
    "is",
    "la",
    "las",
    "los",
    "para",
    "que",
    "qué",
    "sobre",
    "the",
    "what",
    "which",
    "y",
}


def detect_question_language(query: str) -> str:
    lower = query.lower()
    if re.search(r"[¿¡áéíóúñ]", lower):
        return "es"
    tokens = set(re.findall(r"\b[\wáéíóúñ]+\b", lower, flags=re.I))
    spanish_score = len(tokens & SPANISH_LANGUAGE_HINTS)
    english_score = len(tokens & ENGLISH_LANGUAGE_HINTS)
    if english_score > spanish_score:
        return "en"
    return "es"


@dataclass
class Hit:
    rank: int
    score: float
    chunk: dict[str, Any]
    reason: str = ""

    @property
    def citation(self) -> str:
        title = self.chunk.get("title") or "Untitled"
        unit_type = self.chunk.get("unit_type") or "unit"
        unit = self.chunk.get("unit")
        where = f"{unit_type} {unit}" if unit not in (None, "") else unit_type
        return f"{title}, {where}"


@lru_cache(maxsize=2)
def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Index not found at {path}. Run `python tips_ingest.py` first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=8)
def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def acronym_entries_by_key() -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for entry in load_json(ACRONYMS_PATH):
        acronym = str(entry.get("acronym", "")).strip()
        if acronym:
            mapping[acronym.upper()] = entry
    return mapping


def normalize_query(query: str) -> str:
    query = query.strip()
    replacements = {
        "liquidez": "liquidity",
        "precios": "pricing fees",
        "tarifas": "pricing fees",
        "conectividad": "connectivity MEPT NSP",
        "tipa": "TIPS TARGET Instant Payment Settlement",
        "divisa": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "divisas": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "moneda": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "monedas": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "currencies": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "currency": "currency currencies TIPS settles instant payments euro Swedish kronor Danish kroner EUR SEK DKK",
        "mensajes": "messages schemas",
        "mensaje": "message schema",
        "parties": "involved actors participants originator participant beneficiary participant leg exit psp leg entry psp source leg destination leg",
        "party": "involved actors participants originator participant beneficiary participant leg exit psp leg entry psp source leg destination leg",
        "actores": "involved actors participants originator participant beneficiary participant leg exit psp leg entry psp",
        "participantes": "involved actors participants originator participant beneficiary participant leg exit psp leg entry psp",
        "transaccion": "transaction instant payment source leg destination leg",
        "transacción": "transaction instant payment source leg destination leg",
        "olo": "one-leg out cross-currency outgoing incoming leg exit psp leg entry psp source leg destination leg corridor",
        "esquema": "schema xsd usage guideline mystandards",
        "esquemas": "schemas xsd usage guideline mystandards",
        "rechazo": "rejection",
        "rechazos": "rejections",
        "participantes": "participants",
        "alta": "onboarding participation",
        "incidencias": "production problems incidents",
        "manual": "user handbook UHB",
        "requisitos": "requirements",
        "funcional": "functional",
        "crs": "change requests TIPS CR status published release scope",
        "cr": "change request TIPS CR status published release scope",
        "change request": "change request TIPS CR status published release scope",
        "publicada": "published publication status final version ECB webpages",
        "publicadas": "published publication status final version ECB webpages",
        "publicados": "published publication status final version ECB webpages",
        "acronimo": "acronym definition meaning glossary",
        "acronimos": "acronyms definition meaning glossary",
        "sigla": "acronym definition meaning glossary",
        "siglas": "acronyms definition meaning glossary",
    }
    tokens = re.findall(r"[\w./-]+", query, flags=re.UNICODE)
    cleaned = " ".join(token for token in tokens if token.lower() not in QUESTION_STOPWORDS)
    expanded = [cleaned or query]
    low = query.lower()
    for key, value in replacements.items():
        if key in low:
            expanded.append(value)
    return " ".join(expanded)


def query_release(query: str) -> str:
    match = RELEASE_RE.search(query)
    if match:
        return f"R{match.group(1)}.{match.group(2).upper()}"
    match = RELEASE_REVERSE_RE.search(query)
    if match:
        return f"R{match.group(2)}.{match.group(1).upper()}"
    return ""


def query_message_codes(query: str) -> list[str]:
    codes: list[str] = []
    for match in MESSAGE_RE.finditer(query):
        code = f"{match.group(1).lower()}.{match.group(2)}"
        if code not in codes:
            codes.append(code)
    return codes


def query_cr_codes(query: str) -> list[str]:
    entries = cr_entries_by_code()
    if not entries:
        return []
    candidates: list[str] = []
    for match in re.finditer(r"\bTIPS[- .]?(\d{4})(?:[- .]?(URD|SYS))?\b", query, re.I):
        number, kind = match.group(1), match.group(2)
        candidates.append(f"TIPS-{number}-{kind.upper()}" if kind else f"TIPS-{number}")
    if is_change_request_question(query):
        for match in re.finditer(r"\b(?:CR|CRS)\s*[-#:]?\s*(\d{4})\b", query, re.I):
            candidates.append(f"TIPS-{match.group(1)}")
        if re.search(r"\b(?:cr|crs|change request|resumen|summary|resume|summarize)\b", query, re.I):
            for match in re.finditer(r"\b(\d{4})\b", query):
                candidates.append(f"TIPS-{match.group(1)}")

    resolved: list[str] = []
    for candidate in candidates:
        code = clean_cr_code(candidate)
        matches = [key for key in entries if key == code or key.startswith(f"{code}-")]
        matches.sort(key=lambda key: (entries[key].get("release") or "", key), reverse=True)
        for match in matches:
            if match not in resolved:
                resolved.append(match)
                break
    return resolved


def query_acronyms(query: str) -> list[str]:
    known = acronym_entries_by_key()
    stop = {item.upper() for item in QUESTION_STOPWORDS} | {
        "ACRONYM",
        "ACRONIMO",
        "DEFINITION",
        "DEFINE",
        "ECB",
        "FUNCTION",
        "FUNCTIONS",
        "LEG",
        "MEANING",
        "SOURCE",
        "SIGLA",
        "TIPS",
        "USER",
        "WHAT",
    }
    acronyms: list[str] = []
    for token in re.findall(r"\b[A-Za-z0-9/]{2,10}\b", query):
        upper = token.upper()
        if upper in stop:
            continue
        mixed_case_acronym = len(token) <= 6 and sum(1 for ch in token if ch.isupper()) >= 2
        if token.isupper() or mixed_case_acronym or upper in known or upper in ACRONYM_DEFINITION_OVERRIDES:
            if upper not in acronyms:
                acronyms.append(upper)
    return acronyms


def is_acronym_intent(query: str) -> bool:
    low = query.lower()
    return any(
        term in low
        for term in [
            "que es",
            "qué es",
            "quÃ© es",
            "what is",
            "define",
            "definition",
            "acronym",
            "acronimo",
            "acrónimo",
            "sigla",
            "significa",
            "meaning",
            "stands for",
        ]
    )


def family_query_boosts(query: str) -> dict[str, float]:
    low = query.lower()
    boosts: dict[str, float] = {}
    for family, hints in FAMILY_HINTS.items():
        if any(hint in low for hint in hints):
            boosts[family] = 0.12
    return boosts


def acronym_bonus(chunk: dict[str, Any], query: str) -> float:
    acronyms = query_acronyms(query)
    if not acronyms:
        return 0.0
    definition_query = is_acronym_question(query)
    hay = " ".join(
        [
            chunk.get("title", ""),
            chunk.get("family", ""),
            chunk.get("category", ""),
            chunk.get("release", ""),
            str(chunk.get("unit", "")),
            " ".join(chunk.get("context_path", [])),
            chunk.get("text", "")[:1800],
        ]
    )
    bonus = 0.0
    for acronym in acronyms:
        if re.search(rf"\b{re.escape(acronym)}\b", hay):
            bonus += 0.16 if definition_query else 0.035
        if definition_query and re.search(rf"\b{re.escape(acronym)}\b\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)?", hay):
            bonus += 0.18
        if definition_query and "list of acronyms" in hay:
            bonus += 0.08
    return min(bonus, 0.28 if definition_query else 0.07)


def is_acronym_question(query: str) -> bool:
    return bool(query_acronyms(query)) and any(
        term in query.lower() for term in ["que es", "qué es", "what is", "acronym", "significa", "meaning"]
    )


def is_acronym_question(query: str) -> bool:
    return bool(query_acronyms(query)) and is_acronym_intent(query)


def is_change_request_question(query: str) -> bool:
    low = f" {query.lower()} "
    has_cr_language = any(
        term in low
        for term in [
            " change request",
            " change requests",
            " cr ",
            " crs ",
            " publicad",
            " published",
            " resumen ",
            " summary ",
            " summarize ",
        ]
    )
    has_code = bool(re.search(r"\b(?:TIPS[- .]?)?\d{4}\b", query, re.I))
    has_named_tips_code = bool(re.search(r"\bTIPS[- .]?\d{4}\b", query, re.I))
    return (has_cr_language and (has_code or "change request" in low or " cr" in low)) or has_named_tips_code


def is_change_request_summary_question(query: str) -> bool:
    low = query.lower()
    return is_change_request_question(query) and any(
        term in low
        for term in [
            "resumen",
            "resume",
            "summary",
            "summarize",
            "sintesis",
            "síntesis",
            "explica",
            "explain",
            "de que va",
            "de qué va",
        ]
    )


def is_olo_parties_question(query: str) -> bool:
    low = query.lower()
    if is_olo_messages_question(query):
        return False
    return "olo" in low and any(
        term in low
        for term in [
            "parties",
            "party",
            "actores",
            "roles",
            "participantes",
            "intervienen",
            "involved",
            "quien",
            "quién",
            "transaccion",
            "transacción",
            "transaction",
        ]
    )


def is_olo_messages_question(query: str) -> bool:
    low = query.lower()
    if not is_olo_question(query):
        return False
    return any(
        term in low
        for term in [
            "mensaje",
            "mensajes",
            "message",
            "messages",
            "schema",
            "schemas",
            "mystandards",
            "intervienen",
            "involved messages",
        ]
    )


def is_olo_question(query: str) -> bool:
    low = query.lower()
    return bool(re.search(r"\bolo\b", query, re.I)) or "one-leg out" in low or "one leg out" in low


def is_olo_lkt_comparison_question(query: str) -> bool:
    low = query.lower()
    mentions_olo = is_olo_question(query)
    mentions_lkt = bool(re.search(r"\blkt\b", query, re.I)) or "linked transaction" in low
    asks_comparison = any(
        term in low
        for term in [
            "diferencia",
            "diferencias",
            "diferente",
            "mismo",
            "misma",
            "igual",
            "versus",
            " vs ",
            "compare",
            "comparison",
            "difference",
            "different",
            "same",
        ]
    )
    return mentions_olo and mentions_lkt and asks_comparison


def is_lkt_question(query: str) -> bool:
    low = query.lower()
    mentions_lkt = (
        bool(re.search(r"\blkt\b", query, re.I))
        or "linked transaction" in low
        or "linked payment message" in low
        or "linked payment" in low
    )
    asks_about_model = is_acronym_intent(query) or any(
        term in low
        for term in [
            "aplica",
            "apply",
            "atom",
            "configur",
            "model",
            "modelo",
            "settlement",
            "settle",
            "liquida",
            "liquidacion",
            "liquidación",
            "cross-currency",
            "que es",
            "what is",
            "define",
            "detall",
            "enlaz",
            "explica",
            "significa",
            "diferencia",
            "difference",
            "compar",
            "mismo",
            "same",
            "flujo",
            "flow",
            "leg",
            "paso",
            "pair",
            "par",
            "mapping",
            "step",
            "usa",
            "use",
        ]
    )
    return mentions_lkt and asks_about_model


def is_investigation_offset_question(query: str) -> bool:
    low = query.lower()
    mentions_investigation = "investigation" in low or "investigacion" in low or "investigación" in low
    mentions_offset = "offset" in low or "desfase" in low or "margen" in low
    return mentions_investigation and mentions_offset


def is_currency_question(query: str) -> bool:
    low = query.lower()
    if is_source_currency_question(query):
        return False
    mentions_tips = any(
        term in low
        for term in ["tips", "tipa", "target instant payment", "cross-currency", "olo", "lkt"]
    )
    mentioned_codes = set(re.findall(r"\b[A-Z]{3}\b", query.upper())) & CURRENCY_CODES
    asks_currency = any(
        term in low
        for term in [
            "ccy",
            "currencies",
            "currency",
            "divisa",
            "divisas",
            "moneda",
            "monedas",
            "soporta",
            "support",
        ]
    ) or bool(mentioned_codes)
    return mentions_tips and asks_currency


def is_domain_query(query: str) -> bool:
    low = normalize_query(query).lower()
    if query_acronyms(query) or query_message_codes(query) or is_change_request_question(query):
        return True
    domain_terms = {
        "tips",
        "target instant payment",
        "olo",
        "one-leg",
        "one leg",
        "lkt",
        "linked transaction",
        "oct inst",
        "sctinst",
        "cross-currency",
        "cross-currency flag",
        "currency",
        "currencies",
        "divisa",
        "divisas",
        "moneda",
        "monedas",
        "participant",
        "participante",
        "party",
        "parties",
        "leg exit",
        "leg entry",
        "source leg",
        "destination leg",
        "source currency",
        "destination currency",
        "investigation",
        "investigacion",
        "investigación",
        "recall",
        "authorised account user",
        "authorized account user",
        "mapping table",
        "linked payment",
        "pacs",
        "camt",
        "acmt",
        "reda",
        "schema",
        "xsd",
        "mystandards",
        "udfs",
        "uhb",
        "urd",
        "crdm",
        "liquidity",
        "liquidez",
        "settlement",
        "liquidacion",
    }
    return any(term in low for term in domain_terms)


def is_source_currency_question(query: str) -> bool:
    low = query.lower()
    mentions_context = any(
        term in low
        for term in ["tips", "tipa", "target instant payment", "olo", "one-leg", "one leg", "cross-currency", "lkt"]
    )
    mentions_source_or_destination = any(
        term in low
        for term in [
            "source currency",
            "destination currency",
            "source leg",
            "destination leg",
            "origin currency",
            "originator leg",
            "moneda origen",
            "divisa origen",
            "moneda destino",
            "divisa destino",
        ]
    )
    bare_leg_definition = bool(re.search(r"\b(source|destination)\s+leg\b", low))
    return mentions_source_or_destination and (mentions_context or bare_leg_definition)


def is_uhb_u2a_question(query: str) -> bool:
    low = normalize_query(query).lower()
    mentions_tips = any(term in low for term in ["tips", "target instant payment", "tipa"])
    mentions_uhb = any(term in low for term in ["uhb", "user handbook", "handbook", "manual"])
    mentions_u2a = any(
        term in low
        for term in [
            "u2a",
            "user-to-application",
            "user to application",
            "gui",
            "graphical user interface",
            "interfaz",
            "pantalla",
        ]
    )
    asks_availability = any(
        term in low
        for term in [
            "hay",
            "existe",
            "tiene",
            "available",
            "availability",
            "soporta",
            "support",
            "o no",
            "is there",
            "does",
        ]
    )
    return mentions_tips and mentions_u2a and (mentions_uhb or asks_availability)


def is_u2a_functions_question(query: str) -> bool:
    low = normalize_query(query).lower()
    mentions_tips = any(term in low for term in ["tips", "target instant payment", "tipa"])
    mentions_u2a = any(
        term in low
        for term in [
            "u2a",
            "user-to-application",
            "user to application",
            "gui",
            "graphical user interface",
            "interfaz",
            "pantalla",
        ]
    )
    asks_scope = any(
        term in low
        for term in [
            "que",
            "qué",
            "cuales",
            "cuáles",
            "what",
            "which",
            "hacer",
            "pueden",
            "can",
            "funcion",
            "funciones",
            "function",
            "functions",
            "operacion",
            "operaciones",
            "transaction",
            "transactions",
            "transaccion",
            "transacciones",
            "available",
            "disponible",
            "disponibles",
        ]
    )
    return mentions_tips and mentions_u2a and asks_scope


def is_investigation_question(query: str) -> bool:
    low = query.lower()
    mentions = "investigation" in low or "investigacion" in low or "investigación" in low
    return mentions and not is_investigation_offset_question(query)


def is_recall_question(query: str) -> bool:
    return bool(re.search(r"\brecall\b", query, re.I))


def is_cross_currency_flag_question(query: str) -> bool:
    low = query.lower()
    return "cross-currency flag" in low or ("cross currency flag" in low)


def is_authorised_account_user_question(query: str) -> bool:
    low = query.lower()
    return "authorised account user" in low or "authorized account user" in low


def message_code_bonus(chunk: dict[str, Any], query: str) -> float:
    codes = query_message_codes(query)
    if not codes:
        return 0.0
    meta_hay = " ".join(
        [
            chunk.get("title", ""),
            chunk.get("family", ""),
            chunk.get("category", ""),
            chunk.get("release", ""),
            str(chunk.get("unit", "")),
            " ".join(chunk.get("context_path", [])),
        ]
    ).lower()
    text_hay = chunk.get("text", "")[:1000].lower()
    bonus = 0.0
    for code in codes:
        prefix, number = code.split(".", 1)
        variants = [code, f"{prefix}_{number}", f"{prefix} {number}"]
        if any(variant in meta_hay for variant in variants):
            bonus += 0.11
        elif any(variant in text_hay for variant in variants):
            bonus += 0.04
    low = query.lower()
    wants_schema = any(term in low for term in ["schema", "schemas", "xsd", "xml", "mystandards", "my standards", "usage guideline"])
    if chunk.get("family") == "mystandards_udfs" and wants_schema:
        bonus += 0.14
    elif chunk.get("family") == "mystandards_udfs" and query_release(query):
        bonus += 0.06
    return min(bonus, 0.28)


def recency_boost(release: str) -> float:
    if not release:
        return 0.0
    match = RELEASE_RE.search(release)
    if not match:
        return 0.0
    year = int(match.group(1))
    month_weight = {"NOV": 0.03, "OCT": 0.02, "JUN": 0.01, "MAR": 0.0}.get(match.group(2).upper(), 0.0)
    return max(0.0, min(0.08, (year - 2023) * 0.015 + month_weight))


def context_bonus(chunk: dict[str, Any], query: str) -> float:
    low = query.lower()
    hay = " ".join(
        [
            chunk.get("title", ""),
            chunk.get("family", ""),
            chunk.get("category", ""),
            chunk.get("release", ""),
            " ".join(chunk.get("context_path", [])),
        ]
    ).lower()
    bonus = 0.0
    for token in re.findall(r"[a-zA-Z0-9._-]{3,}", low):
        if token in hay:
            bonus += 0.008
    if chunk.get("revision_status") == "clean":
        bonus += 0.015
    return min(bonus, 0.08)


def functional_olo_bonus(chunk: dict[str, Any], query: str) -> float:
    if not is_olo_parties_question(query):
        return 0.0
    text = chunk.get("text", "")
    hay = " ".join(
        [
            chunk.get("title", ""),
            chunk.get("family", ""),
            chunk.get("release", ""),
            " ".join(chunk.get("context_path", [])),
            text[:2600],
        ]
    ).lower()
    bonus = 0.0
    if "list of acronyms" in hay or "list of acronyms item description" in hay:
        bonus -= 0.30
    if "the involved actors are" in hay:
        bonus += 0.38
    if "originator participant" in hay and "beneficiary participant" in hay:
        bonus += 0.22
    if "leg exit psp acts as beneficiary" in hay:
        bonus += 0.24
    if "leg entry psp acts as originator" in hay:
        bonus += 0.24
    if "source leg" in hay and "destination leg" in hay:
        bonus += 0.12
    if "incoming olo cross-currency" in hay or "outgoing olo cross-currency" in hay:
        bonus += 0.14
    if "corridor shows" in hay and "olo" in hay:
        bonus += 0.08
    return bonus


def structured_data_bonus(chunk: dict[str, Any], query: str) -> float:
    family = chunk.get("family", "")
    text = " ".join([chunk.get("title", ""), str(chunk.get("unit", "")), chunk.get("text", "")[:1200]]).lower()
    low = query.lower()
    bonus = 0.0
    if is_change_request_question(query) and family == "change_requests":
        bonus += 0.32
        desired_release = query_release(query)
        if desired_release and chunk.get("release") == desired_release:
            bonus += 0.22
        for code in re.findall(r"\bTIPS-\d{4}\b", query, re.I):
            if code.lower() in text:
                bonus += 0.35
        if any(term in low for term in ["publicad", "published", "status"]) and "published on ecb change request page" in text:
            bonus += 0.16
    if is_acronym_question(query) and family == "acronyms":
        bonus += 0.42
        for acronym in query_acronyms(query):
            if f"acronym: {acronym.lower()}" in text:
                bonus += 0.50
    if is_currency_question(query):
        if "tips settles instant payments in euro" in text and "swedish kronor" in text and "danish kroner" in text:
            bonus += 0.60
        if "currency supported in tips" in text and "eur, sek, dkk" in text:
            bonus += 0.46
        if "currently defined in the system" in text and "eur, sek" in text and "dkk" in text:
            bonus += 0.42
    return bonus


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    max_value = float(values.max())
    if max_value <= 0:
        return np.zeros_like(values, dtype=float)
    return values / max_value


def bm25_scores(index: dict[str, Any], query: str) -> np.ndarray:
    chunks = index.get("chunks") or []
    n_chunks = len(chunks)
    if not n_chunks or "bm25_vectorizer" not in index or "bm25_matrix" not in index:
        return np.zeros(n_chunks, dtype=float)

    query_vec = index["bm25_vectorizer"].transform([query])
    term_indices = query_vec.indices
    if not len(term_indices):
        return np.zeros(n_chunks, dtype=float)
    if len(term_indices) > 48:
        order = np.argsort(query_vec.data)[::-1][:48]
        term_indices = term_indices[order]

    matrix = index["bm25_matrix"]
    idf = np.asarray(index["bm25_idf"], dtype=float)
    doc_len = np.asarray(index["bm25_doc_len"], dtype=float)
    avgdl = float(index.get("bm25_avgdl") or doc_len.mean() or 1.0)
    length_norm = 1.0 - 0.75 + 0.75 * (doc_len / avgdl)
    k1 = 1.35
    tf = matrix[:, term_indices].toarray().astype(float)
    if not tf.any():
        return np.zeros(n_chunks, dtype=float)
    denom = tf + (k1 * length_norm[:, None])
    term_scores = (tf * (k1 + 1.0)) / np.where(denom == 0, 1.0, denom)
    return (term_scores * idf[term_indices]).sum(axis=1)


def important_query_terms(query: str) -> list[str]:
    stop = {item.lower() for item in QUESTION_STOPWORDS} | {
        "tips",
        "target",
        "instant",
        "payment",
        "settlement",
        "what",
        "which",
        "define",
        "dime",
        "dame",
        "explica",
    }
    terms: list[str] = []
    for token in re.findall(r"[\w./-]{3,}", query.lower(), flags=re.UNICODE):
        if token in stop or token.isdigit():
            continue
        if token not in terms:
            terms.append(token)
    return terms[:32]


def local_rerank_bonus(chunk: dict[str, Any], query: str, expanded_query: str) -> float:
    text = re.sub(
        r"\s+",
        " ",
        " ".join(
            [
                chunk.get("title", ""),
                chunk.get("family", ""),
                chunk.get("category", ""),
                chunk.get("release", ""),
                " ".join(chunk.get("context_path", [])),
                chunk.get("text", "")[:3200],
            ]
        ),
    ).lower()
    query_terms = important_query_terms(expanded_query)
    if not query_terms:
        return 0.0

    present = [term for term in query_terms if term in text]
    coverage = len(present) / max(1, len(query_terms))
    bonus = 0.22 * coverage

    raw_low = query.lower()
    for phrase in re.findall(r"[A-Za-z0-9][A-Za-z0-9 ./-]{5,}", raw_low):
        phrase = re.sub(r"\s+", " ", phrase.strip())
        if len(phrase.split()) >= 2 and phrase in text:
            bonus += 0.08

    for code in query_message_codes(query):
        if code in text:
            bonus += 0.25
    for acronym in query_acronyms(query):
        if re.search(rf"\b{re.escape(acronym.lower())}\b", text):
            bonus += 0.18

    if chunk.get("revision_status") == "clean":
        bonus += 0.025
    if chunk.get("family") in {"tips_udfs", "mystandards_udfs", "change_requests", "acronyms"}:
        bonus += 0.025
    return min(bonus, 0.65)


def retrieve(index: dict[str, Any], query: str, top_k: int = 8, pool: int = 160) -> list[Hit]:
    chunks: list[dict[str, Any]] = index["chunks"]
    if not chunks:
        return []

    expanded_query = normalize_query(query)
    query_variants: list[str] = []
    for candidate in [query, expanded_query]:
        candidate = candidate.strip()
        if candidate and candidate not in query_variants:
            query_variants.append(candidate)

    variant_scores: list[np.ndarray] = []
    for candidate in query_variants:
        word_q = index["word_vectorizer"].transform([candidate])
        char_q = index["char_vectorizer"].transform([candidate])
        word_scores = linear_kernel(word_q, index["word_matrix"]).ravel()
        char_scores = linear_kernel(char_q, index["char_matrix"]).ravel()
        bm25 = bm25_scores(index, candidate)
        variant_scores.append(
            (0.43 * normalize_scores(word_scores))
            + (0.22 * normalize_scores(char_scores))
            + (0.35 * normalize_scores(bm25))
        )
    scores = np.max(np.vstack(variant_scores), axis=0)

    rerank_pool = min(len(chunks), max(pool * 6, top_k * 24, 800))
    candidate_idx = np.argsort(scores)[::-1][:rerank_pool]

    family_boosts = family_query_boosts(expanded_query)
    desired_release = query_release(expanded_query)
    for idx in candidate_idx:
        chunk = chunks[int(idx)]
        score = float(scores[idx])
        family = chunk.get("family", "")
        release = chunk.get("release", "")
        score += family_boosts.get(family, 0.0)
        if desired_release and desired_release == release:
            score += 0.18
        elif desired_release and release and desired_release != release:
            score -= 0.05
        elif not desired_release:
            score += recency_boost(release)
        score += message_code_bonus(chunk, expanded_query)
        score += acronym_bonus(chunk, query)
        score += functional_olo_bonus(chunk, query)
        score += structured_data_bonus(chunk, query)
        score += context_bonus(chunk, expanded_query)
        score += local_rerank_bonus(chunk, query, expanded_query)
        scores[idx] = score

    candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]][: max(pool, top_k)]
    selected: list[Hit] = []
    seen_doc_units: set[tuple[str, str, str]] = set()
    per_doc_count: dict[str, int] = {}

    for raw_idx in candidate_idx:
        chunk = chunks[int(raw_idx)]
        score = float(scores[int(raw_idx)])
        if score <= 0:
            continue
        doc_id = chunk.get("doc_id", "")
        unit_key = (doc_id, str(chunk.get("unit_type", "")), str(chunk.get("unit", "")))
        if unit_key in seen_doc_units:
            continue
        if per_doc_count.get(doc_id, 0) >= 3 and len(selected) >= 4:
            continue
        seen_doc_units.add(unit_key)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
        selected.append(Hit(rank=len(selected) + 1, score=score, chunk=chunk, reason="hybrid"))
        if len(selected) >= top_k:
            break
    return selected


def augment_hits(index: dict[str, Any], query: str, hits: list[Hit]) -> list[Hit]:
    low = query.lower()
    chunks: list[dict[str, Any]] = index["chunks"]
    supplements: list[Hit] = []

    if is_acronym_question(query):
        for acronym in query_acronyms(query):
            wanted = f"acr-index:{acronym}"
            for chunk in chunks:
                if chunk.get("chunk_id") == wanted:
                    supplements.append(Hit(rank=0, score=1.25, chunk=chunk))
                    break

    if is_change_request_question(query):
        desired_release = query_release(query)
        wanted_codes = {code.upper() for code in query_cr_codes(query)}
        for chunk in chunks:
            if chunk.get("family") != "change_requests":
                continue
            unit = str(chunk.get("unit", "")).upper()
            if unit in wanted_codes or any(unit.startswith(f"{code}-") for code in wanted_codes) or (
                desired_release and chunk.get("release") == desired_release and chunk.get("unit_type") == "change_request"
            ):
                supplements.append(Hit(rank=0, score=1.18, chunk=chunk))
        if wanted_codes:
            summary_needles = [
                "reason for change",
                "description of requested change",
                "enhanced linked transaction",
                "all transactions are settled or none",
                "instant payment transaction steps for cross-currency",
                "both currencies",
                "mapping table",
                "advanced cross-currency payment transaction query",
                "summary of application development impact",
            ]
            for needle in summary_needles:
                matches = []
                for chunk in chunks:
                    if chunk.get("family") != "change_requests":
                        continue
                    hay = " ".join([chunk.get("title", ""), str(chunk.get("unit", "")), chunk.get("text", "")])
                    hay_upper = hay.upper()
                    if not any(code in hay_upper for code in wanted_codes):
                        continue
                    if needle in hay.lower():
                        matches.append(chunk)
                if matches:
                    matches.sort(
                        key=lambda chunk: (
                            chunk.get("revision_status") == "clean",
                            chunk.get("release") or "",
                            -len(str(chunk.get("text", ""))),
                        ),
                        reverse=True,
                    )
                    supplements.append(Hit(rank=0, score=1.16, chunk=matches[0]))
        for chunk in chunks:
            title = (chunk.get("title") or "").lower()
            text = chunk.get("text", "").lower()
            if desired_release and chunk.get("release") == desired_release and (
                "cover note" in title or "main milestones" in title or "content of tips release" in title
            ):
                supplements.append(Hit(rank=0, score=1.05, chunk=chunk))
            elif not desired_release and ("list of tips change requests" in title or "structured tips change request catalogue" in title):
                supplements.append(Hit(rank=0, score=1.02, chunk=chunk))

    if "olo" in low:
        needle_sets = [
            ("cross-currency (one-leg out)", "one-leg out instant credit transfer"),
            ("one-leg out instant credit transfer", "designed and implemented in tips"),
            ("cross-currency settlement service", "one-leg out instant credit transfer"),
            ("cross-currency payment", "the involved actors are", "source leg", "destination leg", "originator participant", "beneficiary participant"),
            ("leg exit psp acts as beneficiary", "leg entry psp acts as originator"),
            ("olo one-leg out",),
            ("incoming olo cross-currency", "outgoing olo cross-currency"),
            ("corridor shows", "olo transactions"),
        ]
        for needles in needle_sets:
            matches: list[dict[str, Any]] = []
            for chunk in chunks:
                text = " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        chunk.get("family") == "tips_udfs",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                chunk = matches[0]
                supplements.append(Hit(rank=0, score=0.99, chunk=chunk))

    if is_lkt_question(query) or is_olo_lkt_comparison_question(query):
        needle_sets = [
            (1.28, ("cross-currency linked-transactions", "preferred settlement model")),
            (1.22, ("linked payment message", "settle simultaneously")),
            (1.20, ("enhanced lkt settlement model", "all transactions are settled or none")),
            (1.16, ("both currencies", "pair is configured in the mapping table")),
        ]
        for score, needles in needle_sets:
            matches = []
            for chunk in chunks:
                text = re.sub(
                    r"\s+",
                    " ",
                    " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]),
                ).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        chunk.get("family") == "change_requests",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    if is_investigation_offset_question(query):
        needle_sets = [
            (1.26, ("investigation offset", "configurable offset foreseen in sct inst scheme", "5,000")),
            (1.20, ("investigation offset", "non-euro currency", "can hold a negative value")),
            (1.18, ("answers to an investigation request only if", "expired for more than 5 seconds")),
            (1.16, ("investigation request has been received after", "sctinst timestamp timeout + investigation offset")),
        ]
        for score, needles in needle_sets:
            matches = []
            for chunk in chunks:
                text = re.sub(
                    r"\s+",
                    " ",
                    " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]),
                ).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        chunk.get("family") == "tips_udfs",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    if is_currency_question(query):
        needle_sets = [
            (1.24, ("tips settles instant payments in euro", "swedish kronor", "danish kroner")),
            (1.16, ("for each currency supported in tips", "eur, sek, dkk")),
            (1.13, ("currently defined in the system", "eur, sek", "dkk")),
            (1.08, ("dkk among the active settlement currencies", "eur, sek")),
            (1.02, ("norges bank", "nok business date")),
        ]
        for score, needles in needle_sets:
            matches = []
            for chunk in chunks:
                text = " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        "onboarding" in str(chunk.get("title", "")).lower(),
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    if is_uhb_u2a_question(query) or is_u2a_functions_question(query):
        needle_sets = [
            (1.34, ("the gui is a browser-based application", "communication with tips in u2a mode")),
            (1.33, ("the complete list of functions available 24/7/365 via the tips gui", "payment transaction advanced status query")),
            (1.32, ("increase/decrease of a cmb limit", "task management task list", "enter liquidity transfer order")),
            (1.26, ("gui menu is structured into two hierarchical menu levels", "payment transaction", "audit trail")),
            (1.18, ("audit trail", "reference and transactional objects", "available only in u2a mode")),
            (1.28, ("target instant payment settlement user handbook", "uhb user handbook", "u2a user-to-application")),
            (1.22, ("each interaction with tips", "a2a or u2a mode", "message or a gui screen")),
            (1.16, ("tips platform can be accessed", "a2a mode and u2a mode")),
            (1.1, ("this function is available in u2a mode only", "screen access")),
        ]
        for score, needles in needle_sets:
            matches = []
            for chunk in chunks:
                text = re.sub(
                    r"\s+",
                    " ",
                    " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]),
                ).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        chunk.get("family") == "tips_uhb",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    if is_source_currency_question(query):
        needle_sets = [
            (
                1.24,
                (
                    "entire end-to-end cross-currency instant payment transaction flow is composed of two separate and independent instant payments",
                    "currencies of the source and destination leg",
                ),
            ),
            (1.18, ("originator participant", "starting the cross-currency payment in the source leg")),
            (
                1.13,
                (
                    "interbank settlement amount",
                    "destination currency",
                    "instructed amount",
                    "source currency",
                ),
            ),
            (1.08, ("from tips (source currency)", "tips (destination currency)")),
        ]
        for score, needles in needle_sets:
            matches = []
            for chunk in chunks:
                text = re.sub(
                    r"\s+",
                    " ",
                    " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]),
                ).lower()
                if all(needle in text for needle in needles):
                    matches.append(chunk)
            if matches:
                matches.sort(
                    key=lambda chunk: (
                        chunk.get("release") == "R2026.NOV",
                        chunk.get("revision_status") == "clean",
                        chunk.get("family") == "tips_udfs",
                        -len(str(chunk.get("text", ""))),
                    ),
                    reverse=True,
                )
                supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    targeted_needle_sets: list[tuple[float, tuple[str, ...]]] = []
    if is_olo_messages_question(query):
        targeted_needle_sets.extend(
            [
                (1.24, ("instant payment transaction steps for cross-currency (one-leg out)", "fitoficustomercredittransfer")),
                (1.18, ("list of messages for cross-currency model", "pacs.008.001.08")),
                (1.08, ("fitofipaymentcancellationrequest", "camt.056.001.08", "recall")),
            ]
        )
    if is_investigation_question(query):
        targeted_needle_sets.extend(
            [
                (1.24, ("2.4. investigation", "processing of an investigation request")),
                (1.20, ("fitofipaymentstatusrequest", "pacs.028.001.03", "fitofipaymentstatusreport", "pacs.002.001.10")),
                (1.16, ("payment transaction existence", "investigation allowed")),
            ]
        )
    if is_recall_question(query):
        targeted_needle_sets.extend(
            [
                (1.24, ("2.3. recall", "request the cancellation", "return of funds")),
                (1.18, ("fitofipaymentcancellationrequest", "paymentreturn", "resolutionofinvestigation")),
                (1.14, ("recall steps", "camt.056.001.08")),
            ]
        )
    if is_cross_currency_flag_question(query):
        targeted_needle_sets.extend(
            [
                (1.22, ("cross-currency flag", "incoming olo cross-currency")),
                (1.18, ("cross-currency flag", "outgoing olo cross-currency")),
                (1.12, ("authorised account user bic", "authorised to accept cross-currency payments")),
            ]
        )
    if is_authorised_account_user_question(query):
        targeted_needle_sets.extend(
            [
                (1.22, ("record type", "authorised account user", "cash account")),
                (1.16, ("reachable parties defined as authorised account users")),
                (1.12, ("authorised account user bic", "cash account")),
            ]
        )
    for score, needles in targeted_needle_sets:
        matches = []
        for chunk in chunks:
            text = re.sub(
                r"\s+",
                " ",
                " ".join([chunk.get("title", ""), chunk.get("release", ""), chunk.get("text", "")]),
            ).lower()
            if all(needle in text for needle in needles):
                matches.append(chunk)
        if matches:
            matches.sort(
                key=lambda chunk: (
                    chunk.get("release") == "R2026.NOV",
                    chunk.get("revision_status") == "clean",
                    chunk.get("family") == "tips_udfs",
                    chunk.get("family") == "change_requests",
                    -len(str(chunk.get("text", ""))),
                ),
                reverse=True,
            )
            supplements.append(Hit(rank=0, score=score, chunk=matches[0]))

    if not supplements:
        return hits

    combined = []
    seen: set[str] = set()
    for hit in [*supplements, *hits]:
        chunk_id = hit.chunk.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        combined.append(hit)
    for rank, hit in enumerate(combined, start=1):
        hit.rank = rank
    return combined


def chunk_position_map(index: dict[str, Any]) -> dict[str, int]:
    mapping = index.get("chunk_id_to_pos")
    if isinstance(mapping, dict) and mapping:
        return {str(key): int(value) for key, value in mapping.items()}
    return {
        str(chunk.get("chunk_id")): idx
        for idx, chunk in enumerate(index.get("chunks", []))
        if chunk.get("chunk_id")
    }


def expand_neighbor_hits(index: dict[str, Any], hits: list[Hit], max_neighbors: int = 1, max_total: int = 24) -> list[Hit]:
    if not hits:
        return hits
    chunks: list[dict[str, Any]] = index.get("chunks", [])
    positions = chunk_position_map(index)
    combined: list[Hit] = []
    seen: set[str] = set()

    def add(hit: Hit) -> None:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        if chunk_id in seen or len(combined) >= max_total:
            return
        seen.add(chunk_id)
        combined.append(hit)

    for hit in hits:
        add(hit)
        chunk_id = str(hit.chunk.get("chunk_id") or "")
        pos = positions.get(chunk_id)
        if pos is None:
            continue
        for distance in range(1, max_neighbors + 1):
            for neighbor_pos in (pos - distance, pos + distance):
                if neighbor_pos < 0 or neighbor_pos >= len(chunks):
                    continue
                neighbor = chunks[neighbor_pos]
                if neighbor.get("doc_id") != hit.chunk.get("doc_id"):
                    continue
                if neighbor.get("unit_type") != hit.chunk.get("unit_type") or str(neighbor.get("unit")) != str(hit.chunk.get("unit")):
                    continue
                add(Hit(rank=0, score=max(hit.score * (0.82 - distance * 0.08), 0.01), chunk=neighbor, reason="neighbor"))

    combined.sort(
        key=lambda item: (
            item.reason != "neighbor",
            item.score,
            item.chunk.get("revision_status") == "clean",
        ),
        reverse=True,
    )
    for rank, hit in enumerate(combined, start=1):
        hit.rank = rank
    return combined


def _chunk_text_for_ranking(chunk: dict[str, Any], max_text: int = 8000) -> str:
    return re.sub(
        r"\s+",
        " ",
        " ".join(
            [
                str(chunk.get("title") or ""),
                str(chunk.get("release") or ""),
                str(chunk.get("unit_type") or ""),
                str(chunk.get("unit") or ""),
                " ".join(str(item) for item in chunk.get("context_path") or []),
                str(chunk.get("text") or "")[:max_text],
            ]
        ),
    )


def _cr_code_aliases(code: str) -> set[str]:
    clean = clean_cr_code(code)
    aliases = {clean}
    match = re.match(r"^(TIPS-\d{4})(?:-(URD|SYS))?$", clean)
    if match:
        aliases.add(match.group(1))
    return aliases


def context_pointer_priority(query: str, hit: Hit) -> tuple[float, float]:
    chunk = hit.chunk
    hay = _chunk_text_for_ranking(chunk)
    hay_low = hay.lower()
    hay_upper = hay.upper()
    priority = 0.0

    if chunk.get("revision_status") == "clean":
        priority += 0.2
    if chunk.get("family") in {"tips_udfs", "mystandards_udfs", "change_requests", "acronyms"}:
        priority += 0.12
    if hit.reason == "neighbor":
        priority -= 0.12

    release = query_release(query)
    if release:
        priority += 3.0 if chunk.get("release") == release else -0.25

    if is_change_request_question(query):
        wanted_codes = query_cr_codes(query)
        if wanted_codes:
            aliases = set().union(*(_cr_code_aliases(code) for code in wanted_codes))
            title_unit_upper = " ".join(
                [
                    str(chunk.get("title") or ""),
                    str(chunk.get("unit_type") or ""),
                    str(chunk.get("unit") or ""),
                    str(chunk.get("local_path") or ""),
                ]
            ).upper()
            direct_code_match = any(alias in title_unit_upper for alias in aliases)
            related_code_match = any(alias in hay_upper for alias in aliases)
            unit = str(chunk.get("unit") or "").upper()
            if direct_code_match:
                priority += 24.0
            elif related_code_match:
                priority += 8.0
            elif chunk.get("family") == "change_requests":
                priority -= 7.0
            if unit in aliases or any(unit.startswith(f"{alias}-") for alias in aliases):
                priority += 6.0
            if chunk.get("unit_type") == "change_request":
                priority += 3.0
            for phrase, bonus in [
                ("reason for change", 2.8),
                ("expected benefits", 2.5),
                ("business motivation", 2.5),
                ("description of requested change", 2.8),
                ("description of requested changes", 2.8),
                ("detailed assessment", 2.0),
                ("instant payment transaction steps", 2.0),
                ("summary of application development impact", 2.0),
                ("mapping table", 1.5),
                ("all transactions are settled or none", 1.5),
                ("advanced cross-currency payment transaction query", 1.3),
                ("out-of-scope", 1.0),
            ]:
                if phrase in hay_low:
                    priority += bonus
        elif release and chunk.get("family") == "release_documentation":
            title = str(chunk.get("title") or "").lower()
            if "cover note" in title or "content of tips release" in title or "main milestones" in title:
                priority += 5.0

    for code in query_message_codes(query):
        if code in hay_low:
            priority += 5.0

    for acronym in query_acronyms(query):
        if re.search(rf"\b{re.escape(acronym)}\b", hay_upper):
            priority += 4.0
        if chunk.get("chunk_id") == f"acr-index:{acronym}":
            priority += 8.0

    topic_phrases: list[tuple[bool, list[tuple[str, float]]]] = [
        (
            is_olo_question(query),
            [
                ("one-leg out instant credit transfer", 5.0),
                ("cross-currency (one-leg out)", 4.0),
                ("leg exit psp acts as beneficiary", 3.5),
                ("leg entry psp acts as originator", 3.5),
                ("incoming olo cross-currency", 3.0),
                ("outgoing olo cross-currency", 3.0),
            ],
        ),
        (
            is_lkt_question(query) or is_olo_lkt_comparison_question(query),
            [
                ("enhanced linked transaction", 5.0),
                ("linked-transactions", 4.0),
                ("linked payment message", 3.5),
                ("settle simultaneously", 3.5),
                ("all transactions are settled or none", 3.5),
                ("pair is configured in the mapping table", 3.0),
            ],
        ),
        (
            is_currency_question(query),
            [
                ("tips settles instant payments in euro", 4.5),
                ("swedish kronor", 3.5),
                ("danish kroner", 3.5),
                ("eur, sek, dkk", 3.0),
                ("active settlement currencies", 2.5),
            ],
        ),
        (
            is_investigation_question(query),
            [
                ("2.4. investigation", 4.0),
                ("processing of an investigation request", 4.0),
                ("fitofipaymentstatusrequest", 3.0),
                ("pacs.028.001.03", 3.0),
                ("investigation allowed", 2.5),
            ],
        ),
    ]
    for enabled, phrases in topic_phrases:
        if not enabled:
            continue
        for phrase, bonus in phrases:
            if phrase in hay_low:
                priority += bonus

    return priority, hit.score


def rerank_context_hits(query: str, hits: list[Hit], max_total: int | None = None) -> list[Hit]:
    if not hits:
        return hits
    allow_revisions = bool(re.search(r"\b(revision|revised|tracked|changes|cambios|revisiones)\b", query, re.I))
    ranked = sorted(
        enumerate(hits),
        key=lambda item: (*context_pointer_priority(query, item[1]), -item[0]),
        reverse=True,
    )
    selected: list[Hit] = []
    seen_chunks: set[str] = set()
    seen_texts: set[str] = set()
    for _, hit in ranked:
        title = str(hit.chunk.get("title") or "").lower()
        revision_status = str(hit.chunk.get("revision_status") or "").lower()
        if not allow_revisions and ("with revisions" in title or revision_status in {"rev", "revised"}):
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        text_key = re.sub(r"\W+", " ", str(hit.chunk.get("text") or "").lower())[:700]
        if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
            continue
        seen_chunks.add(chunk_id)
        if text_key:
            seen_texts.add(text_key)
        selected.append(hit)
        if max_total and len(selected) >= max_total:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def trim_excerpt(text: str, max_chars: int = 950) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    if last_stop > 350:
        return cut[: last_stop + 1].strip()
    return cut.rstrip() + "..."


def wants_schema_list(query: str) -> bool:
    low = query.lower()
    asks_for_list = any(term in low for term in ["que ", "qué ", "cuales", "cuáles", "lista", "list", "hay", "which"])
    schema_terms = ["mensaje", "mensajes", "message", "schema", "schemas", "esquema", "esquemas", "xsd", "mystandards"]
    return asks_for_list and (any(term in low for term in schema_terms) or bool(query_message_codes(query)))


def clean_mystandards_title(title: str) -> str:
    title = title.replace("MyStandards UDFS TIPS R2026.NOV - ", "")
    return title.strip()


def cite_label(hit: Hit) -> str:
    chunk = hit.chunk
    title = chunk.get("title") or "Untitled"
    unit_type = chunk.get("unit_type") or "unit"
    unit = chunk.get("unit")
    release = chunk.get("release")
    release_part = f", {release}" if release else ""
    if unit_type == "page":
        where = f"page {unit}"
    elif unit_type == "sheet":
        where = f"sheet {unit}"
    elif unit_type == "zip_member":
        where = f"ZIP member {unit}"
    else:
        where = f"{unit_type} {unit}"
    return f"{title}{release_part}, {where}"


def build_schema_list_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not wants_schema_list(query):
        return None
    docs: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk.get("family") != "mystandards_udfs":
            continue
        doc_id = hit.chunk.get("doc_id") or hit.chunk.get("local_path") or hit.chunk.get("title")
        if doc_id in seen:
            continue
        seen.add(str(doc_id))
        docs.append(hit)
    if not docs:
        return None

    if language == "en":
        intro = f"Found {len(docs)} local MyStandards UDFS TIPS R2026.NOV schema package(s):"
        source_title = "References"
    else:
        intro = f"He encontrado {len(docs)} paquete(s) local(es) MyStandards UDFS TIPS R2026.NOV:"
        source_title = "Referencias"

    bullets = []
    citations = []
    for n, hit in enumerate(docs[:8], start=1):
        title = clean_mystandards_title(hit.chunk.get("title") or "Untitled")
        release = hit.chunk.get("release") or ""
        unit = hit.chunk.get("unit") or ""
        bullets.append(f"{n}. {title} (mensaje {unit}, release {release}) [{n}]")
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": release,
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": unit,
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    answer = intro + "\n\n" + "\n".join(bullets)
    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{c['n']}] {c['label']} - {c['local_path']}" for c in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high"}


def extract_acronym_definition(text: str, acronym: str) -> str:
    for match in re.finditer(rf"\b{re.escape(acronym)}\b\s+(.{{1,90}})", text, flags=re.I):
        tail = match.group(1).strip(" .;:,")
        words = re.findall(r"[A-Za-z][A-Za-z-]*|[A-Z0-9]{2,8}", tail)
        definition_words: list[str] = []
        for word in words:
            if definition_words and re.fullmatch(r"[A-Z0-9]{2,8}", word):
                break
            if not word[:1].isupper():
                definition_words = []
                break
            definition_words.append(word)
            if len(definition_words) >= 5:
                break
        if definition_words:
            return " ".join(definition_words).strip()
    return ""


def citation_from_acronym_source(source: dict[str, Any], n: int) -> dict[str, Any]:
    title = source.get("title") or "Structured TIPS acronym dictionary"
    release = source.get("release") or ""
    unit_type = source.get("unit_type") or "acronym"
    unit = source.get("unit") or ""
    label = f"{title}{', ' + release if release else ''}, {unit_type} {unit}".strip()
    return {
        "n": n,
        "title": title,
        "release": release,
        "family": source.get("family") or "acronyms",
        "unit_type": unit_type,
        "unit": unit,
        "local_path": source.get("local_path") or str(ACRONYMS_PATH.relative_to(ROOT)),
        "source_url": source.get("source_url") or "",
        "score": 1.0,
        "label": label,
    }


def build_structured_acronym_answer(query: str, hits: list[Hit], language: str) -> dict[str, Any] | None:
    if not is_acronym_question(query):
        return None
    entries = acronym_entries_by_key()
    requested = query_acronyms(query)
    if not requested:
        return None
    acronym = next((item for item in requested if item in entries), requested[0])
    entry = entries.get(acronym)
    if not entry:
        return None
    definition = ACRONYM_DEFINITION_OVERRIDES.get(acronym) or entry.get("definition") or ""
    sources = sorted(
        entry.get("sources") or [],
        key=lambda source: (
            "tips" in str(source.get("title", "")).lower(),
            source.get("release") == "R2026.NOV",
            source.get("release") == "R2026.JUN",
            "with revisions" not in str(source.get("title", "")).lower(),
            source.get("release") or "",
        ),
        reverse=True,
    )
    citations: list[dict[str, Any]] = []
    for source in sources[:3]:
        citations.append(citation_from_acronym_source(source, len(citations) + 1))

    support_hit = None
    for hit in hits:
        if hit.chunk.get("family") == "acronyms":
            continue
        text = hit.chunk.get("text", "")
        if "list of acronyms" in text.lower():
            continue
        if re.search(rf"\b{re.escape(acronym)}\b", text):
            support_hit = hit
            break
    if support_hit:
        citations.append(
            {
                "n": len(citations) + 1,
                "title": support_hit.chunk.get("title"),
                "release": support_hit.chunk.get("release"),
                "family": support_hit.chunk.get("family"),
                "unit_type": support_hit.chunk.get("unit_type"),
                "unit": support_hit.chunk.get("unit"),
                "local_path": support_hit.chunk.get("local_path"),
                "source_url": support_hit.chunk.get("source_url"),
                "score": round(support_hit.score, 4),
                "label": cite_label(support_hit),
            }
        )

    if language == "en":
        answer = f"{acronym} means {definition}. [{1}]"
        if support_hit:
            answer += f" In the local TIPS corpus it also appears in {support_hit.chunk.get('title')}, so the term is not just a glossary entry; it is used in the operational/specification material. [{citations[-1]['n']}]"
        else:
            answer += " I do not find a stronger functional passage for this acronym beyond the local glossary entry."
        source_title = "References"
    else:
        answer = f"{acronym} significa {definition}. [{1}]"
        if support_hit:
            answer += f" En el corpus local también aparece en {support_hit.chunk.get('title')}; por tanto, no es solo una entrada de glosario, sino un término usado en la documentación operativa o funcional. [{citations[-1]['n']}]"
        else:
            answer += " No aparece un pasaje funcional más fuerte para este acrónimo aparte de la entrada local de glosario."
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_acronym_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_acronym_question(query):
        return None
    structured = build_structured_acronym_answer(query, hits, language)
    if structured:
        return structured
    requested = query_acronyms(query)
    if not requested:
        return None
    acronym = requested[0]
    definition_matches: list[tuple[str, Hit]] = []
    for hit in hits:
        text = re.sub(r"\s+", " ", hit.chunk.get("text", ""))
        found = extract_acronym_definition(text, acronym)
        if found:
            definition_matches.append((found, hit))
    if not definition_matches and acronym in ACRONYM_DEFINITION_OVERRIDES:
        for hit in hits:
            text = hit.chunk.get("text", "")
            if re.search(rf"\b{re.escape(acronym)}\b", text, flags=re.I) or ACRONYM_DEFINITION_OVERRIDES[acronym].lower() in text.lower():
                definition_matches.append((ACRONYM_DEFINITION_OVERRIDES[acronym], hit))
                break
    if not definition_matches:
        return None
    definition_matches.sort(
        key=lambda item: (
            item[1].chunk.get("release") == "R2026.NOV",
            item[1].chunk.get("revision_status") == "clean",
            item[1].score,
        ),
        reverse=True,
    )
    definition, definition_hit = definition_matches[0]

    support_hits = [definition_hit]
    for hit in hits:
        if hit is definition_hit:
            continue
        text = hit.chunk.get("text", "")
        if acronym in text and any(term in text.lower() for term in ["cross-currency", "corridor"]):
            support_hits.append(hit)
            break
    for hit in hits:
        if hit is definition_hit:
            continue
        text = hit.chunk.get("text", "")
        if acronym in text and len(support_hits) < 4:
            support_hits.append(hit)

    if language == "en":
        answer = f"{acronym} means {definition} in the local TIPS documentation. [{1}]"
        if acronym == "OLO" and len(support_hits) > 1:
            answer += f" In TIPS R2026.NOV it appears in cross-currency contexts, for example as a corridor value and in authorisation checks for incoming/outgoing OLO cross-currency transactions. [2]"
        source_title = "References"
    else:
        answer = f"{acronym} significa {definition} en la documentación local de TIPS. [{1}]"
        if acronym == "OLO" and len(support_hits) > 1:
            answer += f" En TIPS R2026.NOV aparece en contexto cross-currency, por ejemplo como valor de corridor y en checks de autorización para transacciones OLO entrantes y salientes. [2]"
        source_title = "Referencias"

    citations = []
    for n, hit in enumerate(support_hits, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{c['n']}] {c['label']} - {c['local_path']}" for c in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def find_hit(hits: list[Hit], *needles: str, exclude: set[int] | None = None) -> Hit | None:
    exclude = exclude or set()
    matches: list[Hit] = []
    for hit in hits:
        if id(hit) in exclude:
            continue
        text = re.sub(
            r"\s+",
            " ",
            " ".join(
                [
                    hit.chunk.get("title", ""),
                    hit.chunk.get("release", ""),
                    str(hit.chunk.get("unit", "")),
                    hit.chunk.get("text", ""),
                ]
            ),
        ).lower()
        if all(needle.lower() in text for needle in needles):
            matches.append(hit)
    if not matches:
        return None
    matches.sort(
        key=lambda hit: (
            hit.chunk.get("release") == "R2026.NOV",
            hit.chunk.get("revision_status") == "clean",
            hit.score,
        ),
        reverse=True,
    )
    return matches[0]


def citations_from_hits(support: list[Hit]) -> list[dict[str, Any]]:
    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    return citations


def ref_for_hit(citations: list[dict[str, Any]], support: list[Hit], hit: Hit | None, fallback: int = 1) -> int:
    if not hit:
        return fallback
    chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
    for item, support_hit in zip(citations, support):
        if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
            return item["n"]
    return fallback


def unique_hits(*hits: Hit | None) -> list[Hit]:
    support: list[Hit] = []
    seen: set[str] = set()
    for hit in hits:
        if not hit:
            continue
        key = "|".join(
            [
                str(hit.chunk.get("local_path") or hit.chunk.get("doc_id") or ""),
                str(hit.chunk.get("unit_type") or ""),
                str(hit.chunk.get("unit") or ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        support.append(hit)
    return support


def append_references(answer: str, citations: list[dict[str, Any]], language: str) -> str:
    source_title = "References" if language == "en" else "Referencias"
    return answer + f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )


def build_olo_messages_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_olo_messages_question(query):
        return None

    steps_hit = find_hit(hits, "instant payment transaction steps for cross-currency (one-leg out)", "fitoficustomercredittransfer")
    list_hit = find_hit(hits, "list of messages for cross-currency model", "pacs.008.001.08")
    recall_hit = find_hit(hits, "fitofipaymentcancellationrequest", "camt.056.001.08", "recall")
    support = unique_hits(steps_hit, list_hit, recall_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    steps_ref = ref_for_hit(citations, support, steps_hit)
    list_ref = ref_for_hit(citations, support, list_hit, steps_ref)
    recall_ref = ref_for_hit(citations, support, recall_hit, list_ref)

    if language == "en":
        answer = (
            f"For OLO, the core payment flow is driven by `FIToFICustomerCreditTransfer` and status-report messages in the UDFS cross-currency one-leg-out steps. [{steps_ref}] "
            f"Mapped to ISO 20022, the cross-currency message list includes `pacs.008.001.08` (`FIToFICustomerCreditTransfer`), `pacs.002.001.10` (`FIToFIPaymentStatusReport`) and `pacs.004.001.09` (`PaymentReturn`). [{list_ref}]"
        )
        if recall_hit:
            answer += f" For recall/investigation paths around the same settlement area, TIPS also uses `camt.056.001.08` (`FIToFIPaymentCancellationRequest`) and `camt.029.001.09` (`ResolutionOfInvestigation`). [{recall_ref}]"
    else:
        answer = (
            f"En OLO, el flujo de pago base se mueve con `FIToFICustomerCreditTransfer` y mensajes de estado dentro de la tabla UDFS de cross-currency one-leg-out. [{steps_ref}] "
            f"En ISO 20022, la lista cross-currency de TIPS incluye `pacs.008.001.08` (`FIToFICustomerCreditTransfer`), `pacs.002.001.10` (`FIToFIPaymentStatusReport`) y `pacs.004.001.09` (`PaymentReturn`). [{list_ref}]"
        )
        if recall_hit:
            answer += f" Para rutas de recall/investigation relacionadas, TIPS usa tambien `camt.056.001.08` (`FIToFIPaymentCancellationRequest`) y `camt.029.001.09` (`ResolutionOfInvestigation`). [{recall_ref}]"

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def build_olo_parties_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_olo_parties_question(query):
        return None

    actor_hit = find_hit(hits, "the involved actors are", "originator participant", "beneficiary participant")
    psp_hit = find_hit(hits, "leg exit psp acts as beneficiary", "leg entry psp acts as originator")
    check_hit = find_hit(hits, "incoming olo cross-currency", "outgoing olo cross-currency")
    corridor_hit = find_hit(hits, "corridor shows", "olo")

    support: list[Hit] = []
    seen: set[int] = set()
    for hit in [actor_hit, psp_hit, check_hit, corridor_hit]:
        if hit and id(hit) not in seen:
            support.append(hit)
            seen.add(id(hit))
    if len(support) < 2:
        return None

    if language == "en":
        answer = (
            "In an OLO (One-Leg Out) cross-currency transaction, the relevant TIPS roles are:\n\n"
            "1. Originator Participant, Ancillary System, or an Instructing Party acting for the Originator Participant/Reachable Party: starts the payment in the source leg. [1]\n"
            "2. Beneficiary Participant, Ancillary System, or an Instructing Party acting for the Beneficiary Participant/Reachable Party: receives the request in the destination leg and accepts or rejects it. [1]\n"
            "3. Leg Exit PSP: in the outgoing/source leg it acts as Beneficiary and is identified through the Instructed Agent BIC, or Intermediary Agent 1 BIC if applicable. [2]\n"
            "4. Leg Entry PSP: in the incoming/destination leg it acts as Originator and is identified through the Instructing Agent BIC, or the last Previous Instructing Agent BIC. [2]\n\n"
            "TIPS describes the end-to-end cross-currency flow as two separate instant payments, one in the source currency and one in the destination currency; the orchestration between the two legs is the responsibility of the external Exit/Entry-Leg PSPs. [1]"
        )
        if check_hit:
            answer += " For OLO-specific business checks, TIPS verifies the Cross-currency Flag on the Creditor Agent for incoming OLO and on the Leg Exit PSP for outgoing OLO. [3]"
        source_title = "References"
    else:
        answer = (
            "En una transacción OLO (One-Leg Out) cross-currency, TIPS distingue cuatro roles principales:\n\n"
            "1. Originator Participant, Ancillary System o Instructing Party que actúa por el Originator Participant o por una Reachable Party: inicia el pago en la source leg. [1]\n"
            "2. Beneficiary Participant, Ancillary System o Instructing Party que actúa por el Beneficiary Participant o por una Reachable Party: recibe la petición en la destination leg y la confirma o la rechaza. [1]\n"
            "3. Leg Exit PSP: en la outgoing/source leg actúa como Beneficiary y se identifica por el Instructed Agent BIC, o por el Intermediary Agent 1 BIC cuando aplica. [2]\n"
            "4. Leg Entry PSP: en la incoming/destination leg actúa como Originator y se identifica por el Instructing Agent BIC, o por el último Previous Instructing Agent BIC. [2]\n\n"
            "TIPS no describe ese flujo cross-currency como una sola pieza interna. Lo trata como dos instant payments separadas, una en la source currency y otra en la destination currency; la orquestación entre las dos legs queda bajo responsabilidad de los Exit/Entry-Leg PSPs externos. [1]"
        )
        if check_hit:
            answer += " En los controles específicos de OLO, TIPS valida el Cross-currency Flag del Creditor Agent para una incoming OLO y el del Leg Exit PSP para una outgoing OLO. [3]"
        source_title = "Referencias"

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{c['n']}] {c['label']} - {c['local_path']}" for c in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high"}


def build_olo_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_olo_question(query) or is_olo_lkt_comparison_question(query) or is_olo_parties_question(query):
        return None

    intro_hit = (
        find_hit(hits, "one-leg out instant credit transfer", "designed and implemented in tips")
        or find_hit(hits, "one-leg out instant credit transfer", "implemented in tips")
        or find_hit(hits, "cross-currency (one-leg out)")
    )
    two_leg_hit = find_hit(hits, "two separate and independent instant payments", "source and destination leg")
    actor_hit = find_hit(hits, "the involved actors are", "originator participant", "beneficiary participant")
    psp_hit = find_hit(hits, "leg exit psp acts as beneficiary", "leg entry psp acts as originator")
    check_hit = find_hit(hits, "incoming olo cross-currency", "outgoing olo cross-currency")
    corridor_hit = find_hit(hits, "corridor shows", "olo")
    acronym_hit = find_hit(hits, "olo one-leg out")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [intro_hit, two_leg_hit, actor_hit, psp_hit, check_hit, corridor_hit, acronym_hit]:
        if not hit:
            continue
        unit_key = "|".join(
            [
                str(hit.chunk.get("local_path") or hit.chunk.get("doc_id") or ""),
                str(hit.chunk.get("unit_type") or ""),
                str(hit.chunk.get("unit") or ""),
            ]
        )
        if unit_key in seen:
            continue
        seen.add(unit_key)
        support.append(hit)

    if len(support) < 2:
        return None

    citations = citations_from_hits(support)
    intro_ref = ref_for_hit(citations, support, intro_hit)
    two_leg_ref = ref_for_hit(citations, support, two_leg_hit, intro_ref)
    actor_ref = ref_for_hit(citations, support, actor_hit, two_leg_ref)
    psp_ref = ref_for_hit(citations, support, psp_hit, actor_ref)
    check_ref = ref_for_hit(citations, support, check_hit, actor_ref)
    corridor_ref = ref_for_hit(citations, support, corridor_hit, intro_ref)
    acronym_ref = ref_for_hit(citations, support, acronym_hit, intro_ref)

    if language == "en":
        answer = (
            f"OLO means `One-Leg Out`, but in TIPS it is more than a glossary label. It is the cross-currency instant-payment model based on the EPC One-Leg Out Instant Credit Transfer / OCT Inst scheme. [{acronym_ref}][{intro_ref}]\n\n"
            f"The practical idea is that the end-to-end cross-currency payment is split into two separate instant payments: one in the source leg and one in the destination leg. [{two_leg_ref}] "
            f"The basic TIPS actors are the Originator Participant on the source leg and the Beneficiary Participant on the destination leg. [{actor_ref}]"
        )
        if psp_hit:
            answer += f" The bridge between legs is handled through a Leg Exit PSP in the outgoing/source leg and a Leg Entry PSP in the incoming/destination leg. [{psp_ref}]"
        if check_hit:
            answer += f" TIPS also has OLO-specific authorisation checks: incoming OLO checks the Creditor Agent cross-currency flag, while outgoing OLO checks the Leg Exit PSP cross-currency flag. [{check_ref}]"
        if corridor_hit:
            answer += f" In the UHB transaction view, the `Corridor` field shows `OLO` for OLO transactions, which distinguishes it from `ELKT` for the Enhanced Linked Transaction model. [{corridor_ref}]"
        source_title = "References"
    else:
        answer = (
            f"OLO significa `One-Leg Out`, pero en TIPS no conviene dejarlo como una simple sigla. Es el modelo de pago instantaneo cross-currency basado en el esquema EPC One-Leg Out Instant Credit Transfer / OCT Inst. [{acronym_ref}][{intro_ref}]\n\n"
            f"La idea practica es partir el pago cross-currency end-to-end en dos instant payments separadas: una en la source leg y otra en la destination leg. [{two_leg_ref}] "
            f"En TIPS, los actores base son el Originator Participant en la source leg y el Beneficiary Participant en la destination leg. [{actor_ref}]"
        )
        if psp_hit:
            answer += f" El puente entre ambas legs lo hacen el Leg Exit PSP en la outgoing/source leg y el Leg Entry PSP en la incoming/destination leg. [{psp_ref}]"
        if check_hit:
            answer += f" Ademas, TIPS aplica controles especificos de OLO: en una incoming OLO valida el Cross-currency Flag del Creditor Agent; en una outgoing OLO valida el Cross-currency Flag del Leg Exit PSP. [{check_ref}]"
        if corridor_hit:
            answer += f" En la vista UHB de la transaccion, el campo `Corridor` muestra `OLO` para operaciones OLO, y lo separa de `ELKT`, que corresponde al Enhanced Linked Transaction model. [{corridor_ref}]"
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_olo_lkt_comparison_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_olo_lkt_comparison_question(query):
        return None

    olo_intro_hit = (
        find_hit(hits, "one-leg out instant credit transfer", "designed and implemented in tips")
        or find_hit(hits, "one-leg out instant credit transfer", "implemented in tips")
    )
    olo_two_leg_hit = find_hit(hits, "two separate and independent instant payments", "source and destination leg")
    olo_roles_hit = find_hit(hits, "leg exit psp acts as beneficiary", "leg entry psp acts as originator")
    corridor_hit = find_hit(hits, "corridor shows", "olo", "elkt")
    lkt_model_hit = find_hit(hits, "cross-currency linked-transactions", "preferred settlement model")
    lkt_message_hit = find_hit(hits, "linked payment message", "settle simultaneously")
    lkt_atomic_hit = find_hit(hits, "enhanced lkt settlement model", "all transactions are settled or none")
    lkt_routing_hit = find_hit(hits, "both currencies", "pair is configured in the mapping table")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [
        olo_intro_hit,
        olo_two_leg_hit,
        olo_roles_hit,
        corridor_hit,
        lkt_model_hit,
        lkt_message_hit,
        lkt_atomic_hit,
        lkt_routing_hit,
    ]:
        if not hit:
            continue
        unit_key = "|".join(
            [
                str(hit.chunk.get("local_path") or hit.chunk.get("doc_id") or ""),
                str(hit.chunk.get("unit_type") or ""),
                str(hit.chunk.get("unit") or ""),
            ]
        )
        if unit_key in seen:
            continue
        seen.add(unit_key)
        support.append(hit)

    has_olo = bool(olo_intro_hit or olo_two_leg_hit or corridor_hit)
    has_lkt = bool(lkt_model_hit or lkt_message_hit or lkt_atomic_hit)
    if not (has_olo and has_lkt):
        return None

    citations = citations_from_hits(support)
    olo_intro_ref = ref_for_hit(citations, support, olo_intro_hit)
    olo_two_ref = ref_for_hit(citations, support, olo_two_leg_hit, olo_intro_ref)
    olo_roles_ref = ref_for_hit(citations, support, olo_roles_hit, olo_two_ref)
    corridor_ref = ref_for_hit(citations, support, corridor_hit, olo_intro_ref)
    lkt_model_ref = ref_for_hit(citations, support, lkt_model_hit)
    lkt_message_ref = ref_for_hit(citations, support, lkt_message_hit, lkt_model_ref)
    lkt_atomic_ref = ref_for_hit(citations, support, lkt_atomic_hit, lkt_model_ref)
    lkt_routing_ref = ref_for_hit(citations, support, lkt_routing_hit, lkt_model_ref)

    if language == "en":
        answer = (
            f"No. OLO and LKT are not the same thing in TIPS. OLO is the One-Leg Out / OCT Inst-based cross-currency model; LKT is the Linked Transaction settlement model. [{olo_intro_ref}][{lkt_model_ref}]\n\n"
            f"With OLO, the end-to-end cross-currency payment is described as two separate and independent instant payments, one in the source leg and one in the destination leg. [{olo_two_ref}] "
            f"The intermediary role changes by direction: Leg Exit PSP in the outgoing/source leg, Leg Entry PSP in the incoming/destination leg. [{olo_roles_ref}]\n\n"
            f"With LKT, the key point is linked settlement: a linked payment message may settle only simultaneously with another linked payment message. [{lkt_message_ref}] "
            f"In the enhanced LKT model, settlement is atomic: all linked transactions settle, or none of them settles. [{lkt_atomic_ref}]\n\n"
            f"The clean operational separator is the `Corridor` value: the UHB shows `OLO` for OLO transactions and `ELKT` for the Enhanced Linked Transaction model. [{corridor_ref}]"
        )
        if lkt_routing_hit:
            answer += f" The LKT flow applies when both currencies are hosted in TIPS and the pair is configured in the mapping table; otherwise the payment follows the regular cross-currency model or is rejected if no currency is hosted in TIPS. [{lkt_routing_ref}]"
        source_title = "References"
    else:
        answer = (
            f"No. OLO y LKT no son lo mismo en TIPS. OLO es el modelo cross-currency basado en One-Leg Out / OCT Inst; LKT es el modelo de liquidacion por transacciones enlazadas. [{olo_intro_ref}][{lkt_model_ref}]\n\n"
            f"En OLO, el pago cross-currency end-to-end se describe como dos instant payments separadas e independientes: una en la source leg y otra en la destination leg. [{olo_two_ref}] "
            f"El intermediario cambia segun la direccion: Leg Exit PSP en la outgoing/source leg, Leg Entry PSP en la incoming/destination leg. [{olo_roles_ref}]\n\n"
            f"En LKT, lo importante es la liquidacion enlazada: un linked payment message solo puede liquidarse al mismo tiempo que otro linked payment message. [{lkt_message_ref}] "
            f"En el LKT mejorado, la liquidacion es atomica: se liquidan todas las transacciones enlazadas o no se liquida ninguna. [{lkt_atomic_ref}]\n\n"
            f"La forma rapida de distinguirlos en operativa es el campo `Corridor`: la UHB muestra `OLO` para operaciones OLO y `ELKT` para el Enhanced Linked Transaction model. [{corridor_ref}]"
        )
        if lkt_routing_hit:
            answer += f" El flujo LKT aplica cuando ambas divisas estan alojadas en TIPS y el par esta configurado en la mapping table; si el par no esta configurado, se usa el modelo cross-currency regular, y si ninguna divisa esta alojada en TIPS la transaccion se rechaza. [{lkt_routing_ref}]"
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_lkt_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_lkt_question(query):
        return None

    model_hit = find_hit(hits, "cross-currency linked-transactions", "preferred settlement model")
    linked_message_hit = find_hit(hits, "linked payment message", "settle simultaneously")
    atomic_hit = find_hit(hits, "enhanced lkt settlement model", "all transactions are settled or none")
    routing_hit = find_hit(hits, "both currencies", "pair is configured in the mapping table")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [model_hit, linked_message_hit, atomic_hit, routing_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)

    if len(support) < 2:
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    model_ref = ref_for(model_hit)
    linked_ref = ref_for(linked_message_hit, model_ref)
    atomic_ref = ref_for(atomic_hit, model_ref)
    routing_ref = ref_for(routing_hit, atomic_ref)

    if language == "en":
        answer = (
            "LKT means `Linked Transaction Model`. In TIPS it is the `Linked-Transactions (LKT) settlement model`, "
            f"identified in the local documentation as the preferred settlement model for cross-currency payments in TIPS. [{model_ref}]\n\n"
            f"The core idea is linked settlement: a `linked payment message` is a payment instruction that TIPS may settle only simultaneously with another linked payment message. [{linked_ref}] "
            f"In the enhanced LKT model, the two mono-currency legs of a cross-currency payment are settled atomically: either all linked transactions settle, or none of them settles. [{atomic_ref}]"
        )
        if routing_hit:
            answer += (
                f"\n\nOperationally, when both currencies in the cross-currency transaction are hosted in TIPS and the currency pair is configured in the mapping table, the payment follows the LKT cross-currency flow. "
                f"If the pair is not configured, it follows the regular cross-currency model; if neither currency is hosted in TIPS, the transaction is rejected. [{routing_ref}]"
            )
        source_title = "References"
    else:
        answer = (
            "LKT significa `Linked Transaction Model`. En TIPS es el `Linked-Transactions (LKT) settlement model`, "
            f"que la documentación local identifica como el modelo preferido para liquidar pagos cross-currency en TIPS. [{model_ref}]\n\n"
            f"La idea es enlazar la liquidación: un `linked payment message` es una instrucción que TIPS solo puede liquidar al mismo tiempo que otro linked payment message. [{linked_ref}] "
            f"En el modelo LKT mejorado, las dos legs mono-currency de un pago cross-currency se liquidan de forma atómica: o se liquidan todas las transacciones enlazadas, o no se liquida ninguna. [{atomic_ref}]"
        )
        if routing_hit:
            answer += (
                f"\n\nEn la práctica, si las dos divisas de la operación cross-currency están alojadas en TIPS y el par está configurado en la mapping table, el pago sigue el flujo LKT cross-currency. "
                f"Si el par no está configurado, se usa el modelo cross-currency regular; si ninguna divisa está alojada en TIPS, la transacción se rechaza. [{routing_ref}]"
            )
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_investigation_offset_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_investigation_offset_question(query):
        return None

    parameter_hit = find_hit(hits, "investigation offset", "configurable offset foreseen in sct inst scheme")
    non_euro_hit = find_hit(hits, "investigation offset", "non-euro currency", "negative value")
    rule_hit = find_hit(hits, "answers to an investigation request only if", "expired for more than 5 seconds")
    check_hit = find_hit(hits, "investigation request has been received after", "sctinst timestamp timeout + investigation offset")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [parameter_hit, non_euro_hit, rule_hit, check_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)
    if not support:
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    parameter_ref = ref_for(parameter_hit)
    non_euro_ref = ref_for(non_euro_hit, parameter_ref)
    rule_ref = ref_for(rule_hit, parameter_ref)
    check_ref = ref_for(check_hit, rule_ref)

    if language == "en":
        answer = (
            f"`Investigation Offset` is the configurable waiting margin that TIPS adds to the Instant Payment timestamp timeout before accepting an Investigation request. [{parameter_ref}] "
            f"In other words, TIPS does not answer an investigation immediately after the payment is sent; it waits until the normal timeout window has expired, plus this offset, so the settlement phase is certain to be complete. [{rule_ref}]\n\n"
            f"For SCT Inst, the documented default value is 5,000 ms. [{parameter_ref}] "
        )
        if non_euro_hit:
            answer += f"For non-euro currencies, TIPS has a separate `Investigation Offset (non-Euro currency)` parameter; the local UDFS says it can be negative, with a documented value of -10,000 ms in the cited table. [{non_euro_ref}] "
        if check_hit:
            answer += f"In the Investigation flow, TIPS checks that the request was received after `SCTInst Timestamp Timeout + Investigation Offset`; if the check is not satisfied, the request follows the error path. [{check_ref}]"
        source_title = "References"
    else:
        answer = (
            f"`Investigation Offset` es el margen configurable que TIPS suma al timeout del Instant Payment antes de aceptar una Investigation request. [{parameter_ref}] "
            f"Dicho de forma simple: TIPS no responde a una investigación en cuanto llega la petición; espera a que haya vencido el timeout normal de la operación, más ese offset, para tener certeza de que la fase de liquidación ya ha terminado. [{rule_ref}]\n\n"
            f"Para SCT Inst, el valor por defecto documentado es 5.000 ms. [{parameter_ref}] "
        )
        if non_euro_hit:
            answer += f"Para divisas no euro existe un parámetro separado, `Investigation Offset (non-Euro currency)`, que puede tener valor negativo; en la tabla citada aparece con -10.000 ms. [{non_euro_ref}] "
        if check_hit:
            answer += f"En el flujo de Investigation, TIPS comprueba que la petición haya llegado después de `SCTInst Timestamp Timeout + Investigation Offset`; si no se cumple, la petición va por la ruta de error. [{check_ref}]"
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_u2a_functions_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_u2a_functions_question(query):
        return None

    list_hit = (
        find_hit(hits, "complete list of functions available 24/7/365 via the tips gui", "payment transaction advanced status query")
        or find_hit(hits, "functions available in tips gui", "account balance and status query")
    )
    continuation_hit = find_hit(hits, "increase/decrease of a cmb limit", "task management task list", "enter liquidity transfer order")
    menu_hit = find_hit(hits, "gui menu is structured into two hierarchical menu levels", "payment transaction", "audit trail")
    audit_hit = find_hit(hits, "audit trail", "reference and transactional objects", "available only in u2a mode")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [list_hit, continuation_hit, menu_hit, audit_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)

    if not list_hit:
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    list_ref = ref_for(list_hit)
    continuation_ref = ref_for(continuation_hit, list_ref)
    menu_ref = ref_for(menu_hit, list_ref)
    audit_ref = ref_for(audit_hit, menu_ref)

    if language == "en":
        answer = (
            f"In U2A, TIPS exposes GUI functions rather than an unrestricted payment-entry channel. The UHB lists the GUI functions available via TIPS 24/7/365. [{list_ref}]\n\n"
            "The U2A/GUI scope includes:\n"
            f"- Queries: account balance and status, CMB limit and status, payment transaction status, liquidity transfer status, advanced payment transaction status, advanced liquidity transfer status, and broadcast query. [{list_ref}]\n"
            f"- Local reference data actions: block/unblock a TIPS Participant or Ancillary System, block/unblock an account, block/unblock a CMB, and increase/decrease a CMB limit. [{list_ref}][{continuation_ref}]\n"
            f"- Task management: Task List. [{continuation_ref}]\n"
            f"- Liquidity management: enter a Liquidity Transfer Order. [{continuation_ref}]\n"
            f"- Broadcast: enter a Broadcast. [{continuation_ref}]\n\n"
            f"The GUI menu also shows the main work areas: TIPS Party, Account, Credit Memorandum Balance, Liquidity Transfer, Payment Transaction, Task List and Audit Trail; Payment Transaction and Liquidity Transfer have Search/Advanced Search navigation. [{menu_ref}] "
            f"Audit Trail is also a U2A-only search area for revisions on reference and transactional objects. [{audit_ref}]\n\n"
            f"So, if by 'transactions' you mean payment operations, the U2A evidence supports querying payment transactions and entering liquidity transfer orders. It does not show the GUI as a free-form channel for initiating every instant payment message. [{list_ref}][{continuation_ref}]"
        )
        source_title = "References"
    else:
        answer = (
            f"En U2A, TIPS expone funciones de GUI, no un canal libre para meter cualquier pago. La UHB lista las funciones disponibles en la GUI de TIPS 24/7/365. [{list_ref}]\n\n"
            "El alcance U2A/GUI incluye:\n"
            f"- Consultas: balance y estado de cuenta, límite y estado de CMB, estado de payment transaction, estado de liquidity transfer, advanced payment transaction status, advanced liquidity transfer status y consulta de broadcast. [{list_ref}]\n"
            f"- Datos de referencia local: bloquear/desbloquear un TIPS Participant o Ancillary System, bloquear/desbloquear una cuenta, bloquear/desbloquear un CMB e incrementar/disminuir el límite de un CMB. [{list_ref}][{continuation_ref}]\n"
            f"- Gestión de tareas: Task List. [{continuation_ref}]\n"
            f"- Gestión de liquidez: introducir una Liquidity Transfer Order. [{continuation_ref}]\n"
            f"- Broadcast: introducir un Broadcast. [{continuation_ref}]\n\n"
            f"El menú de la GUI confirma las áreas principales: TIPS Party, Account, Credit Memorandum Balance, Liquidity Transfer, Payment Transaction, Task List y Audit Trail; Payment Transaction y Liquidity Transfer tienen navegación de Search/Advanced Search. [{menu_ref}] "
            f"Audit Trail también aparece como área U2A-only para buscar revisiones sobre objetos de referencia y transaccionales. [{audit_ref}]\n\n"
            f"Así que, si por “transacciones” te refieres a operativa de pagos, la evidencia U2A soporta consultar payment transactions e introducir órdenes de liquidity transfer. No aparece como canal GUI libre para iniciar cualquier mensaje de instant payment. [{list_ref}][{continuation_ref}]"
        )
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_uhb_u2a_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_uhb_u2a_question(query):
        return None

    glossary_hit = (
        find_hit(hits, "uhb user handbook", "u2a user-to-application")
        or find_hit(hits, "uhb user handbook", "u2a user-to-application", "target instant payment settlement")
    )
    gui_hit = find_hit(hits, "the gui is a browser-based application", "communication with tips in u2a mode")
    interaction_hit = find_hit(hits, "each interaction with tips", "a2a or u2a mode", "message or a gui screen")
    connectivity_hit = find_hit(hits, "tips platform can be accessed", "a2a mode and u2a mode")
    u2a_only_hit = find_hit(hits, "this function is available in u2a mode only", "screen access")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [glossary_hit, gui_hit, interaction_hit, connectivity_hit, u2a_only_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)

    if not (glossary_hit or gui_hit or connectivity_hit):
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    glossary_ref = ref_for(glossary_hit or gui_hit or connectivity_hit)
    gui_ref = ref_for(gui_hit, glossary_ref)
    interaction_ref = ref_for(interaction_hit or connectivity_hit, gui_ref)
    u2a_only_ref = ref_for(u2a_only_hit, interaction_ref)

    if language == "en":
        answer = (
            f"Yes. TIPS has a UHB: `UHB` means `User Handbook`, and the local TIPS documentation also defines `U2A` as `User-to-Application`. [{glossary_ref}]\n\n"
            f"Yes, TIPS also has U2A. The UHB describes the TIPS GUI as a browser-based application used to communicate with TIPS in U2A mode. [{gui_ref}] "
            f"So U2A is the user/GUI channel; it is not a separate payment system.\n\n"
            f"The nuance is that TIPS is not only U2A. TIPS user functions can be triggered in A2A or U2A mode, by message or by GUI screen, depending on the function. [{interaction_ref}] "
            f"For example, some screen functions are explicitly U2A-only, while other functions can be available in both U2A and A2A mode. [{u2a_only_ref}]"
        )
        source_title = "References"
    else:
        answer = (
            f"Sí. En TIPS hay UHB: `UHB` significa `User Handbook`, y la misma documentación local define `U2A` como `User-to-Application`. [{glossary_ref}]\n\n"
            f"Sí, también hay U2A en TIPS. La UHB describe la GUI de TIPS como una aplicación de navegador para comunicarse con TIPS en modo U2A. [{gui_ref}] "
            f"Así que U2A es el canal de usuario/GUI; no es otro sistema de pago.\n\n"
            f"El matiz es que TIPS no es solo U2A. Las funciones de usuario de TIPS pueden activarse en modo A2A o U2A, mediante mensaje o mediante pantalla GUI, según la función. [{interaction_ref}] "
            f"Por ejemplo, algunas pantallas son explícitamente solo U2A, mientras que otras funciones pueden estar disponibles tanto en U2A como en A2A. [{u2a_only_ref}]"
        )
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_source_currency_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_source_currency_question(query):
        return None

    def normalized_text(hit: Hit) -> str:
        return re.sub(
            r"\s+",
            " ",
            " ".join([hit.chunk.get("title", ""), hit.chunk.get("release", ""), hit.chunk.get("text", "")]),
        ).lower()

    def first_with(*needles: str) -> Hit | None:
        matches = [hit for hit in hits if all(needle.lower() in normalized_text(hit) for needle in needles)]
        if not matches:
            return None
        matches.sort(
            key=lambda hit: (
                hit.chunk.get("release") == "R2026.NOV",
                hit.chunk.get("revision_status") == "clean",
                hit.chunk.get("family") == "tips_udfs",
                hit.score,
            ),
            reverse=True,
        )
        return matches[0]

    flow_hit = first_with("two separate and independent instant payments", "source and destination leg")
    actor_hit = first_with("starting the cross-currency payment in the source leg")
    amount_hit = first_with("interbank settlement amount", "destination currency", "instructed amount", "source currency")
    diagram_hit = first_with("from tips (source currency)", "tips (destination currency)")

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [flow_hit, actor_hit, amount_hit, diagram_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)

    if not support:
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    flow_ref = ref_for(flow_hit or actor_hit)
    actor_ref = ref_for(actor_hit, flow_ref)
    amount_ref = ref_for(amount_hit, flow_ref)
    diagram_ref = ref_for(diagram_hit, flow_ref)

    if language == "en":
        answer = (
            "`TIPS (source currency)` is not a separate actor or a special currency. It means the TIPS side/component that processes the source-currency leg of a cross-currency payment. "
            f"In TIPS, an end-to-end cross-currency payment is split into two independent Instant Payments: one in the source leg currency and one in the destination leg currency. [{flow_ref}]\n\n"
            f"The source currency is therefore the currency of the leg where the cross-currency payment starts; the Originator Participant, Ancillary System or Instructing Party starts the payment in that source leg. [{actor_ref}] "
            "The destination currency is the currency of the leg where the Beneficiary side receives and confirms or rejects the request."
        )
        if amount_hit:
            answer += f"\n\nIn message terms, when the destination leg is instructed, the Interbank Settlement Amount carries the amount in the destination currency, while the Instructed Amount carries the amount in the source currency. [{amount_ref}]"
        if diagram_hit:
            answer += f"\n\nSo a diagram label such as `From TIPS (source currency)` / `To TIPS (source currency)` means communication back to the TIPS processing side for the source leg, not a third-party system. [{diagram_ref}]"
        source_title = "References"
    else:
        answer = (
            "`TIPS (source currency)` no es un actor distinto ni una divisa especial. Significa el lado/componente de TIPS que procesa la leg en la divisa de origen de un pago cross-currency. "
            f"En TIPS, una transacción cross-currency end-to-end se divide en dos instant payments independientes: una en la divisa de la source leg y otra en la divisa de la destination leg. [{flow_ref}]\n\n"
            f"La source currency es, por tanto, la divisa de la leg donde arranca el pago. Ahí actúa el Originator Participant, Ancillary System o Instructing Party que inicia el pago cross-currency en la source leg. [{actor_ref}] "
            "La destination currency es la divisa de la leg donde el lado beneficiario recibe la petición y la confirma o la rechaza."
        )
        if amount_hit:
            answer += f"\n\nA nivel de mensaje, cuando se instruye la destination leg, el `Interbank Settlement Amount` lleva el importe en la destination currency, mientras que el `Instructed Amount` lleva el importe en la source currency. [{amount_ref}]"
        if diagram_hit:
            answer += f"\n\nPor eso, una etiqueta de diagrama como `From TIPS (source currency)` o `To TIPS (source currency)` se refiere a comunicación con el lado de TIPS que lleva la leg de origen, no a otro sistema ni a otra entidad. [{diagram_ref}]"
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_currency_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_currency_question(query):
        return None

    current_hit = find_hit(hits, "tips settles instant payments in euro", "swedish kronor", "danish kroner")
    supported_hit = (
        find_hit(hits, "for each currency supported in tips", "eur, sek, dkk")
        or find_hit(hits, "currently defined in the system", "eur, sek", "dkk")
        or find_hit(hits, "any tips hosted currencies", "eur, sek and dkk")
    )
    dkk_hit = find_hit(hits, "dkk among the active settlement currencies", "eur, sek")
    nok_hit = (
        find_hit(hits, "norges bank", "nok business date")
        or find_hit(hits, "enabling new settlement currency in tips", "nok")
        or find_hit(hits, "norges bank is also preparing to onboard")
    )

    support: list[Hit] = []
    seen: set[str] = set()
    for hit in [current_hit, supported_hit, dkk_hit, nok_hit]:
        if not hit:
            continue
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        support.append(hit)

    if not current_hit and not supported_hit:
        return None

    citations = []
    for n, hit in enumerate(support, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )

    def ref_for(hit: Hit | None, fallback: int = 1) -> int:
        if not hit:
            return fallback
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit))
        for item, support_hit in zip(citations, support):
            if str(support_hit.chunk.get("chunk_id") or id(support_hit)) == chunk_id:
                return item["n"]
        return fallback

    current_ref = ref_for(current_hit or supported_hit)
    supported_ref = ref_for(supported_hit, current_ref)
    dkk_ref = ref_for(dkk_hit, supported_ref)
    nok_ref = ref_for(nok_hit, current_ref)

    if language == "en":
        answer = (
            "The TIPS settlement currencies in the local documentation are:\n\n"
            "- EUR: euro.\n"
            "- SEK: Swedish krona/kronor.\n"
            "- DKK: Danish krone/kroner.\n\n"
            f"The cleanest source is the ECB onboarding text: it says that TIPS settles instant payments in euro, Swedish kronor and Danish kroner. [{current_ref}]"
        )
        if supported_hit:
            answer += f" Functional/change-request material also treats EUR, SEK and DKK as the currencies supported or currently defined in TIPS. [{supported_ref}]"
        if dkk_hit and dkk_hit is not supported_hit:
            answer += f" The Danish onboarding material separately says DKK was added among the active settlement currencies, while EUR and SEK were already present. [{dkk_ref}]"
        if nok_hit:
            answer += f" NOK appears in the corpus as Norwegian onboarding/adaptation work, so I would not include it in the main 'currencies already in TIPS' list unless the question is explicitly about upcoming/onboarding currencies. [{nok_ref}]"
        source_title = "References"
    else:
        answer = (
            "Las divisas de liquidación de TIPS que aparecen en la documentación local son:\n\n"
            "- EUR: euro.\n"
            "- SEK: corona sueca / Swedish krona.\n"
            "- DKK: corona danesa / Danish krone.\n\n"
            f"La fuente más directa es el onboarding del ECB: dice que TIPS liquida pagos instantáneos en euro, Swedish kronor y Danish kroner. [{current_ref}]"
        )
        if supported_hit:
            answer += f" Además, material funcional y de CR trata EUR, SEK y DKK como las currencies soportadas o actualmente definidas en TIPS. [{supported_ref}]"
        if dkk_hit and dkk_hit is not supported_hit:
            answer += f" La documentación de onboarding de DKK añade que DKK se incluye entre las active settlement currencies, manteniendo EUR y SEK como divisas ya existentes. [{dkk_ref}]"
        if nok_hit:
            answer += f" NOK aparece en el corpus como onboarding/adaptación de Norges Bank; por eso no lo metería en la lista principal de “divisas que hay en TIPS” salvo que preguntes por divisas futuras o en onboarding. [{nok_ref}]"
        source_title = "Referencias"

    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_investigation_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_investigation_question(query):
        return None

    intro_hit = find_hit(hits, "2.4. investigation", "processing of an investigation request")
    messages_hit = find_hit(hits, "fitofipaymentstatusrequest", "pacs.028.001.03", "fitofipaymentstatusreport", "pacs.002.001.10")
    checks_hit = find_hit(hits, "payment transaction existence", "investigation allowed")
    support = unique_hits(intro_hit, messages_hit, checks_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    intro_ref = ref_for_hit(citations, support, intro_hit)
    messages_ref = ref_for_hit(citations, support, messages_hit, intro_ref)
    checks_ref = ref_for_hit(citations, support, checks_hit, intro_ref)

    if language == "en":
        answer = (
            f"In TIPS, an `Investigation` is a transaction-status investigation: a participant, ancillary system or instructing party asks TIPS for the status of one or more payment transactions. [{intro_ref}] "
            f"The A2A message pattern is `pacs.028.001.03` (`FIToFIPaymentStatusRequest`) in and `pacs.002.001.10` (`FIToFIPaymentStatusReport`) back. [{messages_ref}] "
            f"Before answering, TIPS checks access rights, that the instructing party is authorised, that the payment transaction exists and that the investigation is allowed. [{checks_ref}]"
        )
    else:
        answer = (
            f"En TIPS, una `Investigation` es una investigacion de estado de una transaccion: un Participant, Ancillary System o Instructing Party pregunta a TIPS por el estado de uno o varios pagos. [{intro_ref}] "
            f"En A2A, la peticion entra como `pacs.028.001.03` (`FIToFIPaymentStatusRequest`) y TIPS contesta con `pacs.002.001.10` (`FIToFIPaymentStatusReport`). [{messages_ref}] "
            f"Antes de responder, TIPS comprueba permisos, que el instructing party este autorizado, que la transaccion exista y que la investigation este permitida. [{checks_ref}]"
        )

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def build_recall_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_recall_question(query):
        return None

    intro_hit = find_hit(hits, "2.3. recall", "request the cancellation", "return of funds")
    messages_hit = find_hit(hits, "fitofipaymentcancellationrequest", "paymentreturn", "resolutionofinvestigation")
    steps_hit = find_hit(hits, "recall steps", "camt.056.001.08")
    support = unique_hits(intro_hit, messages_hit, steps_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    intro_ref = ref_for_hit(citations, support, intro_hit)
    messages_ref = ref_for_hit(citations, support, messages_hit, intro_ref)
    steps_ref = ref_for_hit(citations, support, steps_hit, messages_ref)

    if language == "en":
        answer = (
            f"In TIPS, a `Recall` is the process used to request the cancellation and return of funds of a previously settled Instant Payment. [{intro_ref}] "
            f"The Recall Assigner is the originator-side party sending the request; the Recall Assignee is the beneficiary-side party receiving it. [{intro_ref}] "
            f"The key messages are `FIToFIPaymentCancellationRequest` / `camt.056.001.08`, `PaymentReturn` / `pacs.004.001.09` for a positive response, and `ResolutionOfInvestigation` / `camt.029.001.09` for a negative response. [{messages_ref}][{steps_ref}]"
        )
    else:
        answer = (
            f"En TIPS, un `Recall` es el proceso para pedir la cancelacion y la devolucion de fondos de un Instant Payment ya liquidado. [{intro_ref}] "
            f"El Recall Assigner es la parte del lado originador que manda la peticion; el Recall Assignee es la parte beneficiaria que la recibe. [{intro_ref}] "
            f"Los mensajes clave son `FIToFIPaymentCancellationRequest` / `camt.056.001.08`, `PaymentReturn` / `pacs.004.001.09` para una respuesta positiva, y `ResolutionOfInvestigation` / `camt.029.001.09` para una respuesta negativa. [{messages_ref}][{steps_ref}]"
        )

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def build_cross_currency_flag_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_cross_currency_flag_question(query):
        return None

    business_hit = find_hit(hits, "cross-currency flag", "incoming olo cross-currency") or find_hit(hits, "cross-currency flag", "outgoing olo cross-currency")
    directory_hit = find_hit(hits, "authorised account user bic", "authorised to accept cross-currency payments")
    support = unique_hits(business_hit, directory_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    business_ref = ref_for_hit(citations, support, business_hit)
    directory_ref = ref_for_hit(citations, support, directory_hit, business_ref)

    if language == "en":
        answer = (
            f"`Cross-currency Flag` is the authorisation flag TIPS uses to decide whether a BIC is allowed to participate in cross-currency/OLO processing for the relevant account relationship. [{directory_ref}] "
            f"In OLO business checks, TIPS uses that flag for incoming OLO on the Creditor Agent and for outgoing OLO on the Leg Exit PSP; if the flag is not set as required, the transaction is rejected by the business rule. [{business_ref}]"
        )
    else:
        answer = (
            f"`Cross-currency Flag` es el indicador de autorizacion que TIPS usa para saber si un BIC puede participar en pagos cross-currency/OLO dentro de la relacion de cuenta correspondiente. [{directory_ref}] "
            f"En los business checks de OLO, TIPS mira ese flag en el Creditor Agent para incoming OLO y en el Leg Exit PSP para outgoing OLO; si el flag no cumple, la transaccion se rechaza por regla de negocio. [{business_ref}]"
        )

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def build_authorised_account_user_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_authorised_account_user_question(query):
        return None

    definition_hit = find_hit(hits, "record type", "authorised account user", "cash account")
    reachable_hit = find_hit(hits, "reachable parties defined as authorised account users")
    xcy_hit = find_hit(hits, "authorised account user bic", "authorised to accept cross-currency payments")
    support = unique_hits(definition_hit, reachable_hit, xcy_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    definition_ref = ref_for_hit(citations, support, definition_hit)
    reachable_ref = ref_for_hit(citations, support, reachable_hit, definition_ref)
    xcy_ref = ref_for_hit(citations, support, xcy_hit, definition_ref)

    if language == "en":
        answer = (
            f"`Authorised Account User` is the CRDM/TIPS record used to define BICs that are authorised to use or be reachable through a Cash Account, TIPS AS Technical Account or CMB. [{definition_ref}] "
            f"The participant information guide describes reachable parties as authorised account users of a participant's accounts or CMBs. [{reachable_ref}] "
            f"For cross-currency, the same area carries the cross-currency authorisation flag for the Authorised Account User BIC. [{xcy_ref}]"
        )
    else:
        answer = (
            f"`Authorised Account User` es el registro CRDM/TIPS que define los BIC autorizados a usar una Cash Account, una TIPS AS Technical Account o un CMB, o a ser alcanzables a traves de ellos. [{definition_ref}] "
            f"La guia de participantes describe las reachable parties como authorised account users de las cuentas o CMBs del participante. [{reachable_ref}] "
            f"En cross-currency, esa misma zona contiene el flag que indica si el Authorised Account User BIC puede aceptar pagos cross-currency. [{xcy_ref}]"
        )

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def clean_cr_code(code: str) -> str:
    return code.upper().replace(".", "-")


def cr_entries_by_code() -> dict[str, dict[str, Any]]:
    entries = load_json(CHANGE_REQUESTS_PATH)
    return {clean_cr_code(entry.get("code", "")): entry for entry in entries if entry.get("code")}


def scope_codes_from_hits(hits: list[Hit], release: str) -> list[str]:
    cover_note_codes: list[str] = []
    content_codes: list[str] = []
    for hit in hits:
        if hit.chunk.get("release") != release:
            continue
        title = (hit.chunk.get("title") or "").lower()
        if "cover note" not in title and "content of tips release" not in title:
            continue
        text = hit.chunk.get("text", "")
        text = re.sub(r"\bTIPS-(\d{4})([345])\b", r"TIPS-\1", text)
        target = cover_note_codes if "cover note" in title else content_codes
        if "cover note" in title and "scope includes" in text.lower():
            lower = text.lower()
            start = lower.find("scope includes")
            end = lower.find("it is foreseen", start)
            text = text[start:end if end > start else None]
        elif "content of tips release" in title:
            marker = re.search(r"\n\s*1\s+", text)
            if marker:
                text = text[: marker.start()]
        for code in re.findall(r"\bTIPS-\d{4}\b", text, re.I):
            code = clean_cr_code(code)
            if code not in target:
                target.append(code)
    return cover_note_codes or content_codes


def find_cr_entry(entries: dict[str, dict[str, Any]], code: str) -> dict[str, Any] | None:
    code = clean_cr_code(code)
    if code in entries:
        return entries[code]
    prefix = f"{code}-"
    matches = [entry for key, entry in entries.items() if key.startswith(prefix)]
    if not matches:
        return None
    matches.sort(key=lambda entry: (bool(entry.get("status")), entry.get("date") or ""), reverse=True)
    return matches[0]


def change_request_citation(n: int, label: str | None = None) -> dict[str, Any]:
    return {
        "n": n,
        "title": "Structured TIPS change request catalogue",
        "release": "",
        "family": "change_requests",
        "unit_type": "change_request_catalogue",
        "unit": "local",
        "local_path": str(CHANGE_REQUESTS_PATH.relative_to(ROOT)),
        "source_url": "https://www.ecb.europa.eu/paym/target/tips/governance/html/changerequests.en.html",
        "score": 1.0,
        "label": label or "Structured TIPS change request catalogue, change_request_catalogue local",
    }


def build_change_request_summary_answer(
    query: str,
    selected_codes: list[str],
    entries: dict[str, dict[str, Any]],
    hits: list[Hit],
    language: str,
) -> dict[str, Any] | None:
    if not selected_codes or not (is_change_request_summary_question(query) or len(selected_codes) == 1):
        return None

    code = selected_codes[0]
    entry = find_cr_entry(entries, code)
    if not entry:
        return None

    code_label = entry.get("code") or code
    title = entry.get("title") or code_label
    status = entry.get("status") or "sin estado en la lista local del ECB"
    published = bool(entry.get("published"))
    date_value = entry.get("date") or ""
    release = entry.get("release") or ""

    code_needles = [code_label, code_label.replace("-URD", ""), code_label.replace("-SYS", "")]

    def hit_for(*needles: str) -> Hit | None:
        for code_needle in code_needles:
            hit = find_hit(hits, code_needle, *needles)
            if hit:
                return hit
        return find_hit(hits, *needles)

    meta_hit = next(
        (
            hit
            for hit in hits
            if hit.chunk.get("family") == "change_requests"
            and str(hit.chunk.get("unit", "")).upper() == clean_cr_code(code_label)
            and hit.chunk.get("unit_type") == "change_request"
        ),
        None,
    )
    reason_hit = hit_for("reason for change", "enhanced linked transaction")
    description_hit = hit_for("description of requested", "two transactions", "atomic")
    flow_hit = hit_for("instant payment transaction steps for cross-currency", "lkt settlement model")
    routing_hit = hit_for("both currencies", "mapping table")
    gui_hit = hit_for("advanced cross-currency payment transaction query")
    impact_hit = hit_for("summary of application development impact")
    related_hit = hit_for("TIPS-0090", "out-of-scope")

    support = unique_hits(meta_hit, reason_hit, description_hit, flow_hit, routing_hit, gui_hit, impact_hit, related_hit)
    if not support:
        return None

    citations = citations_from_hits(support)
    meta_ref = ref_for_hit(citations, support, meta_hit, 1)
    reason_ref = ref_for_hit(citations, support, reason_hit, meta_ref)
    description_ref = ref_for_hit(citations, support, description_hit, reason_ref)
    flow_ref = ref_for_hit(citations, support, flow_hit, description_ref)
    routing_ref = ref_for_hit(citations, support, routing_hit, description_ref)
    gui_ref = ref_for_hit(citations, support, gui_hit, description_ref)
    impact_ref = ref_for_hit(citations, support, impact_hit, gui_ref)
    related_ref = ref_for_hit(citations, support, related_hit, reason_ref)

    if clean_cr_code(code_label).startswith("TIPS-0065"):
        if language == "en":
            answer = (
                f"Short answer: {code_label} is the CR that introduces the Enhanced Linked Transaction model for cross-currency settlement in TIPS. "
                f"It is listed as published on {date_value or 'the ECB CR page'} and allocated to {status}. [{meta_ref}]\n\n"
                f"What changes: the CR moves cross-currency payments between TIPS-hosted currencies away from a purely external orchestration model. "
                f"The goal is to reduce the operational burden on Entry/Exit-Leg PSPs, keep the payment instant end-to-end, and give certainty that funds reach the ultimate beneficiary. [{reason_ref}]\n\n"
                f"How it works: the end-to-end cross-currency payment is split into two mono-currency legs: one between the Originator PSP and the Exit-Leg PSP in the originator currency, and one between the Entry-Leg PSP and the Beneficiary PSP in the beneficiary currency. "
                f"The important point is atomic settlement: TIPS must technically settle the linked legs as all-or-nothing. [{description_ref}] "
                f"The LKT flow uses `FItoFICustomerCreditTransfer`, `FIToFIPaymentStatusReport` and optional `ReturnAccount` messages, and the detailed UDFS-style flow starts with TIPS receiving the cross-currency LKT payment in the source currency component. [{flow_ref}]\n\n"
                f"Routing and configuration: if both currencies are hosted in TIPS and the pair is configured in the mapping table, the payment follows LKT. If the pair is not configured, it falls back to the regular cross-currency model; if none of the currencies is hosted in TIPS, the transaction is rejected. [{routing_ref}]\n\n"
                f"Operational impact: the CR adds dedicated cross-currency GUI query screens so users can inspect linked mono-currency transactions, FX rate, intermediaries and the transaction ID of the other leg. [{gui_ref}] "
                f"On the application side it adds Message Router flows for Source CSM and Destination CSM roles, internal markers/states for linked cross-currency transactions, and inter-CSM communication queues. [{impact_ref}]\n\n"
                f"Limits: recall and positive/negative recall answers are explicitly not part of this CR; the CR says additional business cases would need a separate CR if required. Later TIPS-0090 covers further cross-currency enhancements that were out of scope for TIPS-0065. [{related_ref}]"
            )
        else:
            answer = (
                f"Respuesta corta: {code_label} es la CR que introduce en TIPS el modelo `Enhanced Linked Transaction` para liquidar pagos cross-currency. "
                f"Está publicada/listada en la página local del ECB con fecha {date_value or 'no indicada'} y asignada a {status}. [{meta_ref}]\n\n"
                f"Qué cambia: la CR deja de tratar ciertos pagos cross-currency entre divisas alojadas en TIPS como dos legs que dependen de una orquestación externa completa por parte de los Entry/Exit-Leg PSPs. "
                f"El objetivo es reducir esa complejidad operativa, mantener el pago como instantáneo end-to-end y dar certeza de entrega de fondos al beneficiario final. [{reason_ref}]\n\n"
                f"Cómo funciona: el pago cross-currency se divide en dos legs mono-currency. La primera va entre el Originator PSP y el Exit-Leg PSP en la divisa del originador; la segunda va entre el Entry-Leg PSP y el Beneficiary PSP en la divisa del beneficiario. "
                f"La clave de LKT es la atomicidad: TIPS debe liquidar técnicamente las legs enlazadas como un todo, es decir, se liquidan todas o no se liquida ninguna. [{description_ref}] "
                f"El flujo usa `FItoFICustomerCreditTransfer`, `FIToFIPaymentStatusReport` y, de forma opcional, `ReturnAccount`; la tabla funcional arranca con TIPS recibiendo el pago LKT cross-currency en el componente de source currency. [{flow_ref}]\n\n"
                f"Configuración y routing: si las dos divisas de la operación están alojadas en TIPS y el par está configurado en la mapping table, el pago sigue el flujo LKT. Si el par no está configurado, va por el modelo regular cross-currency; si ninguna de las divisas está alojada en TIPS, la transacción se rechaza. [{routing_ref}]\n\n"
                f"Impacto operativo: la CR introduce pantallas GUI específicas de `Advanced Cross-currency Payment transaction query` para consultar las transacciones enlazadas, el FX rate, los intermediarios y el identificador de la otra leg. [{gui_ref}] "
                f"En aplicación, añade flujos de Message Router para los roles Source CSM y Destination CSM, marcas internas para distinguir linked cross-currency transactions, un nuevo estado intermedio y colas de comunicación inter-CSM. [{impact_ref}]\n\n"
                f"Límites: la CR deja fuera los recall requests y las respuestas positivas/negativas de recall; si esos casos fueran obligatorios, la propia CR dice que haría falta otra CR. De hecho, TIPS-0090 aparece después para cubrir mejoras cross-currency que quedaron fuera de TIPS-0065. [{related_ref}]"
            )
    else:
        if language == "en":
            answer = (
                f"{code_label}: {title}. It is {'published/listed' if published else 'not found as a published CR form'} in the local catalogue"
                f"{' with date ' + date_value if date_value else ''}, and its status/release bucket is {status}. [{meta_ref}]\n\n"
                "I found the CR metadata locally, but I do not have enough curated passages to produce a deeper functional summary without using Codex generation. "
                "Switch to Codex High for a fuller synthesis from the retrieved CR pages."
            )
        else:
            answer = (
                f"{code_label}: {title}. Está {'publicada/listada' if published else 'sin ficha publicada localizada'} en el catálogo local"
                f"{' con fecha ' + date_value if date_value else ''}, y su estado/release es {status}. [{meta_ref}]\n\n"
                "Tengo la ficha local de la CR, pero no suficientes pasajes curados para hacer un resumen funcional potente sin pasar por Codex. "
                "Con Codex High puedo sintetizar las páginas recuperadas y darte una explicación mucho más desarrollada."
            )

    return {"answer": append_references(answer, citations, language), "citations": citations, "confidence": "high", "skip_generation": True}


def build_change_request_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    if not is_change_request_question(query):
        return None
    raw_entries = load_json(CHANGE_REQUESTS_PATH)
    entries = cr_entries_by_code()
    if not entries:
        return None

    wanted_codes = query_cr_codes(query)
    release = query_release(query)
    citations: list[dict[str, Any]] = [change_request_citation(1)]
    source_title = "References" if language == "en" else "Referencias"

    release_codes = scope_codes_from_hits(hits, release) if release else []
    if release and not release_codes:
        release_codes = [code for code, entry in entries.items() if entry.get("release") == release]
    selected_codes = wanted_codes or release_codes

    if selected_codes:
        selected = [
            find_cr_entry(entries, code)
            or {"code": code, "title": code, "status": "", "release": release, "published": False}
            for code in selected_codes
        ]
    else:
        selected = list(entries.values())[:12]

    summary_answer = build_change_request_summary_answer(query, selected_codes, entries, hits, language)
    if summary_answer:
        return summary_answer

    support_hits = []
    seen_support: set[tuple[str, str, str]] = set()
    for hit in hits:
        title = (hit.chunk.get("title") or "").lower()
        if hit.chunk.get("family") == "release_documentation" and (
            "cover note" in title or "main milestones" in title or "content of tips release" in title
        ):
            key = (
                str(hit.chunk.get("local_path") or ""),
                str(hit.chunk.get("unit_type") or ""),
                str(hit.chunk.get("unit") or ""),
            )
            if key not in seen_support:
                support_hits.append(hit)
                seen_support.add(key)

    def release_support_priority(hit: Hit) -> tuple[int, int, int, float]:
        title = str(hit.chunk.get("title") or "").lower()
        unit = str(hit.chunk.get("unit") or "")
        return (
            1 if "main milestones" in title else 0,
            1 if "cover note" in title and unit == "2" else 0,
            1 if "cover note" in title else 0,
            hit.score,
        )

    support_hits.sort(key=release_support_priority, reverse=True)
    for hit in support_hits[:4]:
        citations.append(
            {
                "n": len(citations) + 1,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    date_ref = next(
        (
            item["n"]
            for item in citations
            if "cover note" in str(item.get("title", "")).lower() and str(item.get("unit")) == "2"
        ),
        next(
            (
                item["n"]
                for item in citations
                if "cover note" in str(item.get("title", "")).lower()
                or "main milestones" in str(item.get("title", "")).lower()
            ),
            citations[1]["n"] if len(citations) > 1 else 1,
        ),
    )

    today = date.today().isoformat()
    if language == "en":
        if release:
            answer = f"As of {today}, the local corpus separates two things for {release}: the CR forms listed on the ECB change-request page, and the final SDD publication milestone. "
            answer += "The CR entries are published/listed on the ECB page when they have a CR form link in the structured catalogue. [1]"
        else:
            answer = f"As of {today}, the local ECB page contains {len(raw_entries)} published/listed TIPS change request entries. [1]"
        if support_hits:
            answer += f" For {release or 'the relevant release'}, the release documentation gives the publication and scope context. [2]"
        lines = ["", "Relevant CRs:"]
        for entry in selected[:16]:
            status = entry.get("status") or "no TIPS CR status shown in the ECB list"
            published = "published/listed" if entry.get("published") else "not found as a published CR form"
            lines.append(f"- {entry.get('code')}: {entry.get('title')} | {published}; status: {status}. [1]")
    else:
        if release:
            answer = f"A {today}, para {release} hay que separar dos cosas: las fichas CR publicadas en la página del ECB y la publicación final de los SDD. "
            answer += "Una CR cuenta como publicada/listada cuando aparece en el catálogo local estructurado con enlace a su ficha del ECB. [1]"
        else:
            answer = f"A {today}, la página local del ECB contiene {len(raw_entries)} entradas TIPS Change Request publicadas/listadas. [1]"
        if support_hits:
            answer += f" Para {release or 'la release consultada'}, la documentación de release aporta el alcance y las fechas de publicación de los documentos. [2]"
        lines = ["", "CRs relevantes:"]
        for entry in selected[:16]:
            status = entry.get("status") or "sin TIPS CR status en la lista del ECB"
            published = "publicada/listada" if entry.get("published") else "no encontrada como ficha CR publicada"
            lines.append(f"- {entry.get('code')}: {entry.get('title')} | {published}; estado: {status}. [1]")

    if release == "R2026.NOV":
        if language == "en":
            lines.append("")
            lines.append(f"Important nuance: R2026.NOV final clean/revised TIPS SDDs are scheduled for 2026-05-06 in the local milestones/cover note, so on 2026-04-27 that final publication date is still in the future. [{date_ref}]")
        else:
            lines.append("")
            lines.append(f"Matiz importante: los SDD finales clean/revised de R2026.NOV están previstos para el 2026-05-06 según los milestones y la cover note locales. El 2026-04-27 esa fecha todavía es futura. [{date_ref}]")

    answer += "\n".join(lines)
    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{item['n']}] {item['label']} - {item['local_path']}" for item in citations
    )
    return {"answer": answer, "citations": citations, "confidence": "high", "skip_generation": True}


def build_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any]:
    if not is_domain_query(query):
        msg = (
            "I cannot see a TIPS term, message, acronym, release or process in that question. Ask with the specific TIPS concept and I will answer from the local corpus."
            if language == "en"
            else "No veo ningun termino TIPS, mensaje, acronimo, release o proceso en esa pregunta. Pon el concepto TIPS concreto y respondo con el corpus local."
        )
        return {"answer": msg, "citations": [], "confidence": "low", "skip_generation": True}

    if not hits:
        msg = (
            "I cannot find enough evidence in the local TIPS index."
            if language == "en"
            else "No aparece evidencia suficiente en el índice local de TIPS."
        )
        return {"answer": msg, "citations": [], "confidence": "low"}

    olo_lkt_comparison_answer = build_olo_lkt_comparison_answer(query, hits, language=language)
    if olo_lkt_comparison_answer:
        return olo_lkt_comparison_answer

    u2a_functions_answer = build_u2a_functions_answer(query, hits, language=language)
    if u2a_functions_answer:
        return u2a_functions_answer

    uhb_u2a_answer = build_uhb_u2a_answer(query, hits, language=language)
    if uhb_u2a_answer:
        return uhb_u2a_answer

    source_currency_answer = build_source_currency_answer(query, hits, language=language)
    if source_currency_answer:
        return source_currency_answer

    olo_messages_answer = build_olo_messages_answer(query, hits, language=language)
    if olo_messages_answer:
        return olo_messages_answer

    olo_parties_answer = build_olo_parties_answer(query, hits, language=language)
    if olo_parties_answer:
        return olo_parties_answer

    olo_answer = build_olo_answer(query, hits, language=language)
    if olo_answer:
        return olo_answer

    lkt_answer = build_lkt_answer(query, hits, language=language)
    if lkt_answer:
        return lkt_answer

    investigation_offset_answer = build_investigation_offset_answer(query, hits, language=language)
    if investigation_offset_answer:
        return investigation_offset_answer

    investigation_answer = build_investigation_answer(query, hits, language=language)
    if investigation_answer:
        return investigation_answer

    recall_answer = build_recall_answer(query, hits, language=language)
    if recall_answer:
        return recall_answer

    cross_currency_flag_answer = build_cross_currency_flag_answer(query, hits, language=language)
    if cross_currency_flag_answer:
        return cross_currency_flag_answer

    authorised_account_user_answer = build_authorised_account_user_answer(query, hits, language=language)
    if authorised_account_user_answer:
        return authorised_account_user_answer

    currency_answer = build_currency_answer(query, hits, language=language)
    if currency_answer:
        return currency_answer

    acronym_answer = build_acronym_answer(query, hits, language=language)
    if acronym_answer:
        return acronym_answer

    change_request_answer = build_change_request_answer(query, hits, language=language)
    if change_request_answer:
        return change_request_answer

    schema_answer = build_schema_list_answer(query, hits, language=language)
    if schema_answer:
        return schema_answer

    top = hits[:4]
    if language == "en":
        intro = "I do not have enough direct evidence for a closed answer. These are the local passages I would inspect first:"
        source_title = "References"
    else:
        intro = "No tengo evidencia directa suficiente para una respuesta cerrada. Estos son los pasajes locales que revisaria primero:"
        source_title = "Referencias"

    bullets = []
    citations = []
    for n, hit in enumerate(top, start=1):
        label = cite_label(hit)
        excerpt = trim_excerpt(hit.chunk.get("text", ""))
        bullets.append(f"{n}. {excerpt} [{n}]")
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": label,
            }
        )

    answer = intro + "\n\n" + "\n\n".join(bullets)
    answer += f"\n\n{source_title}:\n" + "\n".join(
        f"[{c['n']}] {c['label']} - {c['local_path']}" for c in citations
    )
    confidence = "high" if hits[0].score >= 0.25 else "medium" if hits[0].score >= 0.12 else "low"
    return {"answer": answer, "citations": citations, "confidence": confidence}


def prioritize_generation_hits(query: str, hits: list[Hit], max_hits: int) -> list[Hit]:
    allow_revisions = bool(re.search(r"\b(revision|revised|tracked|changes|cambios|revisiones)\b", query, re.I))
    hits = rerank_context_hits(query, hits, max_total=max(len(hits), max_hits))
    preferred: list[Hit] = []
    if is_change_request_question(query):
        wanted_codes = query_cr_codes(query)
        for code in wanted_codes:
            aliases = _cr_code_aliases(code)
            exact_cr_hits = [
                hit
                for hit in hits
                if any(
                    alias
                    in " ".join(
                        [
                            str(hit.chunk.get("title") or ""),
                            str(hit.chunk.get("unit_type") or ""),
                            str(hit.chunk.get("unit") or ""),
                            str(hit.chunk.get("local_path") or ""),
                        ]
                    ).upper()
                    for alias in aliases
                )
            ]
            for hit in [
                find_hit(exact_cr_hits, code),
                find_hit(exact_cr_hits, "reason for change"),
                find_hit(exact_cr_hits, "expected benefits"),
                find_hit(exact_cr_hits, "description of requested"),
                find_hit(exact_cr_hits, "instant payment transaction steps"),
                find_hit(exact_cr_hits, "mapping table"),
                find_hit(exact_cr_hits, "summary of application development impact"),
                find_hit(exact_cr_hits, "detailed assessment"),
                find_hit(exact_cr_hits, "out-of-scope"),
                find_hit(hits, code, "out-of-scope"),
            ]:
                if hit:
                    preferred.append(hit)
    if is_olo_question(query):
        for hit in [
            find_hit(hits, "one-leg out instant credit transfer", "designed and implemented in tips"),
            find_hit(hits, "two separate and independent instant payments", "source and destination leg"),
            find_hit(hits, "olo one-leg out"),
            find_hit(hits, "the involved actors are", "originator participant", "beneficiary participant"),
            find_hit(hits, "leg exit psp acts as beneficiary", "leg entry psp acts as originator"),
            find_hit(hits, "incoming olo cross-currency", "outgoing olo cross-currency"),
            find_hit(hits, "corridor shows", "olo"),
        ]:
            if hit:
                preferred.append(hit)
    if is_lkt_question(query) or is_olo_lkt_comparison_question(query):
        for hit in [
            find_hit(hits, "cross-currency linked-transactions", "preferred settlement model"),
            find_hit(hits, "linked payment message", "settle simultaneously"),
            find_hit(hits, "enhanced lkt settlement model", "all transactions are settled or none"),
            find_hit(hits, "both currencies", "pair is configured in the mapping table"),
        ]:
            if hit:
                preferred.append(hit)

    selected: list[Hit] = []
    seen_ids: set[int] = set()
    seen_texts: set[str] = set()

    def add(hit: Hit) -> None:
        if len(selected) >= max_hits or id(hit) in seen_ids:
            return
        title = (hit.chunk.get("title") or "").lower()
        revision_status = (hit.chunk.get("revision_status") or "").lower()
        if not allow_revisions and ("with revisions" in title or revision_status in {"rev", "revised"}):
            return
        text_key = re.sub(r"\W+", " ", hit.chunk.get("text", "").lower())[:500]
        if text_key in seen_texts:
            return
        selected.append(hit)
        seen_ids.add(id(hit))
        if text_key:
            seen_texts.add(text_key)

    for hit in preferred:
        add(hit)
    for hit in hits:
        add(hit)
    return selected


def build_codex_context(query: str, hits: list[Hit], max_hits: int = GENERATION_CONTEXT_HITS) -> dict[str, Any]:
    hits = prioritize_generation_hits(query, hits, max_hits=max_hits)
    instructions = (
        "Answer the user's TIPS question using the local evidence dossier below. "
        "Cite every substantive claim with [n]. If evidence is weak or absent, say so. "
        "Treat the retrieved chunks as pointers to the right documentation, not as a pre-written answer. "
        "Prefer official clean ECB/TIPS/MyStandards evidence over revised or older duplicates when the question does not specify a version. "
        "Use neighboring chunks only to complete local context; do not let them override a stronger direct hit."
    )
    evidence = []
    for n, hit in enumerate(hits[:max_hits], start=1):
        evidence.append(
            {
                "ref": n,
                "score": round(hit.score, 4),
                "retrieval_reason": hit.reason or "hybrid",
                "citation": cite_label(hit),
                "title": hit.chunk.get("title"),
                "family": hit.chunk.get("family"),
                "release": hit.chunk.get("release"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=3000),
            }
        )
    return {
        "question": query,
        "retrieval_pipeline": "hybrid TF-IDF word + TF-IDF char + BM25 + metadata boosts + local rerank + neighbor expansion",
        "instructions_for_codex_high": instructions,
        "evidence": evidence,
    }


def format_chat_history(chat_history: list[dict[str, str]] | None, language: str) -> str:
    if not chat_history:
        return ""
    labels = {
        "user": "Usuario" if language == "es" else "User",
        "assistant": "Asistente" if language == "es" else "Assistant",
    }
    lines: list[str] = []
    for turn in chat_history[-10:]:
        role = str(turn.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()
        if content:
            lines.append(f"{labels[role]}: {content[:1600]}")
    if not lines:
        return ""
    title = (
        "Conversacion reciente, solo para resolver referencias de seguimiento:"
        if language == "es"
        else "Recent conversation, only to resolve follow-up references:"
    )
    return title + "\n" + "\n".join(lines)


def build_generation_prompt(
    query: str,
    hits: list[Hit],
    language: str = "es",
    max_hits: int = GENERATION_CONTEXT_HITS,
    chat_history: list[dict[str, str]] | None = None,
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if language == "auto":
        language = detect_question_language(query)
    lang_name = "Spanish" if language == "es" else "English"
    ranking_query = context_query or query
    hits = prioritize_generation_hits(ranking_query, hits, max_hits=max_hits)
    context = build_codex_context(ranking_query, hits, max_hits=max_hits)
    evidence_lines = []
    for item in context["evidence"]:
        evidence_lines.append(
            "\n".join(
                [
                    f"[{item['ref']}] {item['citation']}",
                    f"Retrieval: {item.get('retrieval_reason', 'hybrid')} | score={item.get('score')}",
                    f"Local path: {item['local_path']}",
                    f"Source URL: {item['source_url']}",
                    f"Excerpt: {item['excerpt']}",
                ]
            )
        )
    history_block = format_chat_history(chat_history, language)
    draft_block = ""
    if draft_answer:
        draft_block = (
            "Structured local draft. Use it only as a safety rail. Do not copy its brevity; improve it substantially when it is too short, too list-like, or too shallow:\n"
            + re.sub(r"\s+", " ", draft_answer).strip()[:5000]
            + "\n\n"
        )
    if language == "es":
        style_rules = """Write in Spanish with a natural human voice, following the local Doodly/escriturahumana rules.
Start with the answer, then develop it. Do not stop at a short template when the user asks for a summary, CR, explanation, comparison, flow, impact, "detallado" or "paso a paso".
For substantive TIPS questions, prefer a rich answer with useful sections: respuesta corta, que cambia, como funciona, impacto operativo, limites/exclusiones and references.
Target 600-1200 words for summaries, CR explanations, flows and comparisons unless the user explicitly asks for a short answer.
For follow-up requests such as "mas detalle", "hazme un esquema", "redactalo" or "paso a paso", keep the previous answer in mind and reshape or deepen it instead of restarting from scratch.
Use complete Spanish: clear subjects, conjugated verbs, needed articles, concrete nouns and explicit causality.
Do not write like a ficha, a search result or a report index. Each sentence must answer, connect, prove, contrast or close.
Do not use generic second person. Do not use filler such as "cabe destacar", "en este sentido", "es importante mencionar" or "de alguna manera".
Do not say "evidencia proporcionada", "prompt" or "contexto". If something is missing, say "No aparece en la documentación local recuperada".
Keep official TIPS role, message and schema names in English when those are the names used by the documentation."""
    else:
        style_rules = """Write in English with a natural, direct technical-assistant voice.
Start with the answer, then develop it. Do not stop at a short template when the user asks for a summary, CR, explanation, comparison, flow, impact or "detailed".
For substantive TIPS questions, prefer a rich answer with useful sections: short answer, what changes, how it works, operational impact, limits/exclusions and references.
Target 600-1200 words for summaries, CR explanations, flows and comparisons unless the user explicitly asks for a short answer.
For follow-up requests such as "more detail", "make an outline", "rewrite it" or "step by step", keep the previous answer in mind and reshape or deepen it instead of restarting from scratch.
Do not write like a search-result list. Each sentence must answer, connect, prove, contrast or close.
Do not say "provided evidence", "prompt" or "context". If something is missing, say "I cannot find it in the retrieved local documentation".
Keep official TIPS role, message and schema names in English."""
    return f"""You are a senior TIPS documentation assistant running inside a local TIPS documentation repository.

Answer the user's question in {lang_name}, naturally and directly, like a high-quality ChatGPT answer.
Use only the local TIPS corpus: the evidence dossier below and the listed local paths as read-only pointers. Do not invent facts and do not use the internet.
The retrieval layer is deliberately generous. Treat it as a set of pointers to the right documents, then use your own reasoning to connect the facts, resolve follow-up references and produce the best answer.
When the excerpt is not enough and a listed local path is clearly relevant, inspect that local document or extracted text with read-only commands before answering.
If an excerpt is not enough but its pointer is clearly relevant, rely on the title/unit/path metadata and say only what the cited local evidence supports.
Do not dump raw excerpts. Synthesize the answer.
Every substantive claim must cite evidence with [n]. Put references at the end with title, page/unit, and local path.
Avoid saying "the strongest references are"; answer the question first.
Treat evidence as ranked but not infallible: prefer direct hits over neighbor hits, clean releases over revision duplicates, and official documents over inferred glossary entries.
{style_rules}

User question:
{query}

{history_block + chr(10) if history_block else ""}Local evidence dossier:

{draft_block}
{chr(10).join(evidence_lines)}
"""


def generate_with_codex(
    query: str,
    hits: list[Hit],
    language: str = "es",
    timeout: int | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if os.environ.get("TIPS_DISABLE_CODEX", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Codex generation disabled by TIPS_DISABLE_CODEX")
    if not shutil.which("codex.cmd") and not shutil.which("codex"):
        raise RuntimeError("codex CLI not found")
    prompt = build_generation_prompt(
        query,
        hits,
        language=language,
        max_hits=GENERATION_CONTEXT_HITS,
        chat_history=chat_history,
        draft_answer=draft_answer,
        context_query=context_query,
    )
    timeout = timeout or int(os.environ.get("TIPS_CODEX_TIMEOUT", "180"))
    preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])
    reasoning = preset["reasoning"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        output_path = Path(tmp.name)
    cmd = [
        CODEX,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "read-only",
        "--color",
        "never",
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-o",
        str(output_path),
        "-",
    ]
    model = os.environ.get("TIPS_CODEX_MODEL", "").strip()
    if model:
        cmd[2:2] = ["-m", model]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "").strip()[:1200])
        answer = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not answer:
            raise RuntimeError("codex returned an empty answer")
        return answer
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def answer_question(
    query: str,
    top_k: int = 16,
    language: str = "auto",
    generate: bool = False,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
) -> dict[str, Any]:
    resolved_language = detect_question_language(query) if language == "auto" else language
    index = load_index()
    search_query = retrieval_query or query
    if generate:
        retrieval_k = max(top_k, GENERATION_RETRIEVAL_HITS)
    elif is_olo_question(search_query) or is_lkt_question(search_query) or is_change_request_question(search_query):
        retrieval_k = max(top_k, 24)
    else:
        retrieval_k = top_k
    hits = retrieve(index, search_query, top_k=retrieval_k)
    hits = augment_hits(index, search_query, hits)
    max_context = max(retrieval_k + 16, top_k, GENERATION_CONTEXT_HITS if generate else top_k)
    hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(max_context, MAX_CONTEXT_HITS))
    hits = rerank_context_hits(search_query, hits, max_total=min(max_context, MAX_CONTEXT_HITS))
    payload = build_answer(query, hits, language=resolved_language)
    if model_preset == "local_rag":
        generate = False
    if generate and hits and payload.get("confidence") != "low":
        draft = payload.get("answer") if payload.get("skip_generation") else None
        generation_hits = prioritize_generation_hits(query, hits, max_hits=GENERATION_CONTEXT_HITS)
        try:
            payload["answer"] = generate_with_codex(
                query,
                generation_hits,
                language=resolved_language,
                chat_history=chat_history,
                model_preset=model_preset,
                draft_answer=draft,
                context_query=query,
            )
            payload["citations"] = citations_from_hits(generation_hits[:GENERATION_CONTEXT_HITS])
            payload["generated_by"] = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])["label"].lower().replace(" ", "_")
            payload.pop("skip_generation", None)
        except Exception as exc:
            payload["generated_by"] = "fallback_extractivo"
            payload["generator_error"] = str(exc)
    elif payload.get("skip_generation"):
        payload["generated_by"] = "structured"
    elif model_preset == "local_rag":
        payload["generated_by"] = "local_rag"
    payload["question"] = query
    payload["language"] = resolved_language
    payload["model"] = model_preset
    payload["hits"] = [
        {
            "rank": hit.rank,
            "score": round(hit.score, 4),
            "reason": hit.reason,
            "citation": cite_label(hit),
            "chunk": {
                key: hit.chunk.get(key)
                for key in [
                    "chunk_id",
                    "doc_id",
                    "title",
                    "category",
                    "family",
                    "release",
                    "revision_status",
                    "unit_type",
                    "unit",
                    "local_path",
                    "source_url",
                    "context_path",
                ]
            },
            "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=900),
        }
        for hit in hits
    ]
    return payload


def read_question(args: argparse.Namespace) -> str:
    if args.question:
        return " ".join(args.question).strip()
    data = sys.stdin.read()
    return data.strip()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Ask questions against the local TIPS index.")
    parser.add_argument("question", nargs="*", help="question; if omitted, stdin is used")
    parser.add_argument("--json", action="store_true", help="print full JSON answer")
    parser.add_argument("--context", action="store_true", help="print optimized evidence JSON for Codex High")
    parser.add_argument("--generate", action="store_true", help="generate a conversational answer with Codex High")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lang", choices=["auto", "es", "en"], default="auto")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="codex_high", help="generation preset")
    args = parser.parse_args(argv)

    query = read_question(args)
    if not query:
        print("ERROR: empty question", file=sys.stderr)
        return 2

    try:
        index = load_index()
        if args.generate:
            retrieval_k = max(args.top_k, GENERATION_RETRIEVAL_HITS)
        elif is_olo_question(query) or is_lkt_question(query) or is_change_request_question(query):
            retrieval_k = max(args.top_k, 24)
        else:
            retrieval_k = args.top_k
        hits = retrieve(index, query, top_k=retrieval_k)
        hits = augment_hits(index, query, hits)
        max_context = max(retrieval_k + 16, args.top_k, GENERATION_CONTEXT_HITS if args.generate else args.top_k)
        hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(max_context, MAX_CONTEXT_HITS))
        hits = rerank_context_hits(query, hits, max_total=min(max_context, MAX_CONTEXT_HITS))
        language = detect_question_language(query) if args.lang == "auto" else args.lang
        if args.context:
            print(json.dumps(build_codex_context(query, hits, max_hits=args.top_k), ensure_ascii=False, indent=2))
        else:
            payload = build_answer(query, hits, language=language)
            if args.model == "local_rag":
                args.generate = False
            if args.generate and hits and payload.get("confidence") != "low":
                draft = payload.get("answer") if payload.get("skip_generation") else None
                generation_hits = prioritize_generation_hits(query, hits, max_hits=GENERATION_CONTEXT_HITS)
                try:
                    payload["answer"] = generate_with_codex(
                        query,
                        generation_hits,
                        language=language,
                        model_preset=args.model,
                        draft_answer=draft,
                        context_query=query,
                    )
                    payload["citations"] = citations_from_hits(generation_hits[:GENERATION_CONTEXT_HITS])
                    payload["generated_by"] = MODEL_PRESETS.get(args.model, MODEL_PRESETS["codex_high"])["label"].lower().replace(" ", "_")
                    payload.pop("skip_generation", None)
                except Exception as exc:
                    payload["generated_by"] = "fallback_extractivo"
                    payload["generator_error"] = str(exc)
            elif payload.get("skip_generation"):
                payload["generated_by"] = "structured"
            elif args.model == "local_rag":
                payload["generated_by"] = "local_rag"
            payload["question"] = query
            payload["language"] = language
            payload["model"] = args.model
            if args.json:
                payload["hits"] = [
                    {
                        "rank": hit.rank,
                        "score": round(hit.score, 4),
                        "reason": hit.reason,
                        "citation": cite_label(hit),
                        "local_path": hit.chunk.get("local_path"),
                        "source_url": hit.chunk.get("source_url"),
                        "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=1100),
                    }
                    for hit in hits
                ]
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(payload["answer"])
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        language = detect_question_language(query) if args.lang == "auto" else args.lang
        answer = (
            "I could not complete this query, but the CLI stayed alive. Try a more specific TIPS term or run with --context."
            if language == "en"
            else "No he podido completar esta consulta, pero el CLI sigue vivo. Prueba con un termino TIPS mas concreto o ejecuta con --context."
        )
        if args.json:
            print(json.dumps({"question": query, "answer": answer, "citations": [], "confidence": "low", "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(answer)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
