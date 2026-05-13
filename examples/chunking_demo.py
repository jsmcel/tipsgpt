#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tips_ingest import Document, build_chunks, split_text  # noqa: E402


def synthetic_unit_text() -> str:
    paragraphs = [
        (
            "Section overview. This synthetic document describes a local payment "
            "operations feature. It has paragraphs, headings and enough repeated "
            "detail to show how the production chunker groups text."
        ),
        (
            "Operational context. The source account sends funds to the destination "
            "account. The platform validates identifiers, checks balances, applies "
            "business rules and records a settlement result."
        ),
        (
            "Message handling. A request is received, validated and routed. A status "
            "response is produced after the operation reaches a terminal state."
        ),
        (
            "User context. Operators can inspect status, review references and use "
            "the surrounding document path to understand where the evidence came from."
        ),
    ]
    return "\n\n".join(paragraphs * 8)


def build_synthetic_document() -> Document:
    return Document(
        id="synthetic-payment-guide",
        title="Synthetic Payment Operations Guide",
        url="https://example.invalid/synthetic-payment-guide",
        ext=".html",
        category="tips_udfs",
        family="tips_udfs",
        release="R2026.NOV",
        revision_status="clean",
        contexts=[
            {
                "path": [
                    "Synthetic documentation",
                    "Payment operations",
                    "Settlement status",
                ]
            }
        ],
        status="downloaded",
        local_path="synthetic/payment-operations-guide.html",
        source_host="example.invalid",
    )


def main() -> int:
    text = synthetic_unit_text()
    unit = {"unit_type": "section", "unit": "2.1", "text": text}
    doc = build_synthetic_document()

    preview_pieces = split_text(text, max_chars=420, overlap=80)
    production_chunks = build_chunks([doc], {doc.id: [unit]})

    print("Readable split preview with smaller limits:")
    for index, piece in enumerate(preview_pieces[:4], start=1):
        compact = " ".join(piece.split())
        print(f"- preview_piece={index} chars={len(piece)} text={compact[:170]}...")

    print("\nProduction-shaped chunks using build_chunks():")
    for chunk in production_chunks:
        print(
            f"- chunk_id={chunk['chunk_id']} chars={len(chunk['text'])} "
            f"family={chunk['family']} release={chunk['release']} "
            f"unit={chunk['unit_type']} {chunk['unit']}"
        )

    print("\nFirst production chunk JSON:")
    print(json.dumps(production_chunks[0], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
