#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tips_ask import (  # noqa: E402
    build_codex_context,
    expand_neighbor_hits,
    rerank_context_hits,
    retrieve,
)


def demo_chunks() -> list[dict[str, Any]]:
    """Return invented chunks that exercise the selector without real documents."""
    return [
        {
            "chunk_id": "demo-liquidity:section:1",
            "doc_id": "demo-liquidity",
            "title": "Synthetic Liquidity Guide",
            "category": "tips_udfs",
            "family": "tips_udfs",
            "release": "R2026.NOV",
            "revision_status": "clean",
            "unit_type": "section",
            "unit": "1",
            "context_path": ["Synthetic guide", "Liquidity transfer"],
            "local_path": "synthetic/demo-liquidity.md",
            "source_url": "",
            "text": (
                "This synthetic section explains a liquidity transfer between two "
                "accounts. It mentions debtor account, creditor account, settlement "
                "and the operational steps used to move funds."
            ),
        },
        {
            "chunk_id": "demo-liquidity:section:2",
            "doc_id": "demo-liquidity",
            "title": "Synthetic Liquidity Guide",
            "category": "tips_udfs",
            "family": "tips_udfs",
            "release": "R2026.NOV",
            "revision_status": "clean",
            "unit_type": "section",
            "unit": "2",
            "context_path": ["Synthetic guide", "Liquidity transfer", "Neighbor detail"],
            "local_path": "synthetic/demo-liquidity.md",
            "source_url": "",
            "text": (
                "This neighboring synthetic chunk adds context about validation, "
                "status feedback and account balance updates after settlement."
            ),
        },
        {
            "chunk_id": "demo-pricing:section:1",
            "doc_id": "demo-pricing",
            "title": "Synthetic Pricing Note",
            "category": "pricing",
            "family": "pricing",
            "release": "R2026.NOV",
            "revision_status": "clean",
            "unit_type": "section",
            "unit": "1",
            "context_path": ["Synthetic note", "Pricing and fees"],
            "local_path": "synthetic/demo-pricing.md",
            "source_url": "",
            "text": (
                "This synthetic pricing note discusses fees, billing, tariff items "
                "and monthly cost allocation. It is designed to win pricing queries."
            ),
        },
        {
            "chunk_id": "demo-connectivity:section:1",
            "doc_id": "demo-connectivity",
            "title": "Synthetic Connectivity Note",
            "category": "connectivity",
            "family": "connectivity",
            "release": "R2025.NOV",
            "revision_status": "clean",
            "unit_type": "section",
            "unit": "1",
            "context_path": ["Synthetic note", "Connectivity"],
            "local_path": "synthetic/demo-connectivity.md",
            "source_url": "",
            "text": (
                "This synthetic connectivity note mentions network service provider, "
                "message exchange, technical requirements, endpoint setup and secure "
                "connectivity checks."
            ),
        },
        {
            "chunk_id": "demo-currency:section:1",
            "doc_id": "demo-currency",
            "title": "Synthetic Currency Note",
            "category": "participation",
            "family": "participation",
            "release": "R2026.NOV",
            "revision_status": "clean",
            "unit_type": "section",
            "unit": "1",
            "context_path": ["Synthetic note", "Currency support"],
            "local_path": "synthetic/demo-currency.md",
            "source_url": "",
            "text": (
                "This synthetic note talks about currency support and settlement "
                "configuration. It includes example tokens EUR, SEK and DKK."
            ),
        },
    ]


def build_demo_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    search_texts = [
        "\n".join(
            [
                chunk.get("title", ""),
                chunk.get("family", ""),
                chunk.get("category", ""),
                chunk.get("release", ""),
                " > ".join(chunk.get("context_path", [])),
                chunk.get("text", ""),
            ]
        )
        for chunk in chunks
    ]
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w./-]{2,}\b",
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    bm25_vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[\w./-]{2,}\b",
    )
    word_matrix = word_vectorizer.fit_transform(search_texts)
    char_matrix = char_vectorizer.fit_transform(search_texts)
    bm25_matrix = bm25_vectorizer.fit_transform(search_texts).tocsr()
    bm25_doc_len = np.asarray(bm25_matrix.sum(axis=1)).ravel().astype(float)
    bm25_df = np.asarray((bm25_matrix > 0).sum(axis=0)).ravel().astype(float)
    bm25_idf = np.log(((len(chunks) - bm25_df + 0.5) / (bm25_df + 0.5)) + 1.0)
    return {
        "built_at": "synthetic-demo",
        "index_url": "",
        "docs": [],
        "chunks": chunks,
        "chunk_id_to_pos": {
            chunk["chunk_id"]: pos for pos, chunk in enumerate(chunks)
        },
        "word_vectorizer": word_vectorizer,
        "word_matrix": word_matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
        "bm25_vectorizer": bm25_vectorizer,
        "bm25_matrix": bm25_matrix,
        "bm25_idf": bm25_idf,
        "bm25_doc_len": bm25_doc_len,
        "bm25_avgdl": float(bm25_doc_len.mean() or 1.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show how TIPS GPT selects information using synthetic chunks."
    )
    parser.add_argument("query", nargs="*", default=["how", "does", "liquidity", "transfer", "work"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    query = " ".join(args.query).strip() or "how does liquidity transfer work"
    index = build_demo_index(demo_chunks())
    hits = retrieve(index, query, top_k=max(args.top_k, 5), pool=20)
    hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=8)
    hits = rerank_context_hits(query, hits, max_total=args.top_k)
    dossier = build_codex_context(query, hits, max_hits=args.top_k)

    print(f"Query: {query}\n")
    print("Selected chunks:")
    for hit in hits[: args.top_k]:
        chunk = hit.chunk
        excerpt = " ".join(chunk.get("text", "").split())[:180]
        print(
            f"- score={hit.score:.4f} family={chunk.get('family')} "
            f"title={chunk.get('title')} unit={chunk.get('unit')}"
        )
        print(f"  {excerpt}")

    print("\nEvidence dossier:")
    print(json.dumps(dossier, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
