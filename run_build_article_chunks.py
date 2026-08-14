from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.chunking.article_detector import enrich_items_with_article_signals
from ocr_chunker.chunking.article_chunker import build_article_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--signals-output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--inline-reference-text", action="store_true")

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    enriched = enrich_items_with_article_signals(items)
    chunks = build_article_chunks(
        enriched,
        inline_reference_text=args.inline_reference_text,
    )

    Path(args.signals_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.signals_output).write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_csv(chunks, Path(args.csv))

    article_chunks = sum(1 for chunk in chunks if chunk.get("chunk_type") == "article")
    refs = sum(len(chunk.get("references") or []) for chunk in chunks)

    print(f"OK: chunks={len(chunks)} article_chunks={article_chunks} references={refs}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.signals_output}")
    print(f"Wrote: {args.csv}")


def _write_csv(chunks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "chunk_type",
                "article_number",
                "page_start",
                "page_end",
                "book",
                "bab",
                "fasl",
                "box_ids",
                "references",
                "needs_review",
                "review_reasons",
                "text_preview",
            ],
        )
        writer.writeheader()

        for chunk in chunks:
            writer.writerow(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "article_number": chunk.get("article_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "book": chunk.get("book"),
                    "bab": chunk.get("bab"),
                    "fasl": chunk.get("fasl"),
                    "box_ids": "|".join(str(v) for v in chunk.get("box_ids") or []),
                    "references": "|".join(
                        f"{ref.get('article_number')}->{ref.get('target_chunk_id')}"
                        for ref in chunk.get("references") or []
                    ),
                    "needs_review": chunk.get("needs_review"),
                    "review_reasons": "|".join(chunk.get("review_reasons") or []),
                    "text_preview": str(chunk.get("text") or "").replace("\n", " / ")[:500],
                }
            )


if __name__ == "__main__":
    main()