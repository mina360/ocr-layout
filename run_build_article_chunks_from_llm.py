from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.chunking.article_chunker_from_llm import build_article_chunks_from_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--inline-reference-text", action="store_true")

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    chunks = build_article_chunks_from_llm(
        items,
        inline_reference_text=args.inline_reference_text,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(chunks, Path(args.csv))

    article_count = sum(1 for chunk in chunks if chunk.get("chunk_type") == "article")
    review_count = sum(1 for chunk in chunks if chunk.get("needs_review"))

    print(f"OK: chunks={len(chunks)} article_chunks={article_count} review_chunks={review_count}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {args.csv}")


def write_csv(chunks: list[dict[str, Any]], path: Path) -> None:
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
                "document_title",
                "book_title",
                "bab_title",
                "fasl_title",
                "topic_title",
                "box_count",
                "metadata_box_count",
                "skipped_duplicate_box_count",
                "references",
                "needs_review",
                "review_reasons",
                "text_preview",
            ],
        )
        writer.writeheader()

        for chunk in chunks:
            metadata = chunk.get("metadata") or {}

            writer.writerow(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "article_number": chunk.get("article_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "document_title": metadata.get("document_title"),
                    "book_title": metadata.get("book_title"),
                    "bab_title": metadata.get("bab_title"),
                    "fasl_title": metadata.get("fasl_title"),
                    "topic_title": metadata.get("topic_title"),
                    "box_count": metadata.get("box_count"),
                    "metadata_box_count": metadata.get("metadata_box_count"),
                    "skipped_duplicate_box_count": metadata.get("skipped_duplicate_box_count"),
                    "references": "|".join(
                        str(ref.get("article_number"))
                        for ref in chunk.get("references") or []
                    ),
                    "needs_review": chunk.get("needs_review"),
                    "review_reasons": "|".join(chunk.get("review_reasons") or []),
                    "text_preview": str(chunk.get("text") or "").replace("\n", " / ")[:500],
                }
            )


if __name__ == "__main__":
    main()