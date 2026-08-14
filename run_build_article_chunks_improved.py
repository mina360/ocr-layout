from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.chunking.article_chunker_improved import build_article_chunks_improved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    chunks = build_article_chunks_improved(items)

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
                "article_number_raw",
                "article_number_resolution",
                "page_start",
                "page_end",
                "heading_box_id",
                "box_count",
                "body_box_count",
                "metadata_box_count",
                "skipped_duplicate_box_count",
                "context_blocks",
                "references",
                "needs_review",
                "review_reasons",
                "text_preview",
            ],
        )
        writer.writeheader()

        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            context_blocks = metadata.get("context_blocks") or []

            writer.writerow(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "article_number": chunk.get("article_number"),
                    "article_number_raw": metadata.get("article_number_raw"),
                    "article_number_resolution": metadata.get("article_number_resolution"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "heading_box_id": chunk.get("heading_box_id"),
                    "box_count": metadata.get("box_count"),
                    "body_box_count": metadata.get("body_box_count"),
                    "metadata_box_count": metadata.get("metadata_box_count"),
                    "skipped_duplicate_box_count": metadata.get("skipped_duplicate_box_count"),
                    "context_blocks": " || ".join(str(block.get("text") or "") for block in context_blocks),
                    "references": "|".join(str(ref.get("article_number")) for ref in chunk.get("references") or []),
                    "needs_review": chunk.get("needs_review"),
                    "review_reasons": "|".join(chunk.get("review_reasons") or []),
                    "text_preview": str(chunk.get("text") or "").replace("\n", " / ")[:700],
                }
            )


if __name__ == "__main__":
    main()