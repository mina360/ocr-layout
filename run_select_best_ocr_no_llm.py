from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.ocr.ocr_selector_no_llm import select_best_ocr_no_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.75)

    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        items = json.load(f)

    updated = select_best_ocr_no_llm(
        items,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(updated, Path(args.csv))

    counts: dict[str, int] = {}

    for item in updated:
        selected = (item.get("ocr") or {}).get("selected") or {}
        engine = str(selected.get("engine") or "missing")
        counts[engine] = counts.get(engine, 0) + 1

    print(f"OK: items={len(updated)}")
    print(f"Selected engines: {counts}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {args.csv}")


def write_csv(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "label",
                "selected_engine",
                "confidence",
                "selection_reason",
                "crop_path",
                "selected_text",
            ],
        )
        writer.writeheader()

        for item in sorted(items, key=sort_key):
            selected = (item.get("ocr") or {}).get("selected") or {}

            writer.writerow(
                {
                    "page": item.get("page"),
                    "region_id": item.get("region_id"),
                    "label": item.get("label"),
                    "selected_engine": selected.get("engine"),
                    "confidence": selected.get("confidence"),
                    "selection_reason": selected.get("selection_reason"),
                    "crop_path": item.get("crop_path"),
                    "selected_text": str(selected.get("text") or "").replace("\n", " / ")[:400],
                }
            )


def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    page = str(item.get("page") or "")
    digits = "".join(ch for ch in page if ch.isdigit())
    page_number = int(digits) if digits else 0

    return (
        page_number,
        int(item.get("reading_order_start") or 0),
        str(item.get("region_id") or ""),
    )


if __name__ == "__main__":
    main()