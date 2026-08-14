from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.ocr.ocr_selector import select_best_ocr_for_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)

    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        items = json.load(f)

    updated = select_best_ocr_for_items(items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_csv(updated, Path(args.csv))

    easy_selected = sum(
        1
        for item in updated
        if ((item.get("ocr") or {}).get("selected") or {}).get("engine") == "easyocr"
    )

    print(f"OK: items={len(updated)} easy_selected={easy_selected}")
    print(f"Wrote: {output_path}")


def _write_csv(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "label",
                "selected_engine",
                "selection_reason",
                "confidence",
                "text_preview",
            ],
        )
        writer.writeheader()

        for item in sorted(items, key=_sort_key):
            selected = (item.get("ocr") or {}).get("selected") or {}
            writer.writerow(
                {
                    "page": item.get("page"),
                    "region_id": item.get("region_id"),
                    "label": item.get("label"),
                    "selected_engine": selected.get("engine"),
                    "selection_reason": selected.get("selection_reason"),
                    "confidence": selected.get("confidence"),
                    "text_preview": str(selected.get("text") or "").replace("\n", " / ")[:300],
                }
            )


def _sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
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