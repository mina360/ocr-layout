from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-text", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    all_text_path = Path(args.all_text)
    csv_path = Path(args.csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_text_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    cleaned_items: list[dict[str, Any]] = []

    for item in items:
        item = dict(item)
        clean = extract_clean_paddle_text(item)

        item.setdefault("ocr", {})
        item["ocr"]["selected"] = {
            "engine": "paddleocr_colab_rec_texts",
            "text": clean["text"],
            "confidence": clean["confidence"],
            "status": "cleaned_from_rec_texts",
            "lines": clean["lines"],
            "low_confidence_lines": clean["low_confidence_lines"],
            "empty": clean["empty"],
        }

        if clean["empty"]:
            item["needs_review"] = True
            item.setdefault("review_reasons", [])
            item["review_reasons"] = _append_unique(
                item["review_reasons"],
                "empty_clean_ocr_text",
            )

        if clean["confidence"] is not None and clean["confidence"] < args.low_confidence_threshold:
            item["needs_review"] = True
            item.setdefault("review_reasons", [])
            item["review_reasons"] = _append_unique(
                item["review_reasons"],
                "low_clean_ocr_confidence",
            )

        cleaned_items.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_items, f, ensure_ascii=False, indent=2)

    write_all_text(cleaned_items, all_text_path)
    write_debug_csv(cleaned_items, csv_path)

    total = len(cleaned_items)
    empty = sum(1 for item in cleaned_items if item["ocr"]["selected"]["empty"])
    low = sum(
        1
        for item in cleaned_items
        if item["ocr"]["selected"]["confidence"] is not None
        and item["ocr"]["selected"]["confidence"] < args.low_confidence_threshold
    )

    print(f"OK: regions={total} empty={empty} low_confidence={low}")
    print(f"Wrote cleaned manifest: {output_path}")
    print(f"Wrote cleaned text: {all_text_path}")
    print(f"Wrote cleaned csv: {csv_path}")


def extract_clean_paddle_text(item: dict[str, Any]) -> dict[str, Any]:
    paddle = (item.get("ocr") or {}).get("paddle") or {}
    raw = paddle.get("raw") or {}

    lines: list[dict[str, Any]] = []

    for block in raw.get("raw") or []:
        if not isinstance(block, dict):
            continue

        res = block.get("res") or {}
        rec_texts = res.get("rec_texts") or []
        rec_scores = res.get("rec_scores") or []

        for index, text in enumerate(rec_texts):
            if not isinstance(text, str):
                continue

            cleaned_text = normalize_line(text)

            if not cleaned_text:
                continue

            score = None
            if index < len(rec_scores):
                score = safe_float(rec_scores[index])

            lines.append(
                {
                    "text": cleaned_text,
                    "confidence": score,
                }
            )

    lines = remove_consecutive_duplicate_lines(lines)

    text = "\n".join(line["text"] for line in lines)

    scores = [
        line["confidence"]
        for line in lines
        if isinstance(line.get("confidence"), (int, float))
    ]

    confidence = round(sum(scores) / len(scores), 4) if scores else None

    low_confidence_lines = [
        line
        for line in lines
        if isinstance(line.get("confidence"), (int, float))
        and line["confidence"] < 0.60
    ]

    return {
        "text": text,
        "confidence": confidence,
        "lines": lines,
        "low_confidence_lines": low_confidence_lines,
        "empty": not bool(text.strip()),
    }


def normalize_line(text: str) -> str:
    text = text.strip()

    if not text:
        return ""

    # Remove known non-OCR metadata that came from the old parser.
    blocked_values = {
        "v3_predict",
        "min",
        "general",
    }

    if text in blocked_values:
        return ""

    if text.startswith("/tmp/"):
        return ""

    return " ".join(text.split())


def remove_consecutive_duplicate_lines(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_text: str | None = None

    for line in lines:
        current_text = line["text"]

        if current_text == previous_text:
            continue

        output.append(line)
        previous_text = current_text

    return output


def write_all_text(items: list[dict[str, Any]], path: Path) -> None:
    pages: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        page = str(item.get("page") or "unknown_page")
        pages.setdefault(page, []).append(item)

    for page_items in pages.values():
        page_items.sort(
            key=lambda item: (
                int(item.get("reading_order_start") or 0),
                int(item.get("reading_order_end") or 0),
                str(item.get("region_id") or ""),
            )
        )

    with path.open("w", encoding="utf-8") as f:
        for page in sorted(pages.keys(), key=page_sort_key):
            f.write("\n\n")
            f.write("#" * 90)
            f.write(f"\n# {page}\n")
            f.write("#" * 90)
            f.write("\n\n")

            for item in pages[page]:
                selected = item["ocr"]["selected"]
                f.write(f"## {item.get('region_id')}\n")
                f.write(f"confidence={selected.get('confidence')}\n")
                f.write(selected.get("text") or "[EMPTY OCR TEXT]")
                f.write("\n\n")


def write_debug_csv(items: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "reading_order_start",
                "reading_order_end",
                "confidence",
                "empty",
                "needs_review",
                "review_reasons",
                "low_confidence_line_count",
                "crop_path",
                "text_preview",
            ],
        )
        writer.writeheader()

        for item in sorted(
            items,
            key=lambda item: (
                page_sort_key(str(item.get("page") or "")),
                int(item.get("reading_order_start") or 0),
            ),
        ):
            selected = item["ocr"]["selected"]
            text = selected.get("text") or ""

            writer.writerow(
                {
                    "page": item.get("page"),
                    "region_id": item.get("region_id"),
                    "reading_order_start": item.get("reading_order_start"),
                    "reading_order_end": item.get("reading_order_end"),
                    "confidence": selected.get("confidence"),
                    "empty": selected.get("empty"),
                    "needs_review": item.get("needs_review"),
                    "review_reasons": "|".join(item.get("review_reasons") or []),
                    "low_confidence_line_count": len(selected.get("low_confidence_lines") or []),
                    "crop_path": item.get("crop_path"),
                    "text_preview": text.replace("\n", " / ")[:300],
                }
            )


def page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)


def safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_unique(values: list[Any], value: str) -> list[Any]:
    output = list(values)

    if value not in output:
        output.append(value)

    return output


if __name__ == "__main__":
    main()