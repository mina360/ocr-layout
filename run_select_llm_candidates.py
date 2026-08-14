from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


LLM_LAYOUT_LABELS = {
    "image",
    "header",
    "doc_title",
    "paragraph_title",
    "figure_title",
    "number",
    "footer",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.78)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    with manifest_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    candidates = select_llm_candidates(
        items,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_csv(candidates, Path(args.csv))

    print(f"OK: items={len(items)} llm_candidates={len(candidates)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {args.csv}")


def select_llm_candidates(
    items: list[dict[str, Any]],
    *,
    low_confidence_threshold: float,
) -> list[dict[str, Any]]:
    page_area_medians = compute_page_area_medians(items)

    candidates: list[dict[str, Any]] = []

    for item in items:
        reasons = candidate_reasons(
            item,
            page_area_medians=page_area_medians,
            low_confidence_threshold=low_confidence_threshold,
        )

        if not reasons:
            continue

        candidates.append(
            {
                "region_id": item.get("region_id"),
                "box_id": item.get("box_id") or item.get("region_id"),
                "page": item.get("page"),
                "page_number": item.get("page_number"),
                "label": item.get("label"),
                "layout_score": item.get("layout_score"),
                "bbox": item.get("bbox"),
                "crop_path": item.get("crop_path"),
                "paddle_text": selected_text(item),
                "paddle_confidence": selected_confidence(item),
                "easyocr_text": easyocr_text(item),
                "easyocr_confidence": easyocr_confidence(item),
                "reasons": reasons,
            }
        )

    return candidates


def candidate_reasons(
    item: dict[str, Any],
    *,
    page_area_medians: dict[str, float],
    low_confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []

    label = str(item.get("label") or "")
    page = str(item.get("page") or "")

    paddle_text = selected_text(item)
    paddle_conf = selected_confidence(item)

    easy_text = easyocr_text(item)
    easy_conf = easyocr_confidence(item)

    area = bbox_area(item.get("bbox"))
    page_median = page_area_medians.get(page) or 0.0

    if label in LLM_LAYOUT_LABELS:
        reasons.append("layout_label_needs_semantic_role")

    if label == "image":
        reasons.append("image_label_but_document_has_no_real_images")

    if easy_text.strip():
        reasons.append("has_easyocr_candidate_output")

    if not paddle_text.strip():
        reasons.append("empty_paddle_text")

    if paddle_conf is not None and paddle_conf < low_confidence_threshold:
        reasons.append("low_paddle_confidence")

    if easy_text.strip() and normalized_compare(easy_text) != normalized_compare(paddle_text):
        reasons.append("paddle_easyocr_disagree")

    if page_median > 0:
        if area <= page_median * 0.35:
            reasons.append("very_small_layout_box")
        elif area >= page_median * 4.0:
            reasons.append("very_large_layout_box")

    return list(dict.fromkeys(reasons))


def compute_page_area_medians(items: list[dict[str, Any]]) -> dict[str, float]:
    by_page: dict[str, list[int]] = {}

    for item in items:
        page = str(item.get("page") or "")
        area = bbox_area(item.get("bbox"))

        if area <= 0:
            continue

        by_page.setdefault(page, []).append(area)

    return {
        page: float(median(areas))
        for page, areas in by_page.items()
        if areas
    }


def selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def selected_confidence(item: dict[str, Any]) -> float | None:
    selected = (item.get("ocr") or {}).get("selected") or {}
    return safe_float(selected.get("confidence"))


def easyocr_text(item: dict[str, Any]) -> str:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    text = easy.get("text")
    return text if isinstance(text, str) else ""


def easyocr_confidence(item: dict[str, Any]) -> float | None:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    return safe_float(easy.get("confidence"))


def bbox_area(value: object) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return 0

    return max(0, x2 - x1) * max(0, y2 - y1)


def normalized_compare(text: str) -> str:
    return "".join(
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .split()
    )


def safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "label",
                "paddle_confidence",
                "easyocr_confidence",
                "reasons",
                "paddle_text",
                "easyocr_text",
                "crop_path",
            ],
        )
        writer.writeheader()

        for item in candidates:
            writer.writerow(
                {
                    "page": item.get("page"),
                    "region_id": item.get("region_id"),
                    "label": item.get("label"),
                    "paddle_confidence": item.get("paddle_confidence"),
                    "easyocr_confidence": item.get("easyocr_confidence"),
                    "reasons": "|".join(item.get("reasons") or []),
                    "paddle_text": str(item.get("paddle_text") or "").replace("\n", " / ")[:250],
                    "easyocr_text": str(item.get("easyocr_text") or "").replace("\n", " / ")[:250],
                    "crop_path": item.get("crop_path"),
                }
            )


if __name__ == "__main__":
    main()