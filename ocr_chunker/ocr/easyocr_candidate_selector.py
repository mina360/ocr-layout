from __future__ import annotations

from statistics import median
from typing import Any


ARTICLE_WORD = "المادة"

TITLE_LABELS = {
    "paragraph_title",
    "header",
    "doc_title",
    "figure_title",
    "number",
    "footer",
}


def select_easyocr_candidates(
    items: list[dict[str, Any]],
    *,
    low_confidence_threshold: float = 0.75,
    short_text_max_chars: int = 30,
    small_box_area_ratio: float = 1.25,
) -> list[dict[str, Any]]:
    """
    Select only suspicious/small OCR regions for EasyOCR.

    No regex is used.
    We rely on:
    - OCR confidence
    - text length
    - box size
    - layout label
    - article-heading-like text
    - numeric/small boxes
    """
    median_area = _median_area(items)

    candidates: list[dict[str, Any]] = []

    for item in items:
        reasons = _candidate_reasons(
            item,
            median_area=median_area,
            low_confidence_threshold=low_confidence_threshold,
            short_text_max_chars=short_text_max_chars,
            small_box_area_ratio=small_box_area_ratio,
        )

        if not reasons:
            continue

        candidates.append(
            {
                "region_id": item.get("region_id"),
                "box_id": item.get("box_id") or item.get("region_id"),
                "page": item.get("page"),
                "page_number": item.get("page_number"),
                "crop_path": item.get("crop_path"),
                "bbox": item.get("bbox"),
                "label": item.get("label"),
                "layout_score": item.get("layout_score"),
                "paddle_text": _selected_text(item),
                "paddle_confidence": _selected_confidence(item),
                "reasons": reasons,
            }
        )

    return candidates


def _candidate_reasons(
    item: dict[str, Any],
    *,
    median_area: float,
    low_confidence_threshold: float,
    short_text_max_chars: int,
    small_box_area_ratio: float,
) -> list[str]:
    reasons: list[str] = []

    text = _selected_text(item)
    confidence = _selected_confidence(item)
    label = str(item.get("label") or "")
    area = _bbox_area(item.get("bbox"))
    is_small = median_area > 0 and area <= median_area * small_box_area_ratio

    clean_len = len(text.replace("\n", "").strip())

    if not text.strip():
        reasons.append("empty_paddle_text")

    if confidence is not None and confidence < low_confidence_threshold:
        reasons.append("low_paddle_confidence")

    if label in TITLE_LABELS:
        reasons.append("title_or_number_label")

    if is_small and clean_len <= short_text_max_chars:
        reasons.append("small_short_box")

    if _contains_article_word(text) and (is_small or clean_len <= 40):
        reasons.append("article_word_small_box")

    if _has_digits(text) and (is_small or clean_len <= short_text_max_chars):
        reasons.append("small_numeric_box")

    if _contains_latin_noise(text):
        reasons.append("latin_noise")

    # Do not waste EasyOCR on long high-confidence body text.
    if (
        clean_len > 120
        and confidence is not None
        and confidence >= 0.88
        and "latin_noise" not in reasons
        and "empty_paddle_text" not in reasons
    ):
        return []

    return list(dict.fromkeys(reasons))


def _selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def _selected_confidence(item: dict[str, Any]) -> float | None:
    selected = (item.get("ocr") or {}).get("selected") or {}
    value = selected.get("confidence")

    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _median_area(items: list[dict[str, Any]]) -> float:
    areas = [_bbox_area(item.get("bbox")) for item in items]
    areas = [area for area in areas if area > 0]

    if not areas:
        return 0.0

    return float(median(areas))


def _bbox_area(value: object) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return 0

    return max(0, x2 - x1) * max(0, y2 - y1)


def _contains_article_word(text: str) -> bool:
    return ARTICLE_WORD in _normalize_arabic(text)


def _has_digits(text: str) -> bool:
    for ch in text:
        if ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩":
            return True
    return False


def _contains_latin_noise(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    latin_count = 0
    for ch in stripped:
        if ("a" <= ch.lower() <= "z") or ch in {"O", "L"}:
            latin_count += 1

    return latin_count > 0 and len(stripped) <= 20


def _normalize_arabic(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .strip()
    )