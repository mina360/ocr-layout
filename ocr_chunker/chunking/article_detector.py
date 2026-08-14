from __future__ import annotations

from statistics import median
from typing import Any


def enrich_items_with_article_signals(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    heading_area_median = _heading_area_median(items)

    enriched: list[dict[str, Any]] = []

    for item in items:
        item = dict(item)

        text = _selected_text(item)
        heading = is_article_heading_candidate(
            item,
            heading_area_median=heading_area_median,
        )

        references = extract_inline_article_references(text)

        item["article_signals"] = {
            "is_article_heading_candidate": heading["is_candidate"],
            "article_number": heading["article_number"],
            "heading_confidence": heading["confidence"],
            "heading_reasons": heading["reasons"],
            "inline_article_references": references,
            "contains_article_word": _contains_article_word(text),
        }

        enriched.append(item)

    return enriched


def is_article_heading_candidate(
    item: dict[str, Any],
    *,
    heading_area_median: float,
) -> dict[str, Any]:
    text = _selected_text(item)
    normalized = _normalize_arabic(text)
    label = str(item.get("label") or "")
    area = _bbox_area(item.get("bbox"))

    compact = " ".join(text.split())
    word_count = _word_count(compact)
    line_count = len([line for line in text.splitlines() if line.strip()])
    article_number = extract_article_number(compact)

    reasons: list[str] = []

    if not _contains_article_word(text):
        return _decision(False, article_number, 0.0, reasons)

    reasons.append("contains_article_word")

    if word_count <= 5:
        reasons.append("short_text")

    if line_count <= 2:
        reasons.append("one_or_two_lines")

    if label in {"paragraph_title", "header", "number", "doc_title"}:
        reasons.append("title_like_label")

    if article_number is not None:
        reasons.append("has_article_number")

    if heading_area_median > 0 and area <= heading_area_median * 3.0:
        reasons.append("heading_sized_box")

    # Negative signal: this is probably a body sentence such as:
    # "وفقاً لما ورد في المادة ٢٥ ..."
    if word_count >= 8:
        reasons.append("too_many_words_for_heading")

    positive_count = sum(1 for r in reasons if r != "too_many_words_for_heading")
    confidence = round(min(1.0, positive_count / 5.0), 3)

    is_candidate = (
        "too_many_words_for_heading" not in reasons
        and "short_text" in reasons
        and (
            "has_article_number" in reasons
            or "heading_sized_box" in reasons
            or "title_like_label" in reasons
        )
    )

    return _decision(is_candidate, article_number, confidence, reasons)


def extract_article_number(text: str) -> int | None:
    normalized = _normalize_arabic(text)
    marker_index = _find_article_marker(normalized)

    if marker_index < 0:
        return None

    marker_end = marker_index + len("الماده")
    number = _extract_number_after_index(normalized, marker_end)

    if number is not None:
        return number

    # Fallback: OCR sometimes returns compact text like المادة٢٨
    digits = ""
    for ch in normalized:
        digit = _digit_value(ch)
        if digit is not None:
            digits += str(digit)
        elif digits:
            break

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def extract_inline_article_references(text: str) -> list[dict[str, Any]]:
    """
    Detect article references inside body text without regex.

    A standalone heading like "المادة ٢٥" is not considered a reference.
    """
    if _word_count(text) <= 5:
        return []

    normalized = _normalize_arabic(text)
    references: list[dict[str, Any]] = []

    start = 0

    while True:
        idx = _find_article_marker(normalized, start=start)

        if idx < 0:
            break

        number = _extract_number_after_index(normalized, idx + len("الماده"))

        if number is not None:
            references.append(
                {
                    "article_number": number,
                    "surface_text": _surface_window(text, idx),
                }
            )

        start = idx + len("الماده")

    return _unique_references(references)


def _decision(
    is_candidate: bool,
    article_number: int | None,
    confidence: float,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "is_candidate": is_candidate,
        "article_number": article_number,
        "confidence": confidence,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _contains_article_word(text: str) -> bool:
    normalized = _normalize_arabic(text)
    return _find_article_marker(normalized) >= 0


def _find_article_marker(text: str, *, start: int = 0) -> int:
    for marker in ("الماده", "الماد"):
        idx = text.find(marker, start)
        if idx >= 0:
            return idx

    return -1


def _extract_number_after_index(text: str, index: int) -> int | None:
    digits = ""

    for ch in text[index:index + 16]:
        digit = _digit_value(ch)

        if digit is not None:
            digits += str(digit)
            continue

        if digits:
            break

        if ch in {" ", "\n", "\t", ":", "ـ", "-", "،", "/", "\\"}:
            continue

        # allow OCR to put article marker then no visible number
        break

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def _surface_window(text: str, index: int) -> str:
    start = max(0, index - 20)
    end = min(len(text), index + 40)
    return " ".join(text[start:end].split())


def _selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def _heading_area_median(items: list[dict[str, Any]]) -> float:
    areas: list[int] = []

    for item in items:
        text = _selected_text(item)

        if not _contains_article_word(text):
            continue

        # Only use short article-like boxes for median size.
        if _word_count(text) > 5:
            continue

        area = _bbox_area(item.get("bbox"))
        if area > 0:
            areas.append(area)

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


def _word_count(text: str) -> int:
    return len([part for part in text.replace("\n", " ").split(" ") if part.strip()])


def _normalize_arabic(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .strip()
    )


def _digit_value(ch: str) -> int | None:
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"

    if ch in arabic_digits:
        return arabic_digits.index(ch)

    if ch.isdigit():
        return int(ch)

    return None


def _unique_references(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for value in values:
        key = f"{value.get('article_number')}::{value.get('surface_text')}"
        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output