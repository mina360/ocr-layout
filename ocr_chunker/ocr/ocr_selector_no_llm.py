from __future__ import annotations

from statistics import median
from typing import Any


TITLE_LIKE_LABELS = {
    "doc_title",
    "header",
    "paragraph_title",
    "figure_title",
    "number",
    "footer",
}


def select_best_ocr_no_llm(
    items: list[dict[str, Any]],
    *,
    low_confidence_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    page_area_medians = _page_area_medians(items)

    output: list[dict[str, Any]] = []

    for item in items:
        item = dict(item)
        item.setdefault("ocr", {})

        page = str(item.get("page") or "")
        page_median_area = page_area_medians.get(page) or 0.0

        decision = choose_ocr_for_item(
            item,
            page_median_area=page_median_area,
            low_confidence_threshold=low_confidence_threshold,
        )

        item["ocr"]["selected"] = decision
        output.append(item)

    return output


def choose_ocr_for_item(
    item: dict[str, Any],
    *,
    page_median_area: float,
    low_confidence_threshold: float,
) -> dict[str, Any]:
    label = str(item.get("label") or "")

    paddle_text = _paddle_text(item)
    paddle_conf = _paddle_confidence(item)

    easy_text = _easy_text(item)
    easy_conf = _easy_confidence(item)

    area = _bbox_area(item.get("bbox"))
    is_small_box = page_median_area > 0 and area <= page_median_area * 0.55
    is_large_box = page_median_area > 0 and area >= page_median_area * 4.0
    is_title_like = label in TITLE_LIKE_LABELS

    paddle_quality = _text_quality(paddle_text)
    easy_quality = _text_quality(easy_text)

    # 1. Empty Paddle: use EasyOCR if available.
    if not paddle_text.strip() and easy_text.strip():
        return _decision(
            engine="easyocr",
            text=_normalize_short_text_if_needed(easy_text, label, is_small_box),
            confidence=easy_conf,
            reason="paddle_empty_easyocr_available",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    # 2. Title-like or small boxes: EasyOCR usually preserves Arabic visual order better.
    if easy_text.strip() and (is_title_like or is_small_box):
        return _decision(
            engine="easyocr",
            text=_normalize_short_text_if_needed(easy_text, label, is_small_box),
            confidence=easy_conf,
            reason="easyocr_preferred_for_title_like_or_small_box",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    # 3. Image boxes in this project are not real images. Treat them as text containers.
    # For large containers, Paddle is often more complete, but keep them marked for chunker.
    if label == "image":
        if easy_text.strip() and easy_quality > paddle_quality + 0.15:
            return _decision(
                engine="easyocr",
                text=easy_text,
                confidence=easy_conf,
                reason="image_label_text_container_easyocr_better_quality",
                paddle_text=paddle_text,
                easy_text=easy_text,
            )

        return _decision(
            engine="paddleocr_colab_rec_texts",
            text=paddle_text,
            confidence=paddle_conf,
            reason="image_label_treated_as_text_container_paddle_kept",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    # 4. Body text: keep Paddle unless low confidence and EasyOCR looks better.
    if (
        easy_text.strip()
        and paddle_conf is not None
        and paddle_conf < low_confidence_threshold
        and easy_quality >= paddle_quality
    ):
        return _decision(
            engine="easyocr",
            text=easy_text,
            confidence=easy_conf,
            reason="easyocr_used_for_low_paddle_confidence_body",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    if paddle_text.strip():
        return _decision(
            engine="paddleocr_colab_rec_texts",
            text=paddle_text,
            confidence=paddle_conf,
            reason="paddle_kept_for_body_or_default",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    if easy_text.strip():
        return _decision(
            engine="easyocr",
            text=easy_text,
            confidence=easy_conf,
            reason="easyocr_fallback",
            paddle_text=paddle_text,
            easy_text=easy_text,
        )

    return _decision(
        engine="none",
        text="",
        confidence=None,
        reason="both_ocr_empty",
        paddle_text=paddle_text,
        easy_text=easy_text,
    )


def _decision(
    *,
    engine: str,
    text: str,
    confidence: float | None,
    reason: str,
    paddle_text: str,
    easy_text: str,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "text": text.strip(),
        "confidence": confidence,
        "status": "best_ocr_selected_no_llm",
        "selection_reason": reason,
        "empty": not bool(text.strip()),
        "debug": {
            "paddle_text_preview": paddle_text.replace("\n", " / ")[:250],
            "easyocr_text_preview": easy_text.replace("\n", " / ")[:250],
            "paddle_quality": _text_quality(paddle_text),
            "easyocr_quality": _text_quality(easy_text),
        },
    }


def _normalize_short_text_if_needed(text: str, label: str, is_small_box: bool) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return ""

    if label in TITLE_LIKE_LABELS or is_small_box:
        if len(lines) <= 4:
            return " ".join(lines)

    return text.strip()


def _paddle_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def _paddle_confidence(item: dict[str, Any]) -> float | None:
    selected = (item.get("ocr") or {}).get("selected") or {}
    return _safe_float(selected.get("confidence"))


def _easy_text(item: dict[str, Any]) -> str:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    text = easy.get("text")
    return text if isinstance(text, str) else ""


def _easy_confidence(item: dict[str, Any]) -> float | None:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    return _safe_float(easy.get("confidence"))


def _text_quality(text: str) -> float:
    text = text.strip()

    if not text:
        return 0.0

    chars = [ch for ch in text if not ch.isspace()]

    if not chars:
        return 0.0

    arabic_count = sum(1 for ch in chars if "\u0600" <= ch <= "\u06ff")
    digit_count = sum(1 for ch in chars if ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩")

    arabic_ratio = arabic_count / len(chars)
    digit_ratio = digit_count / len(chars)

    line_count = len([line for line in text.splitlines() if line.strip()])

    score = arabic_ratio * 0.75
    score += min(0.15, digit_ratio * 0.5)

    if line_count <= 4:
        score += 0.08

    if len(chars) <= 2 and digit_count == 0:
        score -= 0.25

    latin_count = sum(1 for ch in chars if "a" <= ch.lower() <= "z")

    if latin_count:
        score -= min(0.25, latin_count * 0.04)

    return round(max(0.0, min(1.0, score)), 4)


def _page_area_medians(items: list[dict[str, Any]]) -> dict[str, float]:
    by_page: dict[str, list[int]] = {}

    for item in items:
        page = str(item.get("page") or "")
        area = _bbox_area(item.get("bbox"))

        if area <= 0:
            continue

        by_page.setdefault(page, []).append(area)

    result: dict[str, float] = {}

    for page, areas in by_page.items():
        result[page] = float(median(areas))

    return result


def _bbox_area(value: object) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return 0

    return max(0, x2 - x1) * max(0, y2 - y1)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None