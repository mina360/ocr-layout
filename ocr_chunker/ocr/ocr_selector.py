from __future__ import annotations

from typing import Any


def select_best_ocr_for_items(
    items: list[dict[str, Any]],
    *,
    easy_margin: float = 0.08,
    low_paddle_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []

    for item in items:
        item = dict(item)
        item.setdefault("ocr", {})

        paddle_text, paddle_conf = _clean_paddle_baseline(item)
        easy_text, easy_conf = _easyocr_result(item)

        selected = _choose(
            paddle_text=paddle_text,
            paddle_conf=paddle_conf,
            easy_text=easy_text,
            easy_conf=easy_conf,
            item=item,
            easy_margin=easy_margin,
            low_paddle_threshold=low_paddle_threshold,
        )

        item["ocr"]["selected"] = selected
        updated.append(item)

    return updated


def _clean_paddle_baseline(item: dict[str, Any]) -> tuple[str, float | None]:
    """
    Important:
    raw_box_crops_manifest_paddle_clean.json already has the good Paddle text
    under ocr.selected, because the old ocr.paddle.parsed text is polluted
    with v3_predict, /tmp paths, and duplicates.
    """
    selected = (item.get("ocr") or {}).get("selected") or {}

    text = selected.get("text")
    confidence = selected.get("confidence")

    if isinstance(text, str):
        return text, _safe_float(confidence)

    paddle = ((item.get("ocr") or {}).get("paddle") or {}).get("parsed") or {}
    return str(paddle.get("text") or ""), _safe_float(paddle.get("confidence"))


def _easyocr_result(item: dict[str, Any]) -> tuple[str, float | None]:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    return str(easy.get("text") or ""), _safe_float(easy.get("confidence"))


def _choose(
    *,
    paddle_text: str,
    paddle_conf: float | None,
    easy_text: str,
    easy_conf: float | None,
    item: dict[str, Any],
    easy_margin: float,
    low_paddle_threshold: float,
) -> dict[str, Any]:
    paddle_text = paddle_text.strip()
    easy_text = easy_text.strip()

    if easy_text and not paddle_text:
        return _selected("easyocr", easy_text, easy_conf, "paddle_empty_easy_available")

    if paddle_text and not easy_text:
        return _selected("paddleocr_colab_rec_texts", paddle_text, paddle_conf, "paddle_only_or_easy_not_run")

    if not paddle_text and not easy_text:
        return _selected("none", "", None, "both_empty")

    paddle_conf_value = paddle_conf if paddle_conf is not None else 0.0
    easy_conf_value = easy_conf if easy_conf is not None else 0.0

    is_small_article_or_number = _looks_like_small_article_or_number(item, paddle_text)

    if is_small_article_or_number and easy_conf_value >= 0.45 and easy_conf_value >= paddle_conf_value - 0.05:
        return _selected("easyocr", easy_text, easy_conf, "easyocr_preferred_for_small_article_or_number")

    if paddle_conf_value < low_paddle_threshold and easy_conf_value > paddle_conf_value:
        return _selected("easyocr", easy_text, easy_conf, "easyocr_better_low_paddle")

    if easy_conf_value >= paddle_conf_value + easy_margin:
        return _selected("easyocr", easy_text, easy_conf, "easyocr_confidence_margin")

    return _selected("paddleocr_colab_rec_texts", paddle_text, paddle_conf, "paddle_kept")


def _selected(
    engine: str,
    text: str,
    confidence: float | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "text": text,
        "confidence": confidence,
        "status": "best_ocr_selected",
        "selection_reason": reason,
        "empty": not bool(text.strip()),
    }


def _looks_like_small_article_or_number(item: dict[str, Any], text: str) -> bool:
    normalized = _normalize_arabic(text)
    label = str(item.get("label") or "")

    if _contains_article_word(normalized):
        return True

    if label in {"number", "footer", "paragraph_title", "header", "doc_title"}:
        return True

    stripped = text.strip()
    return len(stripped) <= 12 and _has_digits(stripped)


def _contains_article_word(normalized_text: str) -> bool:
    return "الماده" in normalized_text or "الماد" in normalized_text


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩" for ch in text)


def _normalize_arabic(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .strip()
    )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None