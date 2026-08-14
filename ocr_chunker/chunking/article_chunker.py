from __future__ import annotations

from typing import Any

from ocr_chunker.chunking.article_detector import enrich_items_with_article_signals


SKIP_LABELS = {
    "image",
}

CONTEXT_WORDS = {
    "الكتاب": "book",
    "الباب": "bab",
    "الفصل": "fasl",
}


def build_article_chunks(
    items: list[dict[str, Any]],
    *,
    inline_reference_text: bool = False,
) -> list[dict[str, Any]]:
    enriched = enrich_items_with_article_signals(items)
    usable = [_item for _item in enriched if _is_usable_for_chunking(_item)]
    usable.sort(key=_sort_key)

    chunks: list[dict[str, Any]] = []
    current_chunk: dict[str, Any] | None = None

    context = {
        "book": None,
        "bab": None,
        "fasl": None,
    }

    unknown_counter = 0
    last_article_number: int | None = None

    for item in usable:
        selected_text = _selected_text(item).strip()

        if not selected_text:
            continue

        context_update = _detect_context_update(selected_text)

        if context_update:
            context.update(context_update)
            continue

        signals = item.get("article_signals") or {}

        if signals.get("is_article_heading_candidate"):
            if current_chunk is not None:
                chunks.append(_finalize_chunk(current_chunk))

            article_number = signals.get("article_number")

            if article_number is None and last_article_number is not None:
                article_number = last_article_number + 1

            if article_number is None:
                unknown_counter += 1
                article_key = f"unknown_{unknown_counter}"
                needs_review = True
                review_reasons = ["article_number_missing"]
            else:
                article_key = str(article_number)
                needs_review = False
                review_reasons = []
                last_article_number = int(article_number)

            current_chunk = {
                "chunk_id": f"article_{article_key}",
                "chunk_type": "article",
                "article_number": article_number,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "book": context.get("book"),
                "bab": context.get("bab"),
                "fasl": context.get("fasl"),
                "heading_box_id": item.get("region_id"),
                "box_ids": [item.get("region_id")],
                "source_pages": [item.get("page")],
                "text_parts": [selected_text],
                "references": [],
                "needs_review": needs_review,
                "review_reasons": review_reasons,
            }
            continue

        if current_chunk is None:
            current_chunk = {
                "chunk_id": "preamble",
                "chunk_type": "preamble",
                "article_number": None,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "book": context.get("book"),
                "bab": context.get("bab"),
                "fasl": context.get("fasl"),
                "heading_box_id": None,
                "box_ids": [],
                "source_pages": [],
                "text_parts": [],
                "references": [],
                "needs_review": False,
                "review_reasons": [],
            }

        current_chunk["box_ids"].append(item.get("region_id"))
        current_chunk["source_pages"].append(item.get("page"))
        current_chunk["page_end"] = item.get("page_number") or current_chunk["page_end"]
        current_chunk["text_parts"].append(selected_text)

        for reference in signals.get("inline_article_references") or []:
            current_chunk["references"].append(reference)

    if current_chunk is not None:
        chunks.append(_finalize_chunk(current_chunk))

    chunks = _deduplicate_chunk_texts(chunks)
    chunks = _resolve_references(chunks, inline_reference_text=inline_reference_text)

    return chunks


def _finalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        part.strip()
        for part in chunk.pop("text_parts", [])
        if part and part.strip()
    )

    chunk["text"] = text
    chunk["box_ids"] = _unique_keep_order(chunk.get("box_ids") or [])
    chunk["source_pages"] = _unique_keep_order(chunk.get("source_pages") or [])
    chunk["references"] = _unique_references(chunk.get("references") or [])

    return chunk


def _is_usable_for_chunking(item: dict[str, Any]) -> bool:
    label = str(item.get("label") or "")
    text = _selected_text(item).strip()

    if not text:
        return False

    if label in SKIP_LABELS:
        return False

    if _looks_like_page_number(text):
        return False

    if _looks_like_noise(text):
        return False

    return True


def _detect_context_update(text: str) -> dict[str, str] | None:
    normalized = _normalize_arabic(text)
    compact = " ".join(text.split())

    for word, key in CONTEXT_WORDS.items():
        if word in normalized and len(compact) <= 80:
            return {key: compact}

    return None


def _deduplicate_chunk_texts(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for chunk in chunks:
        lines = [
            line.strip()
            for line in str(chunk.get("text") or "").splitlines()
            if line.strip()
        ]

        cleaned_lines: list[str] = []
        seen: set[str] = set()

        for line in lines:
            normalized = _normalize_for_duplicate(line)

            if normalized in seen:
                continue

            # If this line is already included inside a previous longer line,
            # skip it to reduce raw box duplication.
            if any(normalized and normalized in _normalize_for_duplicate(prev) for prev in cleaned_lines):
                continue

            cleaned_lines.append(line)
            seen.add(normalized)

        chunk["text"] = "\n".join(cleaned_lines)

    return chunks


def _resolve_references(
    chunks: list[dict[str, Any]],
    *,
    inline_reference_text: bool,
) -> list[dict[str, Any]]:
    by_article_number: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        number = chunk.get("article_number")
        if isinstance(number, int):
            by_article_number[number] = chunk

    for chunk in chunks:
        resolved: list[dict[str, Any]] = []

        for reference in chunk.get("references") or []:
            number = reference.get("article_number")
            target = by_article_number.get(number)

            resolved_item = dict(reference)
            resolved_item["resolved"] = target is not None
            resolved_item["target_chunk_id"] = target.get("chunk_id") if target else None

            if inline_reference_text and target is not None:
                resolved_item["target_text"] = target.get("text")

            resolved.append(resolved_item)

        chunk["references"] = resolved

        if resolved:
            chunk.setdefault("metadata", {})
            chunk["metadata"]["has_article_references"] = True

    return chunks


def _selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def _sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    page = str(item.get("page") or "")
    digits = "".join(ch for ch in page if ch.isdigit())
    page_number = int(digits) if digits else int(item.get("page_number") or 0)

    return (
        page_number,
        int(item.get("reading_order_start") or 0),
        str(item.get("region_id") or ""),
    )


def _unique_keep_order(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()

    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)

    return output


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


def _looks_like_page_number(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if len(stripped) > 4:
        return False

    return all(ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩" for ch in stripped)


def _looks_like_noise(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    blocked = {
        "v3_predict",
        "min",
        "general",
        "tt",
        "f",
        "L",
        "O",
    }

    if stripped in blocked:
        return True

    if stripped.startswith("/tmp/"):
        return True

    if len(stripped) <= 2 and not any(ch in "ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآؤئ" for ch in stripped):
        return True

    return False


def _normalize_arabic(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .strip()
    )


def _normalize_for_duplicate(text: str) -> str:
    return "".join(_normalize_arabic(text).split())