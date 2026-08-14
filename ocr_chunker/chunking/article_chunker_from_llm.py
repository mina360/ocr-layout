from __future__ import annotations

from typing import Any


TITLE_ROLES = {
    "document_title",
    "book_title",
    "bab_title",
    "fasl_title",
    "topic_title",
}


SKIP_ROLES = {
    "noise",
    "page_number",
    "container_text",
}


BODY_ROLES = {
    "body_text",
    "article_heading_with_body",
    "preamble_text",
    "unknown",
    "signature_or_date",
}


def build_article_chunks_from_llm(
    items: list[dict[str, Any]],
    *,
    inline_reference_text: bool = False,
) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=_sort_key)

    chunks: list[dict[str, Any]] = []

    current_chunk: dict[str, Any] | None = None

    context = {
        "document_title": None,
        "book_title": None,
        "bab_title": None,
        "fasl_title": None,
        "topic_title": None,
    }

    pending_metadata_box_ids: list[str] = []

    last_article_number: int | None = None
    unknown_counter = 0

    for item in sorted_items:
        decision = _decision(item)
        role = decision.get("role") or "unknown"
        text = str(decision.get("selected_text") or "").strip()

        if not text:
            continue

        if role in SKIP_ROLES:
            continue

        if role in TITLE_ROLES:
            context[role] = decision.get("normalized_title") or text
            pending_metadata_box_ids.append(str(item.get("region_id") or ""))
            continue

        if role in {"article_heading", "article_heading_with_body"}:
            if current_chunk is not None:
                chunks.append(_finalize_chunk(current_chunk))

            raw_article_number = _safe_int(decision.get("article_number"))

            number_result = _resolve_article_number(
                raw_number=raw_article_number,
                last_article_number=last_article_number,
            )

            final_article_number = number_result["article_number"]

            if final_article_number is None:
                unknown_counter += 1
                chunk_id = f"article_unknown_{unknown_counter}"
            else:
                chunk_id = f"article_{final_article_number}"
                last_article_number = final_article_number

            review_reasons: list[str] = []

            if number_result["needs_review"]:
                review_reasons.append(number_result["reason"])

            if decision.get("needs_review"):
                review_reasons.extend(decision.get("review_reasons") or [])

            current_chunk = {
                "chunk_id": chunk_id,
                "chunk_type": "article",
                "article_number": final_article_number,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "metadata": {
                    "document_title": context.get("document_title"),
                    "book_title": context.get("book_title"),
                    "bab_title": context.get("bab_title"),
                    "fasl_title": context.get("fasl_title"),
                    "topic_title": context.get("topic_title"),
                    "article_number_raw": raw_article_number,
                    "article_number_final": final_article_number,
                    "article_number_resolution": number_result["reason"],
                    "heading_box_id": item.get("region_id"),
                    "heading_role_confidence": decision.get("role_confidence"),
                    "heading_selection_confidence": decision.get("selection_confidence"),
                    "heading_selected_engine": decision.get("selected_engine"),
                    "metadata_box_ids": list(pending_metadata_box_ids),
                },
                "heading_box_id": item.get("region_id"),
                "box_ids": [item.get("region_id")],
                "metadata_box_ids": list(pending_metadata_box_ids),
                "source_pages": [item.get("page")],
                "text_parts": [text],
                "references": _references_from_decision(decision),
                "needs_review": bool(review_reasons),
                "review_reasons": list(dict.fromkeys(review_reasons)),
                "skipped_duplicate_box_ids": [],
            }

            pending_metadata_box_ids = []
            continue

        if current_chunk is None:
            current_chunk = {
                "chunk_id": "preamble",
                "chunk_type": "preamble",
                "article_number": None,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "metadata": {
                    "document_title": context.get("document_title"),
                    "book_title": context.get("book_title"),
                    "bab_title": context.get("bab_title"),
                    "fasl_title": context.get("fasl_title"),
                    "topic_title": context.get("topic_title"),
                    "metadata_box_ids": list(pending_metadata_box_ids),
                },
                "heading_box_id": None,
                "box_ids": [],
                "metadata_box_ids": list(pending_metadata_box_ids),
                "source_pages": [],
                "text_parts": [],
                "references": [],
                "needs_review": False,
                "review_reasons": [],
                "skipped_duplicate_box_ids": [],
            }

        if role in BODY_ROLES or role not in TITLE_ROLES:
            if _is_duplicate_for_chunk(current_chunk, text):
                current_chunk["skipped_duplicate_box_ids"].append(item.get("region_id"))
                continue

            current_chunk["box_ids"].append(item.get("region_id"))
            current_chunk["source_pages"].append(item.get("page"))
            current_chunk["page_end"] = item.get("page_number") or current_chunk["page_end"]
            current_chunk["text_parts"].append(text)
            current_chunk["references"].extend(_references_from_decision(decision))

    if current_chunk is not None:
        chunks.append(_finalize_chunk(current_chunk))

    chunks = _resolve_references(chunks, inline_reference_text=inline_reference_text)

    return chunks


def _resolve_article_number(
    *,
    raw_number: int | None,
    last_article_number: int | None,
) -> dict[str, Any]:
    if last_article_number is None:
        if raw_number is None:
            return {
                "article_number": None,
                "needs_review": True,
                "reason": "missing_first_article_number",
            }

        return {
            "article_number": raw_number,
            "needs_review": False,
            "reason": "raw_number_used",
        }

    expected = last_article_number + 1

    if raw_number is None:
        return {
            "article_number": expected,
            "needs_review": True,
            "reason": "inferred_missing_number_from_sequence",
        }

    if raw_number == expected:
        return {
            "article_number": raw_number,
            "needs_review": False,
            "reason": "raw_number_matches_sequence",
        }

    if raw_number <= last_article_number:
        return {
            "article_number": expected,
            "needs_review": True,
            "reason": f"corrected_non_monotonic_raw_{raw_number}_expected_{expected}",
        }

    if raw_number > expected + 2:
        return {
            "article_number": expected,
            "needs_review": True,
            "reason": f"corrected_large_jump_raw_{raw_number}_expected_{expected}",
        }

    return {
        "article_number": raw_number,
        "needs_review": True,
        "reason": f"accepted_small_sequence_gap_raw_{raw_number}_expected_{expected}",
    }


def _finalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    text_parts = [
        str(part).strip()
        for part in chunk.pop("text_parts", [])
        if str(part).strip()
    ]

    final_parts: list[str] = []

    for part in text_parts:
        if _is_duplicate_text(final_parts, part):
            continue
        final_parts.append(part)

    chunk["text"] = "\n".join(final_parts)
    chunk["box_ids"] = _unique_keep_order(chunk.get("box_ids") or [])
    chunk["metadata_box_ids"] = _unique_keep_order(chunk.get("metadata_box_ids") or [])
    chunk["source_pages"] = _unique_keep_order(chunk.get("source_pages") or [])
    chunk["skipped_duplicate_box_ids"] = _unique_keep_order(chunk.get("skipped_duplicate_box_ids") or [])
    chunk["references"] = _unique_references(chunk.get("references") or [])

    metadata = chunk.setdefault("metadata", {})
    metadata["box_count"] = len(chunk["box_ids"])
    metadata["metadata_box_count"] = len(chunk["metadata_box_ids"])
    metadata["skipped_duplicate_box_count"] = len(chunk["skipped_duplicate_box_ids"])
    metadata["source_pages"] = chunk["source_pages"]
    metadata["reference_count"] = len(chunk["references"])

    return chunk


def _resolve_references(
    chunks: list[dict[str, Any]],
    *,
    inline_reference_text: bool,
) -> list[dict[str, Any]]:
    by_number: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        number = chunk.get("article_number")
        if isinstance(number, int):
            by_number[number] = chunk

    for chunk in chunks:
        resolved: list[dict[str, Any]] = []

        for ref in chunk.get("references") or []:
            number = ref.get("article_number")
            target = by_number.get(number)

            item = dict(ref)
            item["resolved"] = target is not None
            item["target_chunk_id"] = target.get("chunk_id") if target else None

            if inline_reference_text and target is not None:
                item["target_text"] = target.get("text")

            resolved.append(item)

        chunk["references"] = resolved
        chunk.setdefault("metadata", {})
        chunk["metadata"]["reference_count"] = len(resolved)
        chunk["metadata"]["has_article_references"] = bool(resolved)

    return chunks


def _references_from_decision(decision: dict[str, Any]) -> list[dict[str, Any]]:
    values = decision.get("referenced_article_numbers") or []

    output: list[dict[str, Any]] = []

    for value in values:
        number = _safe_int(value)
        if number is None:
            continue

        output.append(
            {
                "article_number": number,
                "source": "llm_box_judge",
            }
        )

    return output


def _decision(item: dict[str, Any]) -> dict[str, Any]:
    return ((item.get("llm") or {}).get("box_judge") or {})


def _sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    page = str(item.get("page") or "")
    digits = "".join(ch for ch in page if ch.isdigit())
    page_number = int(digits) if digits else int(item.get("page_number") or 0)

    return (
        page_number,
        int(item.get("reading_order_start") or 0),
        str(item.get("region_id") or ""),
    )


def _is_duplicate_for_chunk(chunk: dict[str, Any], new_text: str) -> bool:
    existing = chunk.get("text_parts") or []

    return _is_duplicate_text([str(v) for v in existing], new_text)


def _is_duplicate_text(existing_parts: list[str], new_text: str) -> bool:
    new_norm = _normalize_for_compare(new_text)

    if len(new_norm) < 12:
        return False

    for old in existing_parts:
        old_norm = _normalize_for_compare(old)

        if not old_norm:
            continue

        if new_norm in old_norm:
            return True

        if old_norm in new_norm and len(old_norm) >= 30:
            return True

        if _rough_overlap(new_norm, old_norm) >= 0.88:
            return True

    return False


def _rough_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    shorter = a if len(a) <= len(b) else b
    longer = b if len(a) <= len(b) else a

    if shorter in longer:
        return 1.0

    window = max(10, min(80, len(shorter)))
    total = 0
    hits = 0

    for start in range(0, len(shorter), window):
        piece = shorter[start:start + window]

        if len(piece) < 8:
            continue

        total += 1
        if piece in longer:
            hits += 1

    if total == 0:
        return 0.0

    return hits / total


def _normalize_for_compare(text: str) -> str:
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )

    return "".join(text.split())


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
        key = str(value.get("article_number"))

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None