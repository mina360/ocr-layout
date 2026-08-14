from __future__ import annotations

from typing import Any


CONTEXT_LABELS = {
    "image",
    "doc_title",
    "header",
    "figure_title",
    "paragraph_title",
}


SKIP_LABELS = {
    "number",
    "footer",
}


ARTICLE_WORD = "المادة"


def build_article_chunks_improved(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=sort_key)

    chunks: list[dict[str, Any]] = []
    current_chunk: dict[str, Any] | None = None

    active_context_blocks: list[dict[str, Any]] = []
    pending_context_blocks: list[dict[str, Any]] = []

    last_article_number: int | None = None
    unknown_counter = 0

    for item in sorted_items:
        text = selected_text(item).strip()

        if not text:
            continue

        if is_page_number_text(text):
            continue

        if is_noise_text(text):
            continue

        article_info = detect_article_heading(text)

        if article_info["is_article_heading"]:
            if current_chunk is not None:
                chunks.append(finalize_chunk(current_chunk))

            raw_number = article_info["article_number"]

            number_result = resolve_article_number(
                raw_number=raw_number,
                last_article_number=last_article_number,
                pending_context_blocks=pending_context_blocks,
            )

            final_number = number_result["article_number"]

            if final_number is None:
                unknown_counter += 1
                chunk_id = f"article_unknown_{unknown_counter}"
            else:
                chunk_id = f"article_{final_number}"
                last_article_number = final_number

            review_reasons: list[str] = []

            if number_result["needs_review"]:
                review_reasons.append(number_result["reason"])

            current_context = list(active_context_blocks)

            if pending_context_blocks:
                current_context.extend(pending_context_blocks)
                active_context_blocks.extend(pending_context_blocks)

            current_context = unique_context_blocks(current_context)
            active_context_blocks = unique_context_blocks(active_context_blocks)

            current_chunk = {
                "chunk_id": chunk_id,
                "chunk_type": "article",
                "article_number": final_number,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "heading_box_id": item.get("region_id"),
                "box_ids": [item.get("region_id")],
                "body_box_ids": [],
                "metadata_box_ids": [block["box_id"] for block in pending_context_blocks],
                "skipped_duplicate_box_ids": [],
                "source_pages": [item.get("page")],
                "text_parts": [text],
                "references": references_from_text(text, skip_heading=True),
                "needs_review": bool(review_reasons),
                "review_reasons": list(dict.fromkeys(review_reasons)),
                "metadata": {
                    "article_number_raw": raw_number,
                    "article_number_final": final_number,
                    "article_number_resolution": number_result["reason"],
                    "heading_box_id": item.get("region_id"),
                    "heading_text": text,
                    "context_blocks": current_context,
                    "new_context_blocks": list(pending_context_blocks),
                    "ocr_engine": selected_engine(item),
                    "ocr_confidence": selected_confidence(item),
                    "selection_reason": selected_reason(item),
                },
            }

            pending_context_blocks = []
            continue

        if is_context_block(item, text):
            context_block = make_context_block(item, text)

            pending_context_blocks.append(context_block)
            pending_context_blocks = unique_context_blocks(pending_context_blocks)
            continue

        if is_container_text(item, text):
            # Large overlapping container, especially image boxes.
            # Keep it out of article text to reduce duplication.
            if current_chunk is not None:
                current_chunk["skipped_duplicate_box_ids"].append(item.get("region_id"))
            continue

        if current_chunk is None:
            current_chunk = {
                "chunk_id": "preamble",
                "chunk_type": "preamble",
                "article_number": None,
                "page_start": item.get("page_number"),
                "page_end": item.get("page_number"),
                "heading_box_id": None,
                "box_ids": [],
                "body_box_ids": [],
                "metadata_box_ids": [],
                "skipped_duplicate_box_ids": [],
                "source_pages": [],
                "text_parts": [],
                "references": [],
                "needs_review": False,
                "review_reasons": [],
                "metadata": {
                    "context_blocks": list(active_context_blocks),
                    "new_context_blocks": [],
                },
            }

        if is_duplicate_for_chunk(current_chunk, text):
            current_chunk["skipped_duplicate_box_ids"].append(item.get("region_id"))
            continue

        current_chunk["box_ids"].append(item.get("region_id"))
        current_chunk["body_box_ids"].append(item.get("region_id"))
        current_chunk["source_pages"].append(item.get("page"))
        current_chunk["page_end"] = item.get("page_number") or current_chunk["page_end"]
        current_chunk["text_parts"].append(text)
        current_chunk["references"].extend(references_from_text(text, skip_heading=False))

    if current_chunk is not None:
        chunks.append(finalize_chunk(current_chunk))

    chunks = resolve_references(chunks)

    return chunks


def detect_article_heading(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return {
            "is_article_heading": False,
            "article_number": None,
        }

    first = normalize_arabic(lines[0])
    compact = normalize_arabic(" ".join(lines))

    starts_with_article = first.startswith(normalize_arabic(ARTICLE_WORD)) or compact.startswith(normalize_arabic(ARTICLE_WORD))

    if not starts_with_article:
        return {
            "is_article_heading": False,
            "article_number": None,
        }

    article_number = extract_article_number(text)

    return {
        "is_article_heading": True,
        "article_number": article_number,
    }


def extract_article_number(text: str) -> int | None:
    normalized = normalize_arabic(text)
    marker = normalize_arabic(ARTICLE_WORD)
    idx = normalized.find(marker)

    if idx < 0:
        return None

    start = idx + len(marker)

    digits = ""

    for ch in normalized[start:start + 16]:
        digit = digit_value(ch)

        if digit is not None:
            digits += str(digit)
            continue

        if digits:
            break

        if ch in {" ", "\n", "\t", ":", "ـ", "-", "،", ".", "/"}:
            continue

        break

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def resolve_article_number(
    *,
    raw_number: int | None,
    last_article_number: int | None,
    pending_context_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    has_major_context = any(block.get("is_major_context") for block in pending_context_blocks)

    if last_article_number is None:
        return {
            "article_number": raw_number,
            "needs_review": raw_number is None,
            "reason": "raw_number_used" if raw_number is not None else "missing_first_article_number",
        }

    expected = last_article_number + 1

    # Important case:
    # Preamble/decree may end at article 3, then the actual law starts again at article 1
    # after a major visual context/title block.
    if has_major_context and raw_number in {None, 1} and last_article_number <= 5:
        return {
            "article_number": 1,
            "needs_review": True,
            "reason": "sequence_reset_after_major_context",
        }

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


def is_context_block(item: dict[str, Any], text: str) -> bool:
    label = str(item.get("label") or "")

    if label not in CONTEXT_LABELS:
        return False

    if detect_article_heading(text)["is_article_heading"]:
        return False

    line_count = len([line for line in text.splitlines() if line.strip()])
    char_count = len("".join(text.split()))

    if label == "image":
        # Image boxes are text containers in this project.
        # Short/medium ones are useful context titles.
        return line_count <= 10 and char_count <= 320

    return line_count <= 6 and char_count <= 220


def is_container_text(item: dict[str, Any], text: str) -> bool:
    label = str(item.get("label") or "")
    line_count = len([line for line in text.splitlines() if line.strip()])
    char_count = len("".join(text.split()))
    article_marker_count = count_article_markers(text)

    if label == "image" and (char_count > 320 or line_count > 10 or article_marker_count >= 2):
        return True

    return False


def make_context_block(item: dict[str, Any], text: str) -> dict[str, Any]:
    label = str(item.get("label") or "")
    area = bbox_area(item.get("bbox"))

    is_major = label in {"image", "doc_title"} and area > 0

    return {
        "box_id": item.get("region_id"),
        "page": item.get("page"),
        "page_number": item.get("page_number"),
        "label": label,
        "bbox": item.get("bbox"),
        "text": normalize_context_text(text),
        "ocr_engine": selected_engine(item),
        "ocr_confidence": selected_confidence(item),
        "selection_reason": selected_reason(item),
        "is_major_context": is_major,
    }


def normalize_context_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) <= 6:
        return " / ".join(lines)

    return text.strip()


def finalize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    parts = [
        str(part).strip()
        for part in chunk.pop("text_parts", [])
        if str(part).strip()
    ]

    final_parts: list[str] = []

    for part in parts:
        if is_duplicate_text(final_parts, part):
            continue

        final_parts.append(part)

    chunk["text"] = "\n".join(final_parts)

    chunk["box_ids"] = unique_keep_order(chunk.get("box_ids") or [])
    chunk["body_box_ids"] = unique_keep_order(chunk.get("body_box_ids") or [])
    chunk["metadata_box_ids"] = unique_keep_order(chunk.get("metadata_box_ids") or [])
    chunk["skipped_duplicate_box_ids"] = unique_keep_order(chunk.get("skipped_duplicate_box_ids") or [])
    chunk["source_pages"] = unique_keep_order(chunk.get("source_pages") or [])
    chunk["references"] = unique_references(chunk.get("references") or [])

    metadata = chunk.setdefault("metadata", {})
    metadata["box_count"] = len(chunk["box_ids"])
    metadata["body_box_count"] = len(chunk["body_box_ids"])
    metadata["metadata_box_count"] = len(chunk["metadata_box_ids"])
    metadata["skipped_duplicate_box_count"] = len(chunk["skipped_duplicate_box_ids"])
    metadata["source_pages"] = list(chunk["source_pages"])
    metadata["reference_count"] = len(chunk["references"])

    return chunk


def references_from_text(text: str, *, skip_heading: bool) -> list[dict[str, Any]]:
    normalized = normalize_arabic(text)
    marker = normalize_arabic(ARTICLE_WORD)

    refs: list[dict[str, Any]] = []

    start = 0

    while True:
        idx = normalized.find(marker, start)

        if idx < 0:
            break

        if skip_heading and idx == 0:
            start = idx + len(marker)
            continue

        number = extract_number_after_index(normalized, idx + len(marker))

        if number is not None:
            refs.append(
                {
                    "article_number": number,
                    "source": "article_word_reference",
                }
            )

        start = idx + len(marker)

    return unique_references(refs)


def extract_number_after_index(text: str, index: int) -> int | None:
    digits = ""

    for ch in text[index:index + 16]:
        digit = digit_value(ch)

        if digit is not None:
            digits += str(digit)
            continue

        if digits:
            break

        if ch in {" ", "\n", "\t", ":", "ـ", "-", "،", ".", "/"}:
            continue

        break

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def resolve_references(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

            resolved.append(item)

        chunk["references"] = resolved

        metadata = chunk.setdefault("metadata", {})
        metadata["reference_count"] = len(resolved)
        metadata["has_article_references"] = bool(resolved)

    return chunks


def is_duplicate_for_chunk(chunk: dict[str, Any], new_text: str) -> bool:
    existing_parts = [str(part) for part in chunk.get("text_parts") or []]

    return is_duplicate_text(existing_parts, new_text)


def is_duplicate_text(existing_parts: list[str], new_text: str) -> bool:
    new_norm = normalize_for_compare(new_text)

    if len(new_norm) < 12:
        return False

    for old_text in existing_parts:
        old_norm = normalize_for_compare(old_text)

        if not old_norm:
            continue

        if new_norm in old_norm:
            return True

        if old_norm in new_norm and len(old_norm) >= 30:
            return True

        if rough_overlap(new_norm, old_norm) >= 0.88:
            return True

    return False


def rough_overlap(a: str, b: str) -> float:
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


def count_article_markers(text: str) -> int:
    normalized = normalize_arabic(text)
    marker = normalize_arabic(ARTICLE_WORD)

    count = 0
    start = 0

    while True:
        idx = normalized.find(marker, start)

        if idx < 0:
            break

        count += 1
        start = idx + len(marker)

    return count


def selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def selected_engine(item: dict[str, Any]) -> str | None:
    return ((item.get("ocr") or {}).get("selected") or {}).get("engine")


def selected_confidence(item: dict[str, Any]) -> Any:
    return ((item.get("ocr") or {}).get("selected") or {}).get("confidence")


def selected_reason(item: dict[str, Any]) -> Any:
    return ((item.get("ocr") or {}).get("selected") or {}).get("selection_reason")


def is_page_number_text(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    if len(stripped) > 4:
        return False

    return all(ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩" for ch in stripped)


def is_noise_text(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return True

    if stripped.startswith("/tmp/") or stripped.startswith("tmp/"):
        return True

    if len(stripped) <= 2 and not any("\u0600" <= ch <= "\u06ff" for ch in stripped):
        return True

    return False


def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    page = str(item.get("page") or "")
    digits = "".join(ch for ch in page if ch.isdigit())
    page_number = int(digits) if digits else int(item.get("page_number") or 0)

    return (
        page_number,
        int(item.get("reading_order_start") or 0),
        str(item.get("region_id") or ""),
    )


def unique_keep_order(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()

    for value in values:
        key = str(value)

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def unique_context_blocks(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for value in values:
        key = str(value.get("box_id"))

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def unique_references(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for value in values:
        key = str(value.get("article_number"))

        if key in seen:
            continue

        seen.add(key)
        output.append(value)

    return output


def normalize_arabic(text: str) -> str:
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .strip()
    )


def normalize_for_compare(text: str) -> str:
    return "".join(normalize_arabic(text).split())


def digit_value(ch: str) -> int | None:
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"

    if ch in arabic_digits:
        return arabic_digits.index(ch)

    if ch.isdigit():
        return int(ch)

    return None


def bbox_area(value: object) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return 0

    return max(0, x2 - x1) * max(0, y2 - y1)