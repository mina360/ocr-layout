from __future__ import annotations

from collections import defaultdict

from ocr_chunker.core.geometry import (
    horizontal_overlap_ratio,
    union_bbox,
    vertical_gap,
    x_center,
)
from ocr_chunker.core.review_policy import assess_region_quality
from ocr_chunker.schemas import MergedRegion, OrderedBox, PageRegions


BOUNDARY_LABELS = {
    "doc_title",
    "header",
    "paragraph_title",
    "figure_title",
    "recovered_region",
}

BODY_LABELS = {
    "text",
    "reference_content",
}


def merge_ordered_boxes_into_regions(
    boxes: list[OrderedBox],
    *,
    max_vertical_gap_px: int = 45,
    min_horizontal_overlap: float = 0.25,
    title_to_body_max_gap_px: int = 280,
    title_to_body_min_overlap: float = 0.10,
) -> list[PageRegions]:
    pages: dict[str, list[OrderedBox]] = defaultdict(list)

    for box in boxes:
        pages[box.box.page].append(box)

    result: list[PageRegions] = []

    for page in sorted(pages.keys(), key=_page_sort_key):
        page_boxes = sorted(pages[page], key=lambda b: b.reading_order)

        page_number_candidates = [
            box for box in page_boxes
            if box.is_page_number_candidate
        ]

        content_boxes = [
            box
            for box in page_boxes
            if not box.is_page_number_candidate and not box.is_noise_candidate
        ]

        page_width = max((box.box.bbox[2] for box in page_boxes), default=None)
        page_height = max((box.box.bbox[3] for box in page_boxes), default=None)
        page_number_value = page_boxes[0].box.page_number if page_boxes else 0

        regions = _merge_single_page(
            content_boxes,
            page_width=page_width,
            page_height=page_height,
            max_vertical_gap_px=max_vertical_gap_px,
            min_horizontal_overlap=min_horizontal_overlap,
            title_to_body_max_gap_px=title_to_body_max_gap_px,
            title_to_body_min_overlap=title_to_body_min_overlap,
        )

        result.append(
            PageRegions(
                page=page,
                page_number=page_number_value,
                width=page_width,
                height=page_height,
                page_number_candidates=page_number_candidates,
                regions=regions,
            )
        )

    return result


def _merge_single_page(
    boxes: list[OrderedBox],
    *,
    page_width: int | None,
    page_height: int | None,
    max_vertical_gap_px: int,
    min_horizontal_overlap: float,
    title_to_body_max_gap_px: int,
    title_to_body_min_overlap: float,
) -> list[MergedRegion]:
    if not boxes:
        return []

    regions: list[MergedRegion] = []
    current: list[OrderedBox] = [boxes[0]]
    reasons: list[str] = ["start_region"]

    for curr in boxes[1:]:
        prev = current[-1]

        if _should_start_new_region(prev, curr, current):
            regions.append(
                _make_region(
                    current,
                    len(regions) + 1,
                    reasons,
                    page_width=page_width,
                    page_height=page_height,
                )
            )
            current = [curr]
            reasons = ["new_region_at_layout_boundary"]
            continue

        if _should_merge(
            prev,
            curr,
            max_vertical_gap_px=max_vertical_gap_px,
            min_horizontal_overlap=min_horizontal_overlap,
            title_to_body_max_gap_px=title_to_body_max_gap_px,
            title_to_body_min_overlap=title_to_body_min_overlap,
        ):
            current.append(curr)

            if _is_title_fragment_continuation(prev, curr):
                reasons.append("title_fragment_continuation")
            elif _is_boundary_label(prev) and _is_body_label(curr):
                reasons.append("title_followed_by_body")
            elif _is_body_label(prev) and _is_body_label(curr):
                reasons.append("body_continuation")
            else:
                reasons.append("contiguous_close_boxes")
        else:
            regions.append(
                _make_region(
                    current,
                    len(regions) + 1,
                    reasons,
                    page_width=page_width,
                    page_height=page_height,
                )
            )
            current = [curr]
            reasons = ["new_region_after_gap_or_column_change"]

    regions.append(
        _make_region(
            current,
            len(regions) + 1,
            reasons,
            page_width=page_width,
            page_height=page_height,
        )
    )

    return regions


def _should_start_new_region(
    prev: OrderedBox,
    curr: OrderedBox,
    current_region: list[OrderedBox],
) -> bool:
    if curr.reading_order != prev.reading_order + 1:
        return True

    if not current_region:
        return False

    # Consecutive title fragments should remain together when geometry agrees.
    if _is_title_fragment_continuation(prev, curr):
        return False

    # A new title/header-like layout box usually means a new logical region.
    if _is_boundary_label(curr):
        return True

    return False


def _should_merge(
    prev: OrderedBox,
    curr: OrderedBox,
    *,
    max_vertical_gap_px: int,
    min_horizontal_overlap: float,
    title_to_body_max_gap_px: int,
    title_to_body_min_overlap: float,
) -> bool:
    if curr.reading_order != prev.reading_order + 1:
        return False

    prev_bbox = prev.box.bbox
    curr_bbox = curr.box.bbox

    gap = vertical_gap(prev_bbox, curr_bbox)
    overlap = horizontal_overlap_ratio(prev_bbox, curr_bbox)

    if _is_title_fragment_continuation(prev, curr):
        return True

    if _is_boundary_label(prev) and _is_body_label(curr):
        return gap <= title_to_body_max_gap_px and overlap >= title_to_body_min_overlap

    if _is_body_label(prev) and _is_body_label(curr):
        return gap <= 320 and overlap >= 0.45

    return gap <= max_vertical_gap_px and overlap >= min_horizontal_overlap


def _is_title_fragment_continuation(prev: OrderedBox, curr: OrderedBox) -> bool:
    if not (_is_boundary_label(prev) and _is_boundary_label(curr)):
        return False

    prev_bbox = prev.box.bbox
    curr_bbox = curr.box.bbox

    gap = vertical_gap(prev_bbox, curr_bbox)
    overlap = horizontal_overlap_ratio(prev_bbox, curr_bbox)

    prev_center_x = x_center(prev_bbox)
    curr_center_x = x_center(curr_bbox)
    center_distance = abs(prev_center_x - curr_center_x)

    return gap <= 90 and overlap >= 0.35 and center_distance <= 180


def _make_region(
    boxes: list[OrderedBox],
    region_index: int,
    merge_reasons: list[str],
    *,
    page_width: int | None,
    page_height: int | None,
) -> MergedRegion:
    first = boxes[0]
    last = boxes[-1]
    bboxes = [box.box.bbox for box in boxes]
    texts = [box.box.text.strip() for box in boxes if box.box.text.strip()]

    review = assess_region_quality(
        boxes,
        page_width=page_width,
        page_height=page_height,
    )

    return MergedRegion(
        region_id=f"{first.box.page}_region_{region_index:03d}",
        page=first.box.page,
        page_number=first.box.page_number,
        reading_order_start=first.reading_order,
        reading_order_end=last.reading_order,
        bbox=union_bbox(bboxes),
        child_box_ids=[box.box.box_id for box in boxes],
        region_role=_guess_region_role(boxes),
        text="\n".join(texts),
        needs_ocr=not bool(texts),
        needs_review=review.needs_review,
        merge_reasons=list(dict.fromkeys(merge_reasons)),
        review_reasons=review.review_reasons,
        suggested_actions=review.suggested_actions,
        quality_score=review.quality_score,
    )


def _guess_region_role(boxes: list[OrderedBox]) -> str:
    labels = [box.box.label for box in boxes]

    if not labels:
        return "unknown"

    if labels[0] == "doc_title":
        return "document_title"

    if labels[0] == "header":
        return "section_header"

    if labels[0] in {"paragraph_title", "figure_title"}:
        return "content_block"

    if labels[0] == "recovered_region":
        return "recovered_content"

    return "body_text"


def _is_boundary_label(box: OrderedBox) -> bool:
    return box.box.label in BOUNDARY_LABELS


def _is_body_label(box: OrderedBox) -> bool:
    return box.box.label in BODY_LABELS or box.box.label == "unknown"


def _page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)