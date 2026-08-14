from __future__ import annotations

from dataclasses import dataclass

from ocr_chunker.core.geometry import area, height, union_bbox, vertical_gap, width
from ocr_chunker.schemas import OrderedBox


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
    "unknown",
}


@dataclass(frozen=True)
class ReviewDecision:
    needs_review: bool
    review_reasons: list[str]
    suggested_actions: list[str]
    quality_score: float


def assess_region_quality(
    boxes: list[OrderedBox],
    *,
    page_width: int | None = None,
    page_height: int | None = None,
) -> ReviewDecision:
    """
    Conservative quality policy for OCR layout regions.

    This does not use regex and does not try to understand legal text.
    It only marks suspicious layout/grouping situations for later
    visual review, OCR-based classification, or human correction.
    """
    reasons: list[str] = []
    actions: list[str] = []

    if not boxes:
        return ReviewDecision(
            needs_review=True,
            review_reasons=["empty_region"],
            suggested_actions=["discard_or_investigate_region"],
            quality_score=0.0,
        )

    labels = [box.box.label for box in boxes]
    boundary_count = sum(1 for box in boxes if _is_boundary_label(box))
    body_count = sum(1 for box in boxes if _is_body_label(box))

    if "recovered_region" in labels:
        reasons.append("coverage_recovered_region")
        actions.append("review_recovered_text_area")

    region_bbox = union_bbox([box.box.bbox for box in boxes])

    if len(boxes) == 1 and labels[0] in {"header", "paragraph_title", "figure_title"}:
        reasons.append("single_boundary_region")
        actions.append("review_merge_with_next_or_keep_as_title")

    if boundary_count >= 3:
        reasons.append("many_boundary_boxes_in_one_region")
        actions.append("review_split_region")

    if boundary_count >= 2 and body_count >= 1:
        reasons.append("mixed_multiple_titles_with_body")
        actions.append("review_split_or_keep")

    if _has_large_internal_gap(boxes):
        reasons.append("large_gap_between_child_boxes")
        actions.append("review_split_region")

    if page_width and page_height:
        page_area = page_width * page_height
        region_area = area(region_bbox)

        if page_area > 0 and region_area / page_area >= 0.45:
            reasons.append("region_covers_large_page_area")
            actions.append("review_split_region")

        if height(region_bbox) >= page_height * 0.60 and len(boxes) >= 5:
            reasons.append("very_tall_multi_box_region")
            actions.append("review_split_region")

    low_score_count = sum(
        1
        for box in boxes
        if box.box.layout_score is not None and box.box.layout_score < 0.35
    )

    if low_score_count >= 2:
        reasons.append("multiple_low_layout_scores")
        actions.append("review_boxes_or_redetect_layout")

    quality_score = _compute_quality_score(reasons)
    needs_review = bool(reasons) or quality_score < 0.78

    return ReviewDecision(
        needs_review=needs_review,
        review_reasons=list(dict.fromkeys(reasons)),
        suggested_actions=list(dict.fromkeys(actions)),
        quality_score=quality_score,
    )


def _has_large_internal_gap(boxes: list[OrderedBox]) -> bool:
    if len(boxes) < 2:
        return False

    ordered = sorted(boxes, key=lambda box: box.reading_order)

    gaps: list[int] = []

    for prev, curr in zip(ordered, ordered[1:]):
        gaps.append(vertical_gap(prev.box.bbox, curr.box.bbox))

    if not gaps:
        return False

    max_gap = max(gaps)
    region_height = height(union_bbox([box.box.bbox for box in ordered]))

    return max_gap >= 260 and region_height >= 600


def _compute_quality_score(reasons: list[str]) -> float:
    score = 1.0

    penalties = {
        "single_boundary_region": 0.10,
        "many_boundary_boxes_in_one_region": 0.30,
        "mixed_multiple_titles_with_body": 0.22,
        "large_gap_between_child_boxes": 0.18,
        "region_covers_large_page_area": 0.25,
        "very_tall_multi_box_region": 0.25,
        "multiple_low_layout_scores": 0.15,
        "empty_region": 1.0,
        "coverage_recovered_region": 0.12,
    }

    for reason in reasons:
        score -= penalties.get(reason, 0.10)

    return round(max(0.0, min(1.0, score)), 3)


def _is_boundary_label(box: OrderedBox) -> bool:
    return box.box.label in BOUNDARY_LABELS


def _is_body_label(box: OrderedBox) -> bool:
    return box.box.label in BODY_LABELS