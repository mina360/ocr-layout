from __future__ import annotations

from collections import defaultdict

from ocr_chunker.core.geometry import area, height, width, x_center, y_center
from ocr_chunker.schemas import OrderedBox


PAGE_NUMBER_LABEL_HINTS = {
    "number",
    "footer",
}


def mark_page_number_candidates(
    boxes: list[OrderedBox],
    *,
    page_width: int | None = None,
    page_height: int | None = None,
    allow_top_page_numbers: bool = False,
) -> list[OrderedBox]:
    """
    Mark printed page-number candidates without removing important titles.

    Important rule:
    - By default, only bottom page numbers are detected.
    - Top centered titles must NOT be treated as page numbers.
    """
    if not boxes:
        return []

    pages: dict[str, list[OrderedBox]] = defaultdict(list)

    for box in boxes:
        pages[box.box.page].append(box)

    output: list[OrderedBox] = []

    for page_name in sorted(pages.keys(), key=_page_sort_key):
        page_boxes = pages[page_name]

        inferred_width = page_width or max(b.box.bbox[2] for b in page_boxes)
        inferred_height = page_height or max(b.box.bbox[3] for b in page_boxes)
        page_area = inferred_width * inferred_height

        for ordered in page_boxes:
            is_candidate = _is_page_number_candidate(
                ordered,
                page_width=inferred_width,
                page_height=inferred_height,
                page_area=page_area,
                allow_top_page_numbers=allow_top_page_numbers,
            )

            output.append(
                OrderedBox(
                    box=ordered.box,
                    reading_order=ordered.reading_order,
                    is_page_number_candidate=is_candidate,
                    is_noise_candidate=ordered.is_noise_candidate,
                )
            )

    return output


def _is_page_number_candidate(
    ordered: OrderedBox,
    *,
    page_width: int,
    page_height: int,
    page_area: int,
    allow_top_page_numbers: bool,
) -> bool:
    bbox = ordered.box.bbox
    label = ordered.box.label

    box_area = area(bbox)
    box_width = width(bbox)
    box_height = height(bbox)
    cx = x_center(bbox)
    cy = y_center(bbox)

    near_bottom = cy >= page_height * 0.88
    near_top = cy <= page_height * 0.07

    if not near_bottom and not (allow_top_page_numbers and near_top):
        return False

    small_area = box_area <= page_area * 0.012
    short_box = box_height <= page_height * 0.035
    not_too_wide = box_width <= page_width * 0.20

    centered_or_edge = (
        page_width * 0.35 <= cx <= page_width * 0.65
        or cx <= page_width * 0.18
        or cx >= page_width * 0.82
    )

    has_page_number_label_hint = label in PAGE_NUMBER_LABEL_HINTS

    return (
        small_area
        and short_box
        and not_too_wide
        and centered_or_edge
        and (near_bottom or has_page_number_label_hint)
    )


def _page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)