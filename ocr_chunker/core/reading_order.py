from __future__ import annotations

from collections import defaultdict

from ocr_chunker.core.geometry import height, vertical_overlap_ratio
from ocr_chunker.schemas import LayoutBox, OrderedBox


def assign_arabic_reading_order(
    boxes: list[LayoutBox],
    *,
    same_line_overlap_threshold: float = 0.35,
) -> list[OrderedBox]:
    pages: dict[str, list[LayoutBox]] = defaultdict(list)

    for box in boxes:
        pages[box.page].append(box)

    output: list[OrderedBox] = []

    for page in sorted(pages.keys(), key=_page_sort_key):
        page_boxes = pages[page]
        ordered = _order_single_page(
            page_boxes,
            same_line_overlap_threshold=same_line_overlap_threshold,
        )

        for idx, box in enumerate(ordered, start=1):
            output.append(OrderedBox(box=box, reading_order=idx))

    return output


def _order_single_page(
    boxes: list[LayoutBox],
    *,
    same_line_overlap_threshold: float,
) -> list[LayoutBox]:
    remaining = sorted(boxes, key=lambda b: (b.bbox[1], b.bbox[0]))
    bands: list[list[LayoutBox]] = []

    for box in remaining:
        placed = False

        for band in bands:
            if _belongs_to_band(box, band, same_line_overlap_threshold):
                band.append(box)
                placed = True
                break

        if not placed:
            bands.append([box])

    bands.sort(key=lambda band: min(b.bbox[1] for b in band))

    ordered: list[LayoutBox] = []

    for band in bands:
        # Arabic reading order inside same horizontal band:
        # right to left, so x2 descending.
        ordered.extend(sorted(band, key=lambda b: (-b.bbox[2], b.bbox[1])))

    return ordered


def _belongs_to_band(
    box: LayoutBox,
    band: list[LayoutBox],
    same_line_overlap_threshold: float,
) -> bool:
    if not band:
        return False

    return any(
        vertical_overlap_ratio(box.bbox, other.bbox) >= same_line_overlap_threshold
        or abs(box.bbox[1] - other.bbox[1]) <= max(12, int(height(other.bbox) * 0.35))
        for other in band
    )


def _page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)