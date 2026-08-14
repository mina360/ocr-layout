from __future__ import annotations

from ocr_chunker.schemas import BBox


def width(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0])


def height(bbox: BBox) -> int:
    return max(0, bbox[3] - bbox[1])


def area(bbox: BBox) -> int:
    return width(bbox) * height(bbox)


def union_bbox(boxes: list[BBox]) -> BBox:
    if not boxes:
        raise ValueError("union_bbox requires at least one bbox")

    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def vertical_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    denom = max(1, min(height(a), height(b)))
    return overlap / denom


def horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = max(1, min(width(a), width(b)))
    return overlap / denom


def vertical_gap(a: BBox, b: BBox) -> int:
    if b[1] >= a[3]:
        return b[1] - a[3]
    if a[1] >= b[3]:
        return a[1] - b[3]
    return 0


def x_center(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2


def y_center(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2