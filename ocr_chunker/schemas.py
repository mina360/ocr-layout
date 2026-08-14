from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutBox:
    box_id: str
    page: str
    page_number: int
    label: str
    bbox: BBox
    text: str = ""
    layout_score: float | None = None
    ocr_score: float | None = None
    crop_path: str | None = None
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderedBox:
    box: LayoutBox
    reading_order: int
    is_page_number_candidate: bool = False
    is_noise_candidate: bool = False


@dataclass(frozen=True)
class MergedRegion:
    region_id: str
    page: str
    page_number: int
    reading_order_start: int
    reading_order_end: int
    bbox: BBox
    child_box_ids: list[str]
    region_role: str = "unknown"
    text: str = ""
    needs_ocr: bool = True
    needs_review: bool = False
    merge_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    quality_score: float = 1.0


@dataclass(frozen=True)
class PageRegions:
    page: str
    page_number: int
    width: int | None
    height: int | None
    page_number_candidates: list[OrderedBox]
    regions: list[MergedRegion]