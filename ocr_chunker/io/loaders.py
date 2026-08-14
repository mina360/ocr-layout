from __future__ import annotations

import csv
import json
from pathlib import Path

from ocr_chunker.schemas import LayoutBox


def load_final_results(path: str | Path) -> list[LayoutBox]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    boxes: list[LayoutBox] = []

    for index, row in enumerate(rows, start=1):
        box_id = str(row.get("crop_id") or row.get("box_id") or "").strip()
        page = str(row.get("page") or "").strip()

        if not page:
            page_number = row.get("page_number") or row.get("page_index")
            page = _page_name_from_number(page_number)

        if not box_id:
            box_id = f"{page}_box_{index:04d}"

        if not page:
            continue

        try:
            bbox = (
                int(float(row["x1"])),
                int(float(row["y1"])),
                int(float(row["x2"])),
                int(float(row["y2"])),
            )
        except KeyError as exc:
            raise ValueError(f"Missing bbox field in row {box_id}: {exc}") from exc

        boxes.append(
            LayoutBox(
                box_id=box_id,
                page=_normalize_page_name(page),
                page_number=int(row.get("page_number") or _digits_or_zero(page)),
                label=str(row.get("label") or "unknown"),
                bbox=bbox,
                text=str(row.get("text") or ""),
                layout_score=_safe_float(row.get("layout_score") or row.get("score")),
                ocr_score=_safe_float(row.get("final_score") or row.get("ocr_score")),
                crop_path=row.get("crop_path"),
                source=row,
            )
        )

    return boxes


def load_boxes_csv(path: str | Path) -> list[LayoutBox]:
    """
    Load raw layout boxes from boxes.csv.

    This should be the source of truth for layout detection.
    It does not require OCR text.
    """
    path = Path(path)
    boxes: list[LayoutBox] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            page = _get_first(row, ["page", "page_name", "image", "file", "filename"])
            page_number = _get_first(row, ["page_number", "page_index", "page_idx"])

            if not page:
                page = _page_name_from_number(page_number)

            if not page:
                continue

            page = _normalize_page_name(page)

            try:
                bbox = (
                    int(float(_require_first(row, ["x1", "left", "xmin"]))),
                    int(float(_require_first(row, ["y1", "top", "ymin"]))),
                    int(float(_require_first(row, ["x2", "right", "xmax"]))),
                    int(float(_require_first(row, ["y2", "bottom", "ymax"]))),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid bbox in boxes.csv row {index}: {row}") from exc

            box_id = (
                _get_first(row, ["crop_id", "box_id", "id"])
                or f"{page}_raw_{index:04d}"
            )

            label = _get_first(row, ["label", "class", "type"]) or "unknown"
            score = _get_first(row, ["score", "layout_score", "confidence"])

            boxes.append(
                LayoutBox(
                    box_id=str(box_id),
                    page=page,
                    page_number=int(float(page_number)) if page_number else _digits_or_zero(page),
                    label=str(label),
                    bbox=bbox,
                    text="",  # raw boxes do not have OCR text yet
                    layout_score=_safe_float(score),
                    ocr_score=None,
                    crop_path=None,
                    source=dict(row),
                )
            )

    return boxes


def _get_first(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _require_first(row: dict, keys: list[str]) -> str:
    value = _get_first(row, keys)
    if value is None:
        raise ValueError(f"Missing required column. Tried: {keys}")
    return value


def _normalize_page_name(value: str) -> str:
    stem = Path(str(value)).stem

    if stem.startswith("page_"):
        return stem

    digits = "".join(ch for ch in stem if ch.isdigit())

    if digits:
        return f"page_{int(digits):04d}"

    return stem


def _page_name_from_number(value: object) -> str:
    if value is None or str(value).strip() == "":
        return ""

    return f"page_{int(float(str(value))):04d}"


def _digits_or_zero(value: object) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


def _safe_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None