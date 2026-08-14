from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_chunker.schemas import LayoutBox


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def recover_missing_layout_boxes(
    boxes: list[LayoutBox],
    *,
    page_images_dir: str | Path,
    ink_threshold: int = 210,
    mask_padding_px: int = 8,
    output_padding_px: int = 8,
    min_ink_pixels_per_row: int = 5,
    max_row_gap_px: int = 10,
    min_box_width_px: int = 35,
    min_box_height_px: int = 10,
    max_recovered_boxes_per_page: int = 30,
) -> list[LayoutBox]:
    """
    Add synthetic boxes for visible ink that was not covered by layout detection.

    This layer does not use OCR and does not use regex.
    It only checks page images for uncovered dark pixels.

    It is intentionally conservative:
    - recovered boxes are marked as label='recovered_region'
    - later stages should mark them for review
    """
    pages: dict[str, list[LayoutBox]] = defaultdict(list)

    for box in boxes:
        pages[box.page].append(box)

    recovered: list[LayoutBox] = []

    for page_name, page_boxes in pages.items():
        image_path = _find_page_image(page_name, Path(page_images_dir))

        if image_path is None:
            continue

        page_recovered = _recover_for_page(
            page_name=page_name,
            page_boxes=page_boxes,
            image_path=image_path,
            ink_threshold=ink_threshold,
            mask_padding_px=mask_padding_px,
            output_padding_px=output_padding_px,
            min_ink_pixels_per_row=min_ink_pixels_per_row,
            max_row_gap_px=max_row_gap_px,
            min_box_width_px=min_box_width_px,
            min_box_height_px=min_box_height_px,
            max_recovered_boxes_per_page=max_recovered_boxes_per_page,
        )

        recovered.extend(page_recovered)

    return boxes + recovered


def _recover_for_page(
    *,
    page_name: str,
    page_boxes: list[LayoutBox],
    image_path: Path,
    ink_threshold: int,
    mask_padding_px: int,
    output_padding_px: int,
    min_ink_pixels_per_row: int,
    max_row_gap_px: int,
    min_box_width_px: int,
    min_box_height_px: int,
    max_recovered_boxes_per_page: int,
) -> list[LayoutBox]:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    page_height, page_width = arr.shape

    # Dark pixels are likely printed text.
    content_mask = arr < ink_threshold

    # Remove areas already covered by layout boxes.
    for box in page_boxes:
        x1, y1, x2, y2 = box.bbox

        x1 = max(0, x1 - mask_padding_px)
        y1 = max(0, y1 - mask_padding_px)
        x2 = min(page_width, x2 + mask_padding_px)
        y2 = min(page_height, y2 + mask_padding_px)

        content_mask[y1:y2, x1:x2] = False

    row_counts = content_mask.sum(axis=1)
    active_rows = row_counts >= min_ink_pixels_per_row
    row_bands = _group_active_rows(active_rows, max_gap=max_row_gap_px)

    recovered: list[LayoutBox] = []
    page_number = _infer_page_number(page_name, page_boxes)

    for index, (y1, y2) in enumerate(row_bands, start=1):
        if len(recovered) >= max_recovered_boxes_per_page:
            break

        band_mask = content_mask[y1:y2, :]
        col_counts = band_mask.sum(axis=0)
        active_cols = np.where(col_counts > 0)[0]

        if active_cols.size == 0:
            continue

        x1 = int(active_cols.min())
        x2 = int(active_cols.max()) + 1

        x1 = max(0, x1 - output_padding_px)
        y1_padded = max(0, y1 - output_padding_px)
        x2 = min(page_width, x2 + output_padding_px)
        y2_padded = min(page_height, y2 + output_padding_px)

        box_width = x2 - x1
        box_height = y2_padded - y1_padded

        if box_width < min_box_width_px or box_height < min_box_height_px:
            continue

        recovered.append(
            LayoutBox(
                box_id=f"{page_name}_recovered_{index:03d}",
                page=page_name,
                page_number=page_number,
                label="recovered_region",
                bbox=(x1, y1_padded, x2, y2_padded),
                text="",
                layout_score=None,
                ocr_score=None,
                crop_path=None,
                source={
                    "source": "coverage_recovery",
                    "image_path": str(image_path),
                    "ink_threshold": ink_threshold,
                },
            )
        )

    return recovered


def _group_active_rows(
    active_rows: np.ndarray,
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []

    start: int | None = None
    last_active: int | None = None

    for row_index, is_active in enumerate(active_rows):
        if is_active:
            if start is None:
                start = row_index
            last_active = row_index
            continue

        if start is not None and last_active is not None:
            if row_index - last_active > max_gap:
                bands.append((start, last_active + 1))
                start = None
                last_active = None

    if start is not None and last_active is not None:
        bands.append((start, last_active + 1))

    return bands


def _infer_page_number(page_name: str, page_boxes: list[LayoutBox]) -> int:
    if page_boxes:
        return page_boxes[0].page_number

    digits = "".join(ch for ch in page_name if ch.isdigit())
    return int(digits) if digits else 0


def _find_page_image(page_name: str, page_images_dir: Path) -> Path | None:
    candidates = [
        page_name,
        page_name.lower(),
        page_name.upper(),
    ]

    for stem in candidates:
        for ext in IMAGE_EXTENSIONS:
            path = page_images_dir / f"{stem}{ext}"
            if path.exists():
                return path

    return None