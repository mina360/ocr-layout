from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def render_overlays_from_json(
    *,
    merged_regions_path: str | Path,
    page_images_dir: str | Path,
    output_dir: str | Path,
) -> None:
    merged_regions_path = Path(merged_regions_path)
    page_images_dir = Path(page_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with merged_regions_path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    for page in pages:
        image_path = _find_page_image(page, page_images_dir)

        if image_path is None:
            print(f"WARNING: no image found for page={page.get('page')}")
            continue

        output_path = output_dir / f"{page.get('page')}_regions.png"
        _render_single_page(page, image_path, output_path)

    print(f"Overlay images written to: {output_dir}")


def _render_single_page(
    page: dict[str, Any],
    image_path: Path,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for candidate in page.get("page_number_candidates", []):
        box = candidate.get("box", {})
        bbox = _as_bbox(box.get("bbox"))
        if bbox:
            _draw_box(
                draw,
                bbox,
                label="PAGE",
                color=(60, 120, 255),
                font=font,
                width=3,
            )

    for region in page.get("regions", []):
        bbox = _as_bbox(region.get("bbox"))
        if not bbox:
            continue

        needs_review = bool(region.get("needs_review"))
        quality_score = region.get("quality_score", 1.0)
        region_id = region.get("region_id", "region")
        start = region.get("reading_order_start")
        end = region.get("reading_order_end")

        color = (230, 70, 70) if needs_review else (40, 170, 80)

        label = f"{region_id} [{start}-{end}] q={quality_score}"
        if needs_review:
            label += " REVIEW"

        _draw_box(
            draw,
            bbox,
            label=label,
            color=color,
            font=font,
            width=4 if needs_review else 3,
        )

    image.save(output_path)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    *,
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    width: int,
) -> None:
    x1, y1, x2, y2 = bbox

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    label_x = x1
    label_y = max(0, y1 - 16)

    draw.rectangle(
        [label_x, label_y, label_x + max(80, len(label) * 7), label_y + 14],
        fill=color,
    )
    draw.text((label_x + 3, label_y + 2), label, fill=(255, 255, 255), font=font)


def _find_page_image(page: dict[str, Any], page_images_dir: Path) -> Path | None:
    page_name = str(page.get("page") or "").strip()
    page_number = str(page.get("page_number") or "").strip()

    candidates: list[str] = []

    if page_name:
        candidates.extend(
            [
                page_name,
                page_name.lower(),
                page_name.upper(),
            ]
        )

    if page_number:
        candidates.extend(
            [
                f"page_{int(page_number):04d}",
                f"page_{int(page_number)}",
                page_number,
            ]
        )

    for stem in candidates:
        for ext in IMAGE_EXTENSIONS:
            candidate = page_images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    return None


def _as_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None