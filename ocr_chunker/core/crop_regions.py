from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def crop_regions_from_merged_json(
    *,
    merged_regions_path: str | Path,
    page_images_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    padding_px: int = 10,
) -> None:
    merged_regions_path = Path(merged_regions_path)
    page_images_dir = Path(page_images_dir)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with merged_regions_path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    crops_manifest: list[dict[str, Any]] = []

    for page in pages:
        page_name = str(page.get("page") or "").strip()
        image_path = _find_page_image(page_name, page_images_dir)

        if image_path is None:
            print(f"WARNING: no page image found for {page_name}")
            continue

        image = Image.open(image_path).convert("RGB")
        page_width, page_height = image.size

        page_output_dir = output_dir / page_name
        page_output_dir.mkdir(parents=True, exist_ok=True)

        for region in page.get("regions", []):
            region_id = str(region.get("region_id") or "").strip()
            bbox = _as_bbox(region.get("bbox"))

            if not region_id or bbox is None:
                continue

            padded_bbox = _pad_bbox(
                bbox,
                page_width=page_width,
                page_height=page_height,
                padding_px=padding_px,
            )

            crop = image.crop(padded_bbox)
            crop_filename = f"{region_id}.png"
            crop_path = page_output_dir / crop_filename
            crop.save(crop_path)

            crops_manifest.append(
                {
                    "region_id": region_id,
                    "page": page_name,
                    "page_number": page.get("page_number"),
                    "crop_path": str(crop_path),
                    "source_image_path": str(image_path),
                    "bbox": list(bbox),
                    "padded_bbox": list(padded_bbox),
                    "reading_order_start": region.get("reading_order_start"),
                    "reading_order_end": region.get("reading_order_end"),
                    "child_box_ids": region.get("child_box_ids", []),
                    "region_role": region.get("region_role", "unknown"),
                    "needs_review": region.get("needs_review", False),
                    "review_reasons": region.get("review_reasons", []),
                    "suggested_actions": region.get("suggested_actions", []),
                    "quality_score": region.get("quality_score", 1.0),
                    "ocr": {
                        "paddle": None,
                        "easyocr": None,
                        "selected": None,
                    },
                }
            )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(crops_manifest, f, ensure_ascii=False, indent=2)

    print(f"OK: cropped_regions={len(crops_manifest)}")
    print(f"Wrote crops dir: {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


def _find_page_image(page_name: str, page_images_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = page_images_dir / f"{page_name}{ext}"
        if candidate.exists():
            return candidate

    return None


def _as_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    page_width: int,
    page_height: int,
    padding_px: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    return (
        max(0, x1 - padding_px),
        max(0, y1 - padding_px),
        min(page_width, x2 + padding_px),
        min(page_height, y2 + padding_px),
    )