from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from ocr_chunker.io.loaders import load_boxes_csv


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-csv", required=True)
    parser.add_argument("--page-images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument(
        "--document-has-real-images",
        action="store_true",
        help="If not set, boxes labeled image are still treated as OCR text candidates.",
    )

    args = parser.parse_args()

    boxes = load_boxes_csv(args.boxes_csv)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    page_images_dir = Path(args.page_images_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pages: dict[str, list[Any]] = {}

    for box in boxes:
        pages.setdefault(box.page, []).append(box)

    manifest: list[dict[str, Any]] = []

    for page_name, page_boxes in sorted(pages.items()):
        image_path = _find_page_image(page_name, page_images_dir)

        if image_path is None:
            print(f"WARNING: no page image found for {page_name}")
            continue

        image = Image.open(image_path).convert("RGB")
        page_width, page_height = image.size

        page_output_dir = output_dir / page_name
        page_output_dir.mkdir(parents=True, exist_ok=True)

        page_boxes = sorted(
            page_boxes,
            key=lambda box: (
                box.bbox[1],
                -box.bbox[2],
                box.bbox[0],
                box.box_id,
            ),
        )

        for order, box in enumerate(page_boxes, start=1):
            padded_bbox = _pad_bbox(
                box.bbox,
                page_width=page_width,
                page_height=page_height,
                padding_px=args.padding,
            )

            crop = image.crop(padded_bbox)

            safe_box_id = box.box_id.replace("\\", "_").replace("/", "_")
            crop_path = page_output_dir / f"{safe_box_id}.png"
            crop.save(crop_path)

            label = str(box.label or "unknown")

            should_ocr = True

            if args.document_has_real_images and label == "image":
                should_ocr = False

            manifest.append(
                {
                    "region_id": box.box_id,
                    "box_id": box.box_id,
                    "page": box.page,
                    "page_number": box.page_number,
                    "crop_path": str(crop_path),
                    "source_image_path": str(image_path),
                    "bbox": list(box.bbox),
                    "padded_bbox": list(padded_bbox),
                    "reading_order_start": order,
                    "reading_order_end": order,
                    "label": label,
                    "layout_score": box.layout_score,
                    "region_role": f"raw_{label}",
                    "document_has_real_images": bool(args.document_has_real_images),
                    "layout_label_says_image_but_ocr_enabled": label == "image" and should_ocr,
                    "needs_ocr": should_ocr,
                    "needs_review": False,
                    "review_reasons": [],
                    "suggested_actions": [],
                    "quality_score": 1.0,
                    "ocr": {
                        "paddle": None,
                        "easyocr": None,
                        "selected": None,
                    },
                }
            )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    image_boxes = sum(1 for item in manifest if item.get("label") == "image")

    print(f"OK: raw_boxes={len(manifest)} image_boxes_included={image_boxes}")
    print(f"Wrote crops dir: {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


def _find_page_image(page_name: str, page_images_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = page_images_dir / f"{page_name}{ext}"

        if candidate.exists():
            return candidate

    return None


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


if __name__ == "__main__":
    main()