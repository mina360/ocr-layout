from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes-csv", required=True)
    parser.add_argument("--page-images-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    args = parser.parse_args()

    boxes = _load_boxes_csv(Path(args.boxes_csv))
    render_raw_box_overlays(
        boxes=boxes,
        page_images_dir=Path(args.page_images_dir),
        overlay_dir=Path(args.overlay_dir),
    )

    total = sum(len(v) for v in boxes.values())
    print(f"OK: raw_pages={len(boxes)} raw_boxes={total}")
    print(f"Wrote: {args.overlay_dir}")


def _load_boxes_csv(path: Path) -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            page = _get_first(row, ["page", "page_name", "image", "file", "filename"])
            page_number = _get_first(row, ["page_number", "page_index", "page_idx"])

            if not page:
                if page_number:
                    page = f"page_{int(float(page_number)):04d}"
                else:
                    continue

            page = _normalize_page_name(page)

            try:
                bbox = (
                    int(float(_get_first(row, ["x1", "left", "xmin"]))),
                    int(float(_get_first(row, ["y1", "top", "ymin"]))),
                    int(float(_get_first(row, ["x2", "right", "xmax"]))),
                    int(float(_get_first(row, ["y2", "bottom", "ymax"]))),
                )
            except (TypeError, ValueError):
                continue

            box_id = _get_first(row, ["crop_id", "box_id", "id"]) or f"{page}_raw_{index:04d}"
            label = _get_first(row, ["label", "class", "type"]) or "unknown"
            score = _get_first(row, ["score", "layout_score", "confidence"]) or ""

            pages[page].append(
                {
                    "box_id": box_id,
                    "page": page,
                    "label": label,
                    "score": score,
                    "bbox": bbox,
                }
            )

    return pages


def render_raw_box_overlays(
    *,
    boxes: dict[str, list[dict]],
    page_images_dir: Path,
    overlay_dir: Path,
) -> None:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for page_name, page_boxes in boxes.items():
        image_path = _find_page_image(page_name, page_images_dir)

        if image_path is None:
            print(f"WARNING: no image found for page={page_name}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for i, box in enumerate(page_boxes, start=1):
            x1, y1, x2, y2 = box["bbox"]
            label = str(box["label"])
            score = str(box["score"])

            color = _color_for_label(label)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            text = f"{i}:{label}"
            if score:
                text += f":{score[:4]}"

            label_y = max(0, y1 - 13)
            draw.rectangle(
                [x1, label_y, x1 + max(70, len(text) * 7), label_y + 12],
                fill=color,
            )
            draw.text((x1 + 2, label_y + 1), text, fill=(255, 255, 255), font=font)

        output_path = overlay_dir / f"{page_name}_raw_boxes.png"
        image.save(output_path)


def _color_for_label(label: str) -> tuple[int, int, int]:
    if label in {"doc_title", "header", "paragraph_title", "figure_title"}:
        return (230, 120, 40)

    if label in {"text", "reference_content"}:
        return (40, 170, 80)

    if label in {"number", "footer"}:
        return (60, 120, 255)

    return (150, 80, 200)


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


def _get_first(row: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _normalize_page_name(value: str) -> str:
    value = Path(value).stem

    if value.startswith("page_"):
        return value

    digits = "".join(ch for ch in value if ch.isdigit())

    if digits:
        return f"page_{int(digits):04d}"

    return value


if __name__ == "__main__":
    main()