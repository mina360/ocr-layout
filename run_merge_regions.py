from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from ocr_chunker.core.merge_regions import merge_ordered_boxes_into_regions
from ocr_chunker.core.page_metadata import mark_page_number_candidates
from ocr_chunker.core.reading_order import assign_arabic_reading_order
from ocr_chunker.debug.render_overlay import render_overlays_from_json
from ocr_chunker.io.loaders import load_boxes_csv, load_final_results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=False,
        help="Path to final_results.json. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--boxes-csv",
        required=False,
        help="Path to raw layout boxes.csv. Preferred source of truth.",
    )
    parser.add_argument("--output", required=True, help="Path to merged_regions.json")
    parser.add_argument("--debug-csv", required=True, help="Path to merged_regions_debug.csv")

    parser.add_argument(
        "--page-images-dir",
        required=False,
        help="Optional directory containing rendered page images.",
    )
    parser.add_argument(
        "--overlay-dir",
        required=False,
        help="Optional output directory for visual region overlays.",
    )

    args = parser.parse_args()

    if args.boxes_csv:
        boxes = load_boxes_csv(args.boxes_csv)
        source_name = "boxes_csv"
    elif args.input:
        boxes = load_final_results(args.input)
        source_name = "final_results_json"
    else:
        raise ValueError("Provide either --boxes-csv or --input")

    ordered = assign_arabic_reading_order(boxes)
    marked = mark_page_number_candidates(ordered)
    pages = merge_ordered_boxes_into_regions(marked)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(page) for page in pages], f, ensure_ascii=False, indent=2)

    _write_debug_csv(pages, args.debug_csv)

    if args.page_images_dir and args.overlay_dir:
        render_overlays_from_json(
            merged_regions_path=args.output,
            page_images_dir=args.page_images_dir,
            output_dir=args.overlay_dir,
        )

    total_regions = sum(len(page.regions) for page in pages)
    total_review = sum(
        1
        for page in pages
        for region in page.regions
        if region.needs_review
    )

    print(
        f"OK: source={source_name} "
        f"boxes={len(boxes)} pages={len(pages)} "
        f"regions={total_regions} review_regions={total_review}"
    )
    print(f"Wrote: {output_path}")


def _write_debug_csv(pages, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "page_number",
                "region_id",
                "region_role",
                "reading_order_start",
                "reading_order_end",
                "child_box_ids",
                "bbox",
                "quality_score",
                "needs_review",
                "review_reasons",
                "suggested_actions",
                "merge_reasons",
                "text_preview",
            ],
        )
        writer.writeheader()

        for page in pages:
            for region in page.regions:
                writer.writerow(
                    {
                        "page": region.page,
                        "page_number": region.page_number,
                        "region_id": region.region_id,
                        "region_role": region.region_role,
                        "reading_order_start": region.reading_order_start,
                        "reading_order_end": region.reading_order_end,
                        "child_box_ids": "|".join(region.child_box_ids),
                        "bbox": region.bbox,
                        "quality_score": region.quality_score,
                        "needs_review": region.needs_review,
                        "review_reasons": "|".join(region.review_reasons),
                        "suggested_actions": "|".join(region.suggested_actions),
                        "merge_reasons": "|".join(region.merge_reasons),
                        "text_preview": region.text[:160].replace("\n", " / "),
                    }
                )


if __name__ == "__main__":
    main()