from __future__ import annotations

import argparse

from ocr_chunker.core.crop_regions import crop_regions_from_merged_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-regions", required=True)
    parser.add_argument("--page-images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--padding", type=int, default=10)

    args = parser.parse_args()

    crop_regions_from_merged_json(
        merged_regions_path=args.merged_regions,
        page_images_dir=args.page_images_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        padding_px=args.padding,
    )


if __name__ == "__main__":
    main()