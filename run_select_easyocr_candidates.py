from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ocr_chunker.ocr.easyocr_candidate_selector import select_easyocr_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--short-text-max-chars", type=int, default=30)
    parser.add_argument("--small-box-area-ratio", type=float, default=1.25)

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    candidates = select_easyocr_candidates(
        items,
        low_confidence_threshold=args.low_confidence_threshold,
        short_text_max_chars=args.short_text_max_chars,
        small_box_area_ratio=args.small_box_area_ratio,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_csv(candidates, Path(args.csv))

    print(f"OK: items={len(items)} easyocr_candidates={len(candidates)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {args.csv}")


def _write_csv(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "label",
                "paddle_confidence",
                "reasons",
                "crop_path",
                "paddle_text",
            ],
        )
        writer.writeheader()

        for item in candidates:
            writer.writerow(
                {
                    "page": item.get("page"),
                    "region_id": item.get("region_id"),
                    "label": item.get("label"),
                    "paddle_confidence": item.get("paddle_confidence"),
                    "reasons": "|".join(item.get("reasons") or []),
                    "crop_path": item.get("crop_path"),
                    "paddle_text": str(item.get("paddle_text") or "").replace("\n", " / ")[:300],
                }
            )


if __name__ == "__main__":
    main()