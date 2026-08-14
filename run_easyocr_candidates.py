from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_chunker.ocr.easyocr_runner import run_easyocr_on_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--languages", default="ar,en")

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        manifest_items = json.load(f)

    with Path(args.candidates).open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    languages = [
        lang.strip()
        for lang in args.languages.split(",")
        if lang.strip()
    ]

    updated = run_easyocr_on_candidates(
        manifest_items=manifest_items,
        candidates=candidates,
        languages=languages,
        gpu=args.gpu,
        limit=args.limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    easy_ok = sum(
        1
        for item in updated
        if ((item.get("ocr") or {}).get("easyocr") or {}).get("ok")
    )

    print(f"OK: manifest_items={len(updated)} easyocr_ok={easy_ok}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()