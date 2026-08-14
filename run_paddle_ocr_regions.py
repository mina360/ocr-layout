from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ocr_chunker.ocr.paddle_client import PaddleOcrClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.getenv("PADDLE_OCR_BASE_URL"))
    parser.add_argument("--token", default=os.getenv("PADDLE_OCR_TOKEN"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-review", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    if not args.base_url:
        raise ValueError("Missing PaddleOCR base URL. Set PADDLE_OCR_BASE_URL or pass --base-url.")

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        crops = json.load(f)

    client = PaddleOcrClient(
        base_url=args.base_url,
        token=args.token,
        timeout_seconds=args.timeout,
    )

    print("Checking PaddleOCR health...")
    health = client.health()
    print(f"PaddleOCR health OK: {health}")

    attempted = 0
    processed = 0
    failed = 0

    for index, item in enumerate(crops, start=1):
        if args.limit is not None and attempted >= args.limit:
            break

        if args.only_review and not item.get("needs_review", False):
            continue

        if args.skip_existing and _has_existing_paddle_result(item):
            continue

        attempted += 1

        crop_path = item.get("crop_path")
        region_id = item.get("region_id")
        page = _resolve_page_number(item)

        item.setdefault("ocr", {})

        if not crop_path:
            item["ocr"]["paddle"] = {
                "ok": False,
                "error": "missing_crop_path",
            }
            failed += 1
            continue

        print(f"[attempt {attempted}] [{index}/{len(crops)}] OCR {region_id} -> {crop_path}")

        result = client.infer_image(
            image_path=crop_path,
            page=str(page),
        )

        item["ocr"]["paddle"] = result

        if result.get("ok"):
            parsed = result.get("parsed") or {}
            item["ocr"]["selected"] = {
                "engine": "paddleocr_colab",
                "text": parsed.get("text") or "",
                "confidence": parsed.get("confidence"),
                "status": "auto_selected_initial",
            }
            processed += 1
        else:
            print(f"FAILED {region_id}: {result.get('error')}")
            failed += 1

        _write_json(output_path, crops)

    _write_json(output_path, crops)

    print(f"OK: attempted={attempted} processed={processed} failed={failed}")
    print(f"Wrote: {output_path}")


def _has_existing_paddle_result(item: dict[str, Any]) -> bool:
    ocr = item.get("ocr") or {}
    paddle = ocr.get("paddle")

    return bool(isinstance(paddle, dict) and paddle.get("ok"))


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _resolve_page_number(item: dict[str, Any]) -> str:
    page_number = item.get("page_number")

    if page_number is not None:
        try:
            return str(int(page_number))
        except (TypeError, ValueError):
            pass

    page = str(item.get("page") or "")
    digits = "".join(ch for ch in page if ch.isdigit())

    if digits:
        return str(int(digits))

    return "0"

if __name__ == "__main__":
    main()