from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--html", required=True)
    parser.add_argument("--all-text", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    html_path = Path(args.html)
    all_text_path = Path(args.all_text)
    csv_path = Path(args.csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    all_text_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    pages = _group_by_page(items)

    _write_page_text_files(pages, output_dir)
    _write_all_text(pages, all_text_path)
    _write_csv(pages, csv_path)
    _write_html_preview(pages, html_path)

    total_regions = sum(len(regions) for regions in pages.values())
    empty_regions = sum(
        1
        for regions in pages.values()
        for item in regions
        if not _selected_text(item).strip()
    )

    print(f"OK: pages={len(pages)} regions={total_regions} empty_regions={empty_regions}")
    print(f"Wrote page text dir: {output_dir}")
    print(f"Wrote all text: {all_text_path}")
    print(f"Wrote HTML preview: {html_path}")
    print(f"Wrote CSV: {csv_path}")


def _group_by_page(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        page = str(item.get("page") or "unknown_page")
        pages.setdefault(page, []).append(item)

    for page, regions in pages.items():
        regions.sort(
            key=lambda item: (
                int(item.get("reading_order_start") or 0),
                int(item.get("reading_order_end") or 0),
                str(item.get("region_id") or ""),
            )
        )

    return dict(sorted(pages.items(), key=lambda kv: _page_sort_key(kv[0])))


def _write_page_text_files(
    pages: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    for page, regions in pages.items():
        path = output_dir / f"{page}.txt"

        with path.open("w", encoding="utf-8") as f:
            f.write(f"{page}\n")
            f.write("=" * 80)
            f.write("\n\n")

            for item in regions:
                region_id = item.get("region_id")
                confidence = _selected_confidence(item)
                text = _selected_text(item)

                f.write(f"[{region_id}] confidence={confidence}\n")
                f.write(text.strip() or "[EMPTY OCR TEXT]")
                f.write("\n\n")


def _write_all_text(
    pages: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        for page, regions in pages.items():
            f.write("\n\n")
            f.write("#" * 90)
            f.write(f"\n# {page}\n")
            f.write("#" * 90)
            f.write("\n\n")

            for item in regions:
                region_id = item.get("region_id")
                text = _selected_text(item).strip()

                f.write(f"## {region_id}\n")
                f.write(text or "[EMPTY OCR TEXT]")
                f.write("\n\n")


def _write_csv(
    pages: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page",
                "region_id",
                "reading_order_start",
                "reading_order_end",
                "confidence",
                "needs_review",
                "review_reasons",
                "crop_path",
                "text_preview",
            ],
        )
        writer.writeheader()

        for page, regions in pages.items():
            for item in regions:
                text = _selected_text(item).replace("\n", " / ")
                writer.writerow(
                    {
                        "page": page,
                        "region_id": item.get("region_id"),
                        "reading_order_start": item.get("reading_order_start"),
                        "reading_order_end": item.get("reading_order_end"),
                        "confidence": _selected_confidence(item),
                        "needs_review": item.get("needs_review"),
                        "review_reasons": "|".join(item.get("review_reasons") or []),
                        "crop_path": item.get("crop_path"),
                        "text_preview": text[:250],
                    }
                )


def _write_html_preview(
    pages: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    parts: list[str] = []

    parts.append(
        """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>OCR Preview</title>
<style>
body {
  font-family: Arial, "Tahoma", sans-serif;
  background: #f6f7f9;
  margin: 24px;
  color: #111827;
}
.page {
  background: white;
  border: 1px solid #ddd;
  border-radius: 12px;
  margin-bottom: 28px;
  padding: 18px;
}
.page h2 {
  margin-top: 0;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 10px;
}
.region {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 16px;
  direction: ltr;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 12px;
  margin: 12px 0;
  background: #ffffff;
}
.region.review {
  border-color: #dc2626;
  background: #fff7f7;
}
.crop img {
  max-width: 340px;
  max-height: 260px;
  border: 1px solid #ddd;
  background: #fff;
}
.text {
  direction: rtl;
  text-align: right;
  white-space: pre-wrap;
  line-height: 1.9;
  font-size: 18px;
}
.meta {
  direction: ltr;
  text-align: left;
  font-size: 13px;
  color: #4b5563;
  margin-bottom: 10px;
}
.badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 999px;
  background: #e5e7eb;
  margin-inline-end: 6px;
}
.badge.review {
  background: #fee2e2;
  color: #991b1b;
}
.empty {
  color: #991b1b;
  font-weight: bold;
}
</style>
</head>
<body>
<h1>OCR Preview</h1>
"""
    )

    for page, regions in pages.items():
        parts.append(f'<section class="page">')
        parts.append(f"<h2>{html.escape(page)}</h2>")

        for item in regions:
            region_id = str(item.get("region_id") or "")
            crop_path = str(item.get("crop_path") or "")
            crop_uri = Path(crop_path).resolve().as_uri() if crop_path else ""
            text = _selected_text(item).strip()
            confidence = _selected_confidence(item)
            needs_review = bool(item.get("needs_review"))
            review_reasons = item.get("review_reasons") or []

            region_class = "region review" if needs_review else "region"

            parts.append(f'<div class="{region_class}">')

            parts.append('<div class="crop">')
            if crop_uri:
                parts.append(f'<img src="{html.escape(crop_uri)}" />')
            else:
                parts.append("<p>No crop image</p>")
            parts.append("</div>")

            parts.append("<div>")
            parts.append('<div class="meta">')
            parts.append(f'<span class="badge">{html.escape(region_id)}</span>')
            parts.append(
                f'<span class="badge">order {item.get("reading_order_start")} → {item.get("reading_order_end")}</span>'
            )
            parts.append(f'<span class="badge">confidence={html.escape(str(confidence))}</span>')

            if needs_review:
                parts.append('<span class="badge review">REVIEW</span>')

            if review_reasons:
                parts.append(
                    f'<span class="badge review">{html.escape(" | ".join(review_reasons))}</span>'
                )

            parts.append("</div>")

            if text:
                parts.append(f'<div class="text">{html.escape(text)}</div>')
            else:
                parts.append('<div class="text empty">[EMPTY OCR TEXT]</div>')

            parts.append("</div>")
            parts.append("</div>")

        parts.append("</section>")

    parts.append("</body></html>")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _selected_text(item: dict[str, Any]) -> str:
    ocr = item.get("ocr") or {}

    selected = ocr.get("selected") or {}
    text = selected.get("text")

    if isinstance(text, str) and text.strip():
        return text

    paddle = ocr.get("paddle") or {}
    parsed = paddle.get("parsed") or {}
    text = parsed.get("text")

    if isinstance(text, str):
        return text

    return ""


def _selected_confidence(item: dict[str, Any]) -> Any:
    ocr = item.get("ocr") or {}

    selected = ocr.get("selected") or {}
    if selected.get("confidence") is not None:
        return selected.get("confidence")

    paddle = ocr.get("paddle") or {}
    parsed = paddle.get("parsed") or {}
    return parsed.get("confidence")


def _page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)


if __name__ == "__main__":
    main()