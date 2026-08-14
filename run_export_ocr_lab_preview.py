from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunks", required=False)
    parser.add_argument("--html", required=True)

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    chunks: list[dict[str, Any]] = []
    if args.chunks and Path(args.chunks).exists():
        with Path(args.chunks).open("r", encoding="utf-8") as f:
            chunks = json.load(f)

    html_text = build_html(items, chunks)

    output_path = Path(args.html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"OK: items={len(items)} chunks={len(chunks)}")
    print(f"Wrote: {output_path}")


def build_html(items: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    pages = _group_items_by_page(items)

    total = len(items)
    easy_count = sum(1 for item in items if _has_easyocr(item))
    selected_easy = sum(1 for item in items if _selected_engine(item) == "easyocr")
    article_candidates = sum(
        1
        for item in items
        if ((item.get("article_signals") or {}).get("is_article_heading_candidate"))
    )

    parts: list[str] = []

    parts.append(
        """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>OCR Lab Preview</title>
<style>
body {
  font-family: Arial, Tahoma, sans-serif;
  background: #f5f6f8;
  color: #111827;
  margin: 24px;
}
h1, h2, h3 {
  margin-top: 0;
}
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}
.stat {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 10px 14px;
}
.tabs {
  display: flex;
  gap: 8px;
  margin: 18px 0;
  direction: ltr;
}
button {
  border: 1px solid #d1d5db;
  background: white;
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
}
button.active {
  background: #111827;
  color: white;
}
.section {
  display: none;
}
.section.active {
  display: block;
}
.page {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 24px;
}
.box-card {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 16px;
  direction: ltr;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
  margin: 12px 0;
  background: #fff;
}
.box-card.review {
  border-color: #dc2626;
  background: #fff7f7;
}
.crop img {
  max-width: 320px;
  max-height: 240px;
  border: 1px solid #ddd;
}
.meta {
  direction: ltr;
  text-align: left;
  color: #4b5563;
  font-size: 13px;
  margin-bottom: 8px;
}
.badge {
  display: inline-block;
  margin: 2px;
  padding: 3px 7px;
  border-radius: 999px;
  background: #e5e7eb;
}
.badge.good { background: #dcfce7; color: #166534; }
.badge.warn { background: #fef3c7; color: #92400e; }
.badge.bad { background: #fee2e2; color: #991b1b; }
.text-grid {
  direction: rtl;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
.text-block {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
  background: #fafafa;
}
.text-block h4 {
  margin: 0 0 8px 0;
  color: #374151;
}
.ocr-text {
  white-space: pre-wrap;
  line-height: 1.8;
  font-size: 16px;
}
.selected {
  background: #f0fdf4;
  border-color: #86efac;
}
.chunk {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 16px;
}
.chunk-text {
  white-space: pre-wrap;
  line-height: 1.9;
  font-size: 18px;
}
.ref {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 8px;
  margin: 6px 0;
}
</style>
</head>
<body>
<h1>OCR Lab Preview</h1>
"""
    )

    parts.append('<div class="summary">')
    parts.append(f'<div class="stat">Raw boxes: <strong>{total}</strong></div>')
    parts.append(f'<div class="stat">EasyOCR results: <strong>{easy_count}</strong></div>')
    parts.append(f'<div class="stat">Selected EasyOCR: <strong>{selected_easy}</strong></div>')
    parts.append(f'<div class="stat">Article heading candidates: <strong>{article_candidates}</strong></div>')
    parts.append(f'<div class="stat">Chunks: <strong>{len(chunks)}</strong></div>')
    parts.append("</div>")

    parts.append(
        """
<div class="tabs">
  <button class="active" onclick="showTab('raw', this)">Raw OCR boxes</button>
  <button onclick="showTab('chunks', this)">Article chunks</button>
</div>
"""
    )

    parts.append('<section id="raw" class="section active">')
    for page, page_items in pages.items():
        parts.append(f'<div class="page"><h2>{html.escape(page)}</h2>')

        for item in page_items:
            parts.append(_render_box_card(item))

        parts.append("</div>")
    parts.append("</section>")

    parts.append('<section id="chunks" class="section">')
    for chunk in chunks:
        parts.append(_render_chunk(chunk))
    parts.append("</section>")

    parts.append(
        """
<script>
function showTab(id, btn) {
  document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('button').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}
</script>
</body>
</html>
"""
    )

    return "\n".join(parts)


def _render_box_card(item: dict[str, Any]) -> str:
    region_id = str(item.get("region_id") or "")
    crop_path = str(item.get("crop_path") or "")
    crop_uri = _file_uri(crop_path)

    selected = (item.get("ocr") or {}).get("selected") or {}
    paddle = ((item.get("ocr") or {}).get("paddle") or {}).get("parsed") or {}
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}

    selected_text = str(selected.get("text") or "")
    paddle_text = str(paddle.get("text") or "")
    easy_text = str(easy.get("text") or "")

    signals = item.get("article_signals") or {}
    is_heading = bool(signals.get("is_article_heading_candidate"))
    refs = signals.get("inline_article_references") or []

    review = bool(item.get("needs_review")) or is_heading or bool(refs)

    cls = "box-card review" if review else "box-card"

    parts: list[str] = []
    parts.append(f'<div class="{cls}">')

    parts.append('<div class="crop">')
    if crop_uri:
        parts.append(f'<img src="{html.escape(crop_uri)}" />')
    else:
        parts.append("<p>No crop image</p>")
    parts.append("</div>")

    parts.append("<div>")
    parts.append('<div class="meta">')
    parts.append(f'<span class="badge">{html.escape(region_id)}</span>')
    parts.append(f'<span class="badge">page={html.escape(str(item.get("page")))}</span>')
    parts.append(f'<span class="badge">label={html.escape(str(item.get("label")))}</span>')
    parts.append(f'<span class="badge">selected={html.escape(str(selected.get("engine")))}</span>')
    parts.append(f'<span class="badge">conf={html.escape(str(selected.get("confidence")))}</span>')

    if is_heading:
        parts.append(
            f'<span class="badge good">ARTICLE HEADING {html.escape(str(signals.get("article_number")))}</span>'
        )

    if refs:
        parts.append(f'<span class="badge warn">REFS={len(refs)}</span>')

    if selected.get("empty"):
        parts.append('<span class="badge bad">EMPTY</span>')

    parts.append("</div>")

    parts.append('<div class="text-grid">')
    parts.append(_text_block("Paddle", paddle_text, paddle.get("confidence")))
    parts.append(_text_block("EasyOCR", easy_text, easy.get("confidence")))
    parts.append(_text_block("Selected", selected_text, selected.get("confidence"), selected=True))
    parts.append("</div>")

    if refs:
        parts.append('<div class="meta" style="margin-top:10px">References: ')
        parts.append(html.escape(str(refs)))
        parts.append("</div>")

    parts.append("</div>")
    parts.append("</div>")

    return "\n".join(parts)


def _render_chunk(chunk: dict[str, Any]) -> str:
    parts: list[str] = []

    parts.append('<div class="chunk">')
    parts.append(
        f"<h2>{html.escape(str(chunk.get('chunk_id')))}"
        f" — المادة {html.escape(str(chunk.get('article_number')))}</h2>"
    )

    parts.append('<div class="meta">')
    parts.append(f'<span class="badge">page {html.escape(str(chunk.get("page_start")))} → {html.escape(str(chunk.get("page_end")))}</span>')
    parts.append(f'<span class="badge">boxes={len(chunk.get("box_ids") or [])}</span>')

    if chunk.get("book"):
        parts.append(f'<span class="badge">book={html.escape(str(chunk.get("book")))}</span>')
    if chunk.get("bab"):
        parts.append(f'<span class="badge">bab={html.escape(str(chunk.get("bab")))}</span>')
    if chunk.get("fasl"):
        parts.append(f'<span class="badge">fasl={html.escape(str(chunk.get("fasl")))}</span>')

    if chunk.get("needs_review"):
        parts.append('<span class="badge bad">REVIEW</span>')

    parts.append("</div>")

    refs = chunk.get("references") or []
    if refs:
        parts.append("<h3>المراجع داخل المادة</h3>")
        for ref in refs:
            parts.append('<div class="ref">')
            parts.append(
                f"المادة المشار إليها: <strong>{html.escape(str(ref.get('article_number')))}</strong> "
                f"→ {html.escape(str(ref.get('target_chunk_id')))}"
            )
            if ref.get("target_text"):
                parts.append(
                    f'<div class="ocr-text">{html.escape(str(ref.get("target_text")))}</div>'
                )
            parts.append("</div>")

    parts.append(f'<div class="chunk-text">{html.escape(str(chunk.get("text") or ""))}</div>')
    parts.append("</div>")

    return "\n".join(parts)


def _text_block(title: str, text: str, confidence: Any, selected: bool = False) -> str:
    cls = "text-block selected" if selected else "text-block"

    return (
        f'<div class="{cls}">'
        f"<h4>{html.escape(title)} — conf={html.escape(str(confidence))}</h4>"
        f'<div class="ocr-text">{html.escape(text or "[EMPTY]")}</div>'
        f"</div>"
    )


def _group_items_by_page(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        page = str(item.get("page") or "unknown_page")
        pages.setdefault(page, []).append(item)

    for page_items in pages.values():
        page_items.sort(key=_item_sort_key)

    return dict(sorted(pages.items(), key=lambda kv: _page_sort_key(kv[0])))


def _item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (
        int(item.get("reading_order_start") or 0),
        str(item.get("region_id") or ""),
    )


def _page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)


def _has_easyocr(item: dict[str, Any]) -> bool:
    return bool(((item.get("ocr") or {}).get("easyocr") or {}).get("ok"))


def _selected_engine(item: dict[str, Any]) -> str:
    return str(((item.get("ocr") or {}).get("selected") or {}).get("engine") or "")


def _file_uri(path: str) -> str:
    if not path:
        return ""

    p = Path(path)

    try:
        return p.resolve().as_uri()
    except Exception:
        return ""
    

if __name__ == "__main__":
    main()