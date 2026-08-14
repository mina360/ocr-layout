from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--html", required=True)
    parser.add_argument("--max-body-crops-per-chunk", type=int, default=12)

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    with Path(args.chunks).open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    html_text = build_html(
        items,
        chunks,
        max_body_crops_per_chunk=args.max_body_crops_per_chunk,
    )

    output_path = Path(args.html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"OK: items={len(items)} chunks={len(chunks)}")
    print(f"Wrote: {output_path}")


def build_html(
    items: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    max_body_crops_per_chunk: int,
) -> str:
    pages = group_items_by_page(items)
    item_by_id = build_item_index(items)

    parts: list[str] = []

    parts.append(
        """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>OCR Lab Improved</title>
<style>
body {
  font-family: Arial, Tahoma, sans-serif;
  background: #f5f6f8;
  color: #111827;
  margin: 24px;
}
.summary {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 18px;
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
  direction: ltr;
  margin: 18px 0;
}
button {
  border: 1px solid #d1d5db;
  background: white;
  border-radius: 8px;
  padding: 8px 12px;
  cursor: pointer;
}
button.active {
  background: #111827;
  color: white;
}
.section { display: none; }
.section.active { display: block; }
.page, .chunk {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 22px;
}
.box-card {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 16px;
  direction: ltr;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
  margin: 12px 0;
}
.crop img {
  max-width: 340px;
  max-height: 250px;
  border: 1px solid #d1d5db;
}
.badge {
  display: inline-block;
  direction: ltr;
  background: #e5e7eb;
  border-radius: 999px;
  padding: 3px 7px;
  margin: 2px;
  font-size: 13px;
}
.badge.good { background: #dcfce7; color: #166534; }
.badge.warn { background: #fef3c7; color: #92400e; }
.badge.bad { background: #fee2e2; color: #991b1b; }
.meta {
  direction: ltr;
  text-align: left;
  font-size: 13px;
  color: #4b5563;
  margin-bottom: 8px;
}
.text-grid {
  direction: rtl;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
.text-block {
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 10px;
}
.text-block.selected {
  background: #ecfdf5;
  border-color: #86efac;
}
.ocr-text, .chunk-text {
  white-space: pre-wrap;
  line-height: 1.85;
  font-size: 16px;
}
.chunk-text {
  font-size: 18px;
}
.chunk-layout {
  display: grid;
  grid-template-columns: 420px 1fr;
  gap: 18px;
  direction: ltr;
}
.chunk-main {
  direction: rtl;
}
.chunk-crops {
  direction: rtl;
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 10px;
  max-height: 900px;
  overflow: auto;
}
.crop-group {
  margin-bottom: 18px;
}
.crop-group h4 {
  margin: 8px 0;
}
.article-crop-card {
  border: 1px solid #e5e7eb;
  background: white;
  border-radius: 10px;
  padding: 8px;
  margin-bottom: 10px;
}
.article-crop-card.heading {
  border-color: #86efac;
  background: #f0fdf4;
}
.article-crop-card.context {
  border-color: #7dd3fc;
  background: #f0f9ff;
}
.article-crop-card.body {
  border-color: #e5e7eb;
}
.article-crop-card img {
  max-width: 370px;
  max-height: 220px;
  border: 1px solid #d1d5db;
  display: block;
  margin-bottom: 6px;
}
.crop-caption {
  direction: ltr;
  text-align: left;
  font-size: 12px;
  color: #4b5563;
}
.metadata-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  direction: ltr;
  margin: 10px 0;
}
.metadata-item {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 8px;
}
.context-block {
  background: #f0f9ff;
  border: 1px solid #7dd3fc;
  border-radius: 8px;
  padding: 8px;
  margin: 6px 0;
}
.ref {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 8px;
  margin: 6px 0;
}
pre {
  direction: ltr;
  text-align: left;
  white-space: pre-wrap;
  background: #111827;
  color: #f9fafb;
  border-radius: 8px;
  padding: 10px;
  overflow: auto;
  font-size: 12px;
}
</style>
</head>
<body>
<h1>OCR Lab Improved — No LLM</h1>
"""
    )

    parts.append('<div class="summary">')
    parts.append(f'<div class="stat">Boxes: <strong>{len(items)}</strong></div>')
    parts.append(f'<div class="stat">Chunks: <strong>{len(chunks)}</strong></div>')
    parts.append(f'<div class="stat">Articles: <strong>{sum(1 for c in chunks if c.get("chunk_type") == "article")}</strong></div>')
    parts.append(f'<div class="stat">Review chunks: <strong>{sum(1 for c in chunks if c.get("needs_review"))}</strong></div>')
    parts.append("</div>")

    parts.append(
        """
<div class="tabs">
  <button class="active" onclick="showTab('raw', this)">Raw boxes OCR</button>
  <button onclick="showTab('chunks', this)">Article chunks + crops + metadata</button>
</div>
"""
    )

    parts.append('<section id="raw" class="section active">')
    for page, page_items in pages.items():
        parts.append(f'<div class="page"><h2>{html.escape(page)}</h2>')

        for item in page_items:
            parts.append(render_box(item))

        parts.append("</div>")
    parts.append("</section>")

    parts.append('<section id="chunks" class="section">')
    for chunk in chunks:
        parts.append(
            render_chunk(
                chunk,
                item_by_id=item_by_id,
                max_body_crops_per_chunk=max_body_crops_per_chunk,
            )
        )
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


def render_box(item: dict[str, Any]) -> str:
    crop_path = str(item.get("crop_path") or "")
    crop_uri = file_uri(crop_path)

    selected = (item.get("ocr") or {}).get("selected") or {}
    debug = selected.get("debug") or {}
    paddle = debug.get("paddle_text_preview", "")
    easy = debug.get("easyocr_text_preview", "")

    selected_text = str(selected.get("text") or "")

    parts: list[str] = []

    parts.append('<div class="box-card">')

    parts.append('<div class="crop">')
    if crop_uri:
        parts.append(f'<img src="{html.escape(crop_uri)}" />')
    else:
        parts.append("<p>No crop</p>")
    parts.append("</div>")

    parts.append("<div>")

    parts.append('<div class="meta">')
    parts.append(f'<span class="badge">{html.escape(str(item.get("region_id")))}</span>')
    parts.append(f'<span class="badge">page={html.escape(str(item.get("page")))}</span>')
    parts.append(f'<span class="badge">layout={html.escape(str(item.get("label")))}</span>')
    parts.append(f'<span class="badge good">selected={html.escape(str(selected.get("engine")))}</span>')
    parts.append(f'<span class="badge">conf={html.escape(str(selected.get("confidence")))}</span>')
    parts.append(f'<span class="badge">reason={html.escape(str(selected.get("selection_reason")))}</span>')
    parts.append("</div>")

    parts.append('<div class="text-grid">')
    parts.append(text_block("Selected", selected_text, selected.get("confidence"), selected=True))
    parts.append(text_block("Paddle clean preview", str(paddle), None))
    parts.append(text_block("EasyOCR preview", str(easy), None))
    parts.append("</div>")

    parts.append("</div>")
    parts.append("</div>")

    return "\n".join(parts)


def render_chunk(
    chunk: dict[str, Any],
    *,
    item_by_id: dict[str, dict[str, Any]],
    max_body_crops_per_chunk: int,
) -> str:
    metadata = chunk.get("metadata") or {}

    parts: list[str] = []

    parts.append('<div class="chunk">')

    parts.append(
        f"<h2>{html.escape(str(chunk.get('chunk_id')))} — المادة {html.escape(str(chunk.get('article_number')))}</h2>"
    )

    parts.append('<div class="chunk-layout">')

    parts.append('<aside class="chunk-crops">')
    parts.append(render_chunk_crops(chunk, item_by_id, max_body_crops_per_chunk=max_body_crops_per_chunk))
    parts.append("</aside>")

    parts.append('<main class="chunk-main">')

    parts.append('<div class="meta">')
    parts.append(f'<span class="badge">page {html.escape(str(chunk.get("page_start")))} → {html.escape(str(chunk.get("page_end")))}</span>')
    parts.append(f'<span class="badge">boxes={html.escape(str(metadata.get("box_count")))}</span>')
    parts.append(f'<span class="badge">body_boxes={html.escape(str(metadata.get("body_box_count")))}</span>')
    parts.append(f'<span class="badge">metadata_boxes={html.escape(str(metadata.get("metadata_box_count")))}</span>')
    parts.append(f'<span class="badge">skipped_duplicates={html.escape(str(metadata.get("skipped_duplicate_box_count")))}</span>')

    if chunk.get("needs_review"):
        parts.append('<span class="badge bad">REVIEW</span>')

    parts.append("</div>")

    parts.append('<div class="metadata-grid">')
    keys = [
        "article_number_raw",
        "article_number_final",
        "article_number_resolution",
        "heading_box_id",
        "ocr_engine",
        "ocr_confidence",
        "selection_reason",
        "box_count",
        "body_box_count",
        "metadata_box_count",
        "skipped_duplicate_box_count",
        "reference_count",
        "source_pages",
    ]

    for key in keys:
        parts.append(
            f'<div class="metadata-item"><strong>{html.escape(key)}</strong>: {html.escape(str(metadata.get(key)))}</div>'
        )

    parts.append("</div>")

    context_blocks = metadata.get("context_blocks") or []

    if context_blocks:
        parts.append("<h3>Context / Metadata blocks</h3>")

        for block in context_blocks:
            parts.append('<div class="context-block">')
            parts.append(f'<strong>{html.escape(str(block.get("box_id")))}</strong> ')
            parts.append(f'[{html.escape(str(block.get("label")))}] ')
            parts.append(f'page={html.escape(str(block.get("page")))}')
            parts.append(f'<div class="ocr-text">{html.escape(str(block.get("text") or ""))}</div>')
            parts.append("</div>")

    refs = chunk.get("references") or []

    if refs:
        parts.append("<h3>References</h3>")

        for ref in refs:
            parts.append('<div class="ref">')
            parts.append(
                f"article {html.escape(str(ref.get('article_number')))} → {html.escape(str(ref.get('target_chunk_id')))}"
            )
            parts.append("</div>")

    if chunk.get("review_reasons"):
        parts.append('<div class="metadata-item">')
        parts.append("<strong>review_reasons</strong>: ")
        parts.append(html.escape(" | ".join(chunk.get("review_reasons") or [])))
        parts.append("</div>")

    parts.append(f'<div class="chunk-text">{html.escape(str(chunk.get("text") or ""))}</div>')

    parts.append("<details><summary>Full chunk JSON</summary>")
    parts.append(f"<pre>{html.escape(json.dumps(chunk, ensure_ascii=False, indent=2))}</pre>")
    parts.append("</details>")

    parts.append("</main>")
    parts.append("</div>")
    parts.append("</div>")

    return "\n".join(parts)


def render_chunk_crops(
    chunk: dict[str, Any],
    item_by_id: dict[str, dict[str, Any]],
    *,
    max_body_crops_per_chunk: int,
) -> str:
    parts: list[str] = []

    heading_id = chunk.get("heading_box_id")
    metadata_box_ids = chunk.get("metadata_box_ids") or []
    body_box_ids = chunk.get("body_box_ids") or []

    parts.append("<h3>Crops</h3>")

    if heading_id:
        parts.append('<div class="crop-group">')
        parts.append("<h4>Heading crop</h4>")
        parts.append(render_article_crop_card(heading_id, item_by_id, css_class="heading"))
        parts.append("</div>")

    if metadata_box_ids:
        parts.append('<div class="crop-group">')
        parts.append("<h4>Metadata / context crops</h4>")
        for box_id in metadata_box_ids:
            parts.append(render_article_crop_card(box_id, item_by_id, css_class="context"))
        parts.append("</div>")

    if body_box_ids:
        parts.append('<div class="crop-group">')
        parts.append(f"<h4>Body crops first {max_body_crops_per_chunk}</h4>")

        for box_id in body_box_ids[:max_body_crops_per_chunk]:
            parts.append(render_article_crop_card(box_id, item_by_id, css_class="body"))

        if len(body_box_ids) > max_body_crops_per_chunk:
            hidden_count = len(body_box_ids) - max_body_crops_per_chunk
            parts.append(
                f'<div class="metadata-item">Hidden body crops: {html.escape(str(hidden_count))}. Increase --max-body-crops-per-chunk to show more.</div>'
            )

        parts.append("</div>")

    if not heading_id and not metadata_box_ids and not body_box_ids:
        parts.append('<div class="metadata-item">No crop ids attached to this chunk.</div>')

    return "\n".join(parts)


def render_article_crop_card(
    box_id: Any,
    item_by_id: dict[str, dict[str, Any]],
    *,
    css_class: str,
) -> str:
    box_id_str = str(box_id)
    item = item_by_id.get(box_id_str)

    if item is None:
        return f'<div class="article-crop-card {css_class}"><div class="crop-caption">Missing crop for {html.escape(box_id_str)}</div></div>'

    crop_path = str(item.get("crop_path") or "")
    crop_uri = file_uri(crop_path)
    selected = (item.get("ocr") or {}).get("selected") or {}

    parts: list[str] = []

    parts.append(f'<div class="article-crop-card {css_class}">')

    if crop_uri:
        parts.append(f'<img src="{html.escape(crop_uri)}" />')
    else:
        parts.append('<div class="metadata-item">No image file</div>')

    parts.append('<div class="crop-caption">')
    parts.append(f'{html.escape(box_id_str)} | ')
    parts.append(f'label={html.escape(str(item.get("label")))} | ')
    parts.append(f'engine={html.escape(str(selected.get("engine")))} | ')
    parts.append(f'conf={html.escape(str(selected.get("confidence")))}')
    parts.append("</div>")

    text = str(selected.get("text") or "")
    if text:
        parts.append(f'<div class="ocr-text">{html.escape(text[:500])}</div>')

    parts.append("</div>")

    return "\n".join(parts)


def text_block(title: str, text: str, confidence: Any, *, selected: bool = False) -> str:
    cls = "text-block selected" if selected else "text-block"

    return (
        f'<div class="{cls}">'
        f'<h4>{html.escape(title)} — conf={html.escape(str(confidence))}</h4>'
        f'<div class="ocr-text">{html.escape(text or "[EMPTY]")}</div>'
        f"</div>"
    )


def build_item_index(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}

    for item in items:
        for key in ("region_id", "box_id"):
            value = item.get(key)
            if value:
                output[str(value)] = item

    return output


def group_items_by_page(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        page = str(item.get("page") or "unknown_page")
        pages.setdefault(page, []).append(item)

    for page_items in pages.values():
        page_items.sort(
            key=lambda item: (
                int(item.get("reading_order_start") or 0),
                str(item.get("region_id") or ""),
            )
        )

    return dict(sorted(pages.items(), key=lambda kv: page_sort_key(kv[0])))


def page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)


def file_uri(path: str) -> str:
    if not path:
        return ""

    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


if __name__ == "__main__":
    main()