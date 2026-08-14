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

    args = parser.parse_args()

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        items = json.load(f)

    with Path(args.chunks).open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    text = build_html(items, chunks)

    output_path = Path(args.html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    print(f"OK: items={len(items)} chunks={len(chunks)}")
    print(f"Wrote: {output_path}")


def build_html(items: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    pages = group_by_page(items)

    role_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for item in items:
        decision = box_decision(item)
        role = str(decision.get("role") or "missing")
        engine = str(decision.get("selected_engine") or "missing")

        role_counts[role] = role_counts.get(role, 0) + 1
        selected_counts[engine] = selected_counts.get(engine, 0) + 1

    parts: list[str] = []

    parts.append(
        """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<title>OCR Lab Preview v2</title>
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
.box-card.title { background: #f0f9ff; border-color: #7dd3fc; }
.box-card.article { background: #f0fdf4; border-color: #86efac; }
.box-card.skip { background: #fafafa; color: #6b7280; }
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
<h1>OCR Lab Preview v2</h1>
"""
    )

    parts.append('<div class="summary">')
    parts.append(f'<div class="stat">Boxes: <strong>{len(items)}</strong></div>')
    parts.append(f'<div class="stat">Chunks: <strong>{len(chunks)}</strong></div>')
    parts.append(f'<div class="stat">Roles: <strong>{html.escape(str(role_counts))}</strong></div>')
    parts.append(f'<div class="stat">Selected engines: <strong>{html.escape(str(selected_counts))}</strong></div>')
    parts.append("</div>")

    parts.append(
        """
<div class="tabs">
  <button class="active" onclick="showTab('raw', this)">Raw boxes + LLM decisions</button>
  <button onclick="showTab('chunks', this)">Article chunks + metadata</button>
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
        parts.append(render_chunk(chunk))
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
    decision = box_decision(item)
    role = str(decision.get("role") or "unknown")

    cls = "box-card"
    if role in {"book_title", "bab_title", "fasl_title", "topic_title", "document_title"}:
        cls += " title"
    elif role in {"article_heading", "article_heading_with_body"}:
        cls += " article"
    elif role in {"noise", "page_number", "container_text"}:
        cls += " skip"

    crop_path = str(item.get("crop_path") or "")
    crop_uri = file_uri(crop_path)

    paddle = (item.get("ocr") or {}).get("selected") or {}
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}

    parts: list[str] = []

    parts.append(f'<div class="{cls}">')

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
    parts.append(f'<span class="badge good">role={html.escape(role)}</span>')
    parts.append(f'<span class="badge">selected={html.escape(str(decision.get("selected_engine")))}</span>')
    parts.append(f'<span class="badge">article={html.escape(str(decision.get("article_number")))}</span>')
    parts.append(f'<span class="badge">use={html.escape(str(decision.get("should_use_in_chunks")))}</span>')
    parts.append("</div>")

    parts.append('<div class="text-grid">')
    parts.append(text_block("LLM selected", str(decision.get("selected_text") or ""), decision.get("selection_confidence"), selected=True))
    parts.append(text_block("Paddle clean", str(paddle.get("text") or ""), paddle.get("confidence")))
    parts.append(text_block("EasyOCR", str(easy.get("text") or ""), easy.get("confidence")))
    parts.append("</div>")

    parts.append("<details><summary>LLM decision JSON</summary>")
    parts.append(f"<pre>{html.escape(json.dumps(decision, ensure_ascii=False, indent=2))}</pre>")
    parts.append("</details>")

    parts.append("</div>")
    parts.append("</div>")

    return "\n".join(parts)


def render_chunk(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}

    parts: list[str] = []
    parts.append('<div class="chunk">')

    parts.append(
        f"<h2>{html.escape(str(chunk.get('chunk_id')))} — المادة {html.escape(str(chunk.get('article_number')))}</h2>"
    )

    parts.append('<div class="meta">')
    parts.append(f'<span class="badge">page {html.escape(str(chunk.get("page_start")))} → {html.escape(str(chunk.get("page_end")))}</span>')
    parts.append(f'<span class="badge">boxes={html.escape(str(metadata.get("box_count")))}</span>')
    parts.append(f'<span class="badge">metadata_boxes={html.escape(str(metadata.get("metadata_box_count")))}</span>')
    parts.append(f'<span class="badge">skipped_duplicates={html.escape(str(metadata.get("skipped_duplicate_box_count")))}</span>')
    if chunk.get("needs_review"):
        parts.append('<span class="badge bad">REVIEW</span>')
    parts.append("</div>")

    parts.append('<div class="metadata-grid">')
    keys = [
        "document_title",
        "book_title",
        "bab_title",
        "fasl_title",
        "topic_title",
        "article_number_raw",
        "article_number_final",
        "article_number_resolution",
        "heading_box_id",
        "heading_role_confidence",
        "heading_selection_confidence",
        "heading_selected_engine",
        "reference_count",
        "source_pages",
    ]

    for key in keys:
        parts.append(
            f'<div class="metadata-item"><strong>{html.escape(key)}</strong>: {html.escape(str(metadata.get(key)))}</div>'
        )

    parts.append("</div>")

    if chunk.get("review_reasons"):
        parts.append('<div class="metadata-item">')
        parts.append("<strong>review_reasons</strong>: ")
        parts.append(html.escape(" | ".join(chunk.get("review_reasons") or [])))
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

    parts.append(f'<div class="chunk-text">{html.escape(str(chunk.get("text") or ""))}</div>')

    parts.append("<details><summary>Full chunk JSON</summary>")
    parts.append(f"<pre>{html.escape(json.dumps(chunk, ensure_ascii=False, indent=2))}</pre>")
    parts.append("</details>")

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


def group_by_page(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        page = str(item.get("page") or "unknown_page")
        pages.setdefault(page, []).append(item)

    for page_items in pages.values():
        page_items.sort(key=lambda item: (int(item.get("reading_order_start") or 0), str(item.get("region_id") or "")))

    return dict(sorted(pages.items(), key=lambda kv: page_sort_key(kv[0])))


def page_sort_key(page: str) -> tuple[int, str]:
    digits = "".join(ch for ch in page if ch.isdigit())
    return (int(digits) if digits else 0, page)


def box_decision(item: dict[str, Any]) -> dict[str, Any]:
    return ((item.get("llm") or {}).get("box_judge") or {})


def file_uri(path: str) -> str:
    if not path:
        return ""

    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return ""


if __name__ == "__main__":
    main()