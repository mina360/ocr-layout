from __future__ import annotations

import json
from typing import Any

from ocr_chunker.llm.ollama_client import OllamaClient


VALID_ROLES = {
    "document_title",
    "book_title",
    "bab_title",
    "fasl_title",
    "topic_title",
    "article_heading",
    "body_text",
    "article_heading_with_body",
    "preamble_text",
    "page_number",
    "signature_or_date",
    "container_text",
    "noise",
    "unknown",
}


def judge_boxes_with_llm(
    *,
    items: list[dict[str, Any]],
    client: OllamaClient,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    attempted = 0

    for index, item in enumerate(items, start=1):
        item = dict(item)
        item.setdefault("llm", {})

        if limit is not None and attempted >= limit:
            output.append(item)
            continue

        attempted += 1

        print(f"[{attempted}] LLM judge {item.get('region_id')}")

        try:
            decision = judge_single_box(item=item, client=client)
            item["llm"]["box_judge"] = decision

            item.setdefault("ocr", {})
            item["ocr"]["selected"] = {
                "engine": decision.get("selected_engine") or "llm_selected",
                "text": decision.get("selected_text") or "",
                "confidence": decision.get("selection_confidence"),
                "status": "llm_box_judge_selected",
                "selection_reason": decision.get("selection_reason"),
                "empty": not bool(str(decision.get("selected_text") or "").strip()),
            }

        except Exception as exc:
            item["llm"]["box_judge"] = fallback_decision(item, error=str(exc))

        output.append(item)

    print(f"llm_attempted={attempted}")
    return output


def judge_single_box(
    *,
    item: dict[str, Any],
    client: OllamaClient,
) -> dict[str, Any]:
    prompt = build_box_judge_prompt(item)

    data = client.generate_json(
        prompt=prompt,
        temperature=0.0,
    )

    decision = normalize_decision(data, item)

    return decision


def build_box_judge_prompt(item: dict[str, Any]) -> str:
    paddle_text = get_paddle_clean_text(item)
    paddle_conf = get_paddle_clean_confidence(item)

    easy_text = get_easyocr_text(item)
    easy_conf = get_easyocr_confidence(item)

    payload = {
        "box": {
            "box_id": item.get("region_id") or item.get("box_id"),
            "page": item.get("page"),
            "page_number": item.get("page_number"),
            "reading_order": item.get("reading_order_start"),
            "layout_label": item.get("label"),
            "layout_score": item.get("layout_score"),
            "bbox": item.get("bbox"),
            "padded_bbox": item.get("padded_bbox"),
            "note": (
                "The source documents are legal text documents with no real images. "
                "If layout_label is image, treat it as a text area that may contain headings or body text."
            ),
        },
        "ocr_candidates": {
            "paddle": {
                "text": paddle_text,
                "confidence": paddle_conf,
            },
            "easyocr": {
                "text": easy_text,
                "confidence": easy_conf,
            },
        },
    }

    return f"""
You are a strict OCR and legal-document layout judge for Arabic scanned legal texts.

Task:
Given one OCR crop with layout metadata and two OCR outputs, choose the better text and classify the crop role.

Important rules:
- Do not rely on OCR confidence alone. Different OCR engines have incompatible confidence scores.
- Prefer the text that better matches the visible/legal-document role.
- If layout_label is "image", do NOT assume it is a real image. The document has no real pictures, so it may be a title area or a large text container.
- If the crop contains multiple independent sections/articles merged together, classify it as "container_text".
- If the crop is a heading that applies to following articles, classify it as one of:
  document_title, book_title, bab_title, fasl_title, topic_title.
- If the crop is the start of an article heading, classify it as article_heading.
- If the crop contains the article heading and its body together, classify it as article_heading_with_body.
- If it is ordinary article body text, classify it as body_text.
- If it is page number only, classify it as page_number.
- If it is corrupted OCR/no useful text, classify it as noise.
- Return JSON only.
If you produce a cleaned short heading/title that is better than both OCR strings, use selected_engine="llm_normalized" and put that cleaned title in selected_text. Do not use "none" unless there is no useful text.

Allowed role values:
document_title
book_title
bab_title
fasl_title
topic_title
article_heading
body_text
article_heading_with_body
preamble_text
page_number
signature_or_date
container_text
noise
unknown

Required JSON schema:
{{
  "selected_engine": "paddle|easyocr|llm_normalized|none",
  "selected_text": "string",
  "selection_confidence": 0.0,
  "selection_reason": "short reason",
  "role": "one allowed role",
  "role_confidence": 0.0,
  "article_number": null,
  "referenced_article_numbers": [],
  "applies_to": "self|following_articles|previous_article|none",
  "should_use_in_chunks": true,
  "needs_review": false,
  "review_reasons": [],
  "normalized_title": null
}}

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def normalize_decision(data: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    selected_engine = str(data.get("selected_engine") or "").strip().lower()

    if selected_engine not in {"paddle", "easyocr", "llm_normalized", "none"}:
        selected_engine = "none"

    role = str(data.get("role") or "unknown").strip()

    if role not in VALID_ROLES:
        role = "unknown"

    selected_text = data.get("selected_text")
    if not isinstance(selected_text, str):
        selected_text = ""

    normalized_title = data.get("normalized_title")
    if normalized_title is not None and not isinstance(normalized_title, str):
        normalized_title = None

    # Important repair:
    # Sometimes the LLM correctly classifies a title and provides normalized_title,
    # but returns selected_engine=none and selected_text="".
    # In that case we preserve the normalized title as selected text.
    if not selected_text.strip() and normalized_title:
        selected_text = normalized_title
        selected_engine = "llm_normalized"

    # If it still selected none but there is usable OCR text, keep a safe fallback.
    if selected_engine == "none" and not selected_text.strip():
        paddle_text = get_paddle_clean_text(item).strip()
        easy_text = get_easyocr_text(item).strip()

        if easy_text:
            selected_text = easy_text
            selected_engine = "easyocr"
        elif paddle_text:
            selected_text = paddle_text
            selected_engine = "paddle"

    article_number = safe_int(data.get("article_number"))

    references = data.get("referenced_article_numbers")
    if not isinstance(references, list):
        references = []

    references = [
        value
        for value in (safe_int(v) for v in references)
        if value is not None
    ]

    applies_to = str(data.get("applies_to") or "self").strip()
    if applies_to not in {"self", "following_articles", "previous_article", "none"}:
        applies_to = "self"

    should_use = data.get("should_use_in_chunks")
    if not isinstance(should_use, bool):
        should_use = role not in {"noise", "page_number", "container_text"}

    needs_review = data.get("needs_review")
    if not isinstance(needs_review, bool):
        needs_review = False

    review_reasons = data.get("review_reasons")
    if not isinstance(review_reasons, list):
        review_reasons = []

    if selected_engine == "llm_normalized":
        review_reasons.append("llm_normalized_text_used")
        needs_review = True

    return {
        "box_id": item.get("region_id") or item.get("box_id"),
        "selected_engine": selected_engine,
        "selected_text": selected_text.strip(),
        "selection_confidence": safe_float(data.get("selection_confidence")),
        "selection_reason": str(data.get("selection_reason") or "").strip(),
        "role": role,
        "role_confidence": safe_float(data.get("role_confidence")),
        "article_number": article_number,
        "referenced_article_numbers": references,
        "applies_to": applies_to,
        "should_use_in_chunks": should_use,
        "needs_review": needs_review,
        "review_reasons": list(dict.fromkeys(str(v) for v in review_reasons)),
        "normalized_title": normalized_title,
    }

def fallback_decision(item: dict[str, Any], *, error: str) -> dict[str, Any]:
    paddle_text = get_paddle_clean_text(item)
    easy_text = get_easyocr_text(item)

    selected_text = easy_text.strip() or paddle_text.strip()
    selected_engine = "easyocr" if easy_text.strip() else "paddle"

    return {
        "box_id": item.get("region_id") or item.get("box_id"),
        "selected_engine": selected_engine if selected_text else "none",
        "selected_text": selected_text,
        "selection_confidence": None,
        "selection_reason": f"fallback_after_llm_error: {error[:200]}",
        "role": "unknown",
        "role_confidence": None,
        "article_number": None,
        "referenced_article_numbers": [],
        "applies_to": "self",
        "should_use_in_chunks": bool(selected_text),
        "needs_review": True,
        "review_reasons": ["llm_judge_failed"],
        "normalized_title": None,
    }


def get_paddle_clean_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def get_paddle_clean_confidence(item: dict[str, Any]) -> float | None:
    selected = (item.get("ocr") or {}).get("selected") or {}
    return safe_float(selected.get("confidence"))


def get_easyocr_text(item: dict[str, Any]) -> str:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    text = easy.get("text")
    return text if isinstance(text, str) else ""


def get_easyocr_confidence(item: dict[str, Any]) -> float | None:
    easy = ((item.get("ocr") or {}).get("easyocr") or {}).get("parsed") or {}
    return safe_float(easy.get("confidence"))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None