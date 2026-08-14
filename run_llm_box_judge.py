from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ocr_chunker.llm.box_judge import judge_single_box
from ocr_chunker.llm.ollama_client import OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    candidate_ids = load_candidate_ids(args.candidates)

    client = OllamaClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout,
    )

    updated: list[dict[str, Any]] = []
    attempted = 0

    for item in items:
        item = dict(item)
        item.setdefault("llm", {})

        region_id = str(item.get("region_id") or "")

        should_call_llm = candidate_ids is None or region_id in candidate_ids

        if not should_call_llm:
            item["llm"]["box_judge"] = default_non_candidate_decision(item)
            updated.append(item)
            continue

        if args.limit is not None and attempted >= args.limit:
            item["llm"]["box_judge"] = default_not_judged_due_limit(item)
            updated.append(item)
            continue

        attempted += 1
        print(f"[{attempted}] LLM judge {region_id}")

        try:
            decision = judge_single_box(item=item, client=client)
            item["llm"]["box_judge"] = decision

            selected_text = str(decision.get("selected_text") or "").strip()
            selected_engine = str(decision.get("selected_engine") or "none")

            item.setdefault("ocr", {})
            item["ocr"]["selected"] = {
                "engine": selected_engine,
                "text": selected_text,
                "confidence": decision.get("selection_confidence"),
                "status": "llm_box_judge_selected",
                "selection_reason": decision.get("selection_reason"),
                "empty": not bool(selected_text),
            }

        except Exception as exc:
            item["llm"]["box_judge"] = fallback_decision(item, error=str(exc))

        updated.append(item)

        # Save after every LLM call so Ctrl+C does not lose all progress.
        write_json(output_path, updated + items[len(updated):])

    write_json(output_path, updated)

    role_counts: dict[str, int] = {}
    engine_counts: dict[str, int] = {}

    for item in updated:
        decision = ((item.get("llm") or {}).get("box_judge") or {})
        role = decision.get("role") or "missing"
        engine = decision.get("selected_engine") or "missing"

        role_counts[role] = role_counts.get(role, 0) + 1
        engine_counts[engine] = engine_counts.get(engine, 0) + 1

    print(f"llm_attempted={attempted}")
    print(f"OK: items={len(updated)}")
    print(f"Role counts: {role_counts}")
    print(f"Engine counts: {engine_counts}")
    print(f"Wrote: {output_path}")


def load_candidate_ids(path: str | None) -> set[str] | None:
    if not path:
        return None

    with Path(path).open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    ids: set[str] = set()

    for item in candidates:
        region_id = item.get("region_id") or item.get("box_id")
        if region_id:
            ids.add(str(region_id))

    return ids


def default_non_candidate_decision(item: dict[str, Any]) -> dict[str, Any]:
    text = selected_text(item)

    return {
        "box_id": item.get("region_id") or item.get("box_id"),
        "selected_engine": selected_engine(item),
        "selected_text": text,
        "selection_confidence": selected_confidence(item),
        "selection_reason": "not_llm_candidate_default_body_text",
        "role": "body_text",
        "role_confidence": None,
        "article_number": None,
        "referenced_article_numbers": [],
        "applies_to": "self",
        "should_use_in_chunks": bool(text.strip()),
        "needs_review": False,
        "review_reasons": [],
        "normalized_title": None,
    }


def default_not_judged_due_limit(item: dict[str, Any]) -> dict[str, Any]:
    text = selected_text(item)

    return {
        "box_id": item.get("region_id") or item.get("box_id"),
        "selected_engine": selected_engine(item),
        "selected_text": text,
        "selection_confidence": selected_confidence(item),
        "selection_reason": "not_judged_due_limit",
        "role": "unknown",
        "role_confidence": None,
        "article_number": None,
        "referenced_article_numbers": [],
        "applies_to": "self",
        "should_use_in_chunks": bool(text.strip()),
        "needs_review": True,
        "review_reasons": ["not_judged_due_limit"],
        "normalized_title": None,
    }


def fallback_decision(item: dict[str, Any], *, error: str) -> dict[str, Any]:
    text = selected_text(item)

    return {
        "box_id": item.get("region_id") or item.get("box_id"),
        "selected_engine": selected_engine(item),
        "selected_text": text,
        "selection_confidence": selected_confidence(item),
        "selection_reason": f"fallback_after_llm_error: {error[:200]}",
        "role": "unknown",
        "role_confidence": None,
        "article_number": None,
        "referenced_article_numbers": [],
        "applies_to": "self",
        "should_use_in_chunks": bool(text.strip()),
        "needs_review": True,
        "review_reasons": ["llm_judge_failed"],
        "normalized_title": None,
    }


def selected_text(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    text = selected.get("text")
    return text if isinstance(text, str) else ""


def selected_confidence(item: dict[str, Any]) -> float | None:
    selected = (item.get("ocr") or {}).get("selected") or {}
    value = selected.get("confidence")

    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def selected_engine(item: dict[str, Any]) -> str:
    selected = (item.get("ocr") or {}).get("selected") or {}
    engine = selected.get("engine")
    return str(engine or "paddleocr_colab_rec_texts")


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()