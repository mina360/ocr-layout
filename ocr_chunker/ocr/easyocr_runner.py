from __future__ import annotations

from pathlib import Path
from typing import Any


def run_easyocr_on_candidates(
    *,
    manifest_items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    languages: list[str] | None = None,
    gpu: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run EasyOCR only on selected candidate crops.

    EasyOCR is loaded inside the function so the rest of the pipeline
    can work even if EasyOCR is not installed.
    """
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is not installed. Run: pip install easyocr"
        ) from exc

    languages = languages or ["ar", "en"]

    reader = easyocr.Reader(languages, gpu=gpu)

    candidate_map = {
        str(candidate.get("region_id")): candidate
        for candidate in candidates
        if candidate.get("region_id")
    }

    updated: list[dict[str, Any]] = []

    attempted = 0

    for item in manifest_items:
        item = dict(item)
        region_id = str(item.get("region_id") or "")

        if region_id not in candidate_map:
            updated.append(item)
            continue

        if limit is not None and attempted >= limit:
            updated.append(item)
            continue

        attempted += 1

        crop_path = Path(str(item.get("crop_path") or ""))

        item.setdefault("ocr", {})

        if not crop_path.exists():
            item["ocr"]["easyocr"] = {
                "ok": False,
                "engine": "easyocr",
                "error": f"crop_not_found: {crop_path}",
                "parsed": {
                    "text": "",
                    "confidence": None,
                    "lines": [],
                },
            }
            updated.append(item)
            continue

        try:
            raw_result = reader.readtext(str(crop_path), detail=1, paragraph=False)
            parsed = _parse_easyocr_result(raw_result)

            item["ocr"]["easyocr"] = {
                "ok": True,
                "engine": "easyocr",
                "image_path": str(crop_path),
                "candidate_reasons": candidate_map[region_id].get("reasons", []),
                "raw": _json_safe_easyocr(raw_result),
                "parsed": parsed,
                "error": None,
            }

        except Exception as exc:
            item["ocr"]["easyocr"] = {
                "ok": False,
                "engine": "easyocr",
                "image_path": str(crop_path),
                "candidate_reasons": candidate_map[region_id].get("reasons", []),
                "raw": None,
                "parsed": {
                    "text": "",
                    "confidence": None,
                    "lines": [],
                },
                "error": str(exc),
            }

        updated.append(item)

    print(f"easyocr_attempted={attempted}")
    return updated


def _parse_easyocr_result(raw_result: list[Any]) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []

    for row in raw_result:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue

        bbox = row[0]
        text = str(row[1] or "").strip()

        try:
            confidence = float(row[2])
        except (TypeError, ValueError):
            confidence = None

        if not text:
            continue

        lines.append(
            {
                "text": " ".join(text.split()),
                "confidence": confidence,
                "bbox": _json_safe_bbox(bbox),
            }
        )

    lines.sort(key=_easyocr_reading_key)

    text = "\n".join(line["text"] for line in lines)

    scores = [
        line["confidence"]
        for line in lines
        if isinstance(line.get("confidence"), (int, float))
    ]

    confidence = round(sum(scores) / len(scores), 4) if scores else None

    return {
        "text": text,
        "confidence": confidence,
        "lines": lines,
        "empty": not bool(text.strip()),
    }


def _easyocr_reading_key(line: dict[str, Any]) -> tuple[int, int]:
    bbox = line.get("bbox") or []
    xs: list[int] = []
    ys: list[int] = []

    for point in bbox:
        if isinstance(point, list) and len(point) >= 2:
            xs.append(int(point[0]))
            ys.append(int(point[1]))

    if not xs or not ys:
        return (0, 0)

    y_center = int(sum(ys) / len(ys))
    x_right = max(xs)

    # Arabic order: top-to-bottom, right-to-left.
    return (round(y_center / 12), -x_right)


def _json_safe_easyocr(raw_result: list[Any]) -> list[Any]:
    output: list[Any] = []

    for row in raw_result:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue

        output.append(
            {
                "bbox": _json_safe_bbox(row[0]),
                "text": str(row[1]),
                "confidence": float(row[2]) if row[2] is not None else None,
            }
        )

    return output


def _json_safe_bbox(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []

    points: list[list[float]] = []

    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue

    return points