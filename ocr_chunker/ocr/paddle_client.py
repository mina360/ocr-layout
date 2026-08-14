from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


class PaddleOcrClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout_seconds: int = 120,
        max_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def health(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/health",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def infer_image(
        self,
        *,
        image_path: str | Path,
        page: str,
    ) -> dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Crop image not found: {image_path}")

        url = f"{self.base_url}/infer"

        # FastAPI /infer implementations often differ:
        # image: UploadFile, file: UploadFile, or upload: UploadFile.
        request_variants = [
            {"file_field": "file", "data": {"page": page}},
            {"file_field": "image", "data": {"page": page}},
            {"file_field": "upload", "data": {"page": page}},
            {"file_field": "image_file", "data": {"page": page}},
            {"file_field": "file", "data": {}},
            {"file_field": "image", "data": {}},
        ]

        errors: list[str] = []

        for variant in request_variants:
            file_field = variant["file_field"]
            data = variant["data"]

            for attempt in range(self.max_retries + 1):
                try:
                    with image_path.open("rb") as f:
                        files = {
                            file_field: (
                                image_path.name,
                                f,
                                "image/png",
                            )
                        }

                        response = requests.post(
                            url,
                            headers=self._headers(),
                            files=files,
                            data=data,
                            timeout=self.timeout_seconds,
                        )

                    if response.status_code == 422:
                        errors.append(
                            f"422 with file_field={file_field}, data_keys={list(data.keys())}: "
                            f"{response.text[:800]}"
                        )
                        break

                    if response.status_code >= 400:
                        errors.append(
                            f"{response.status_code} with file_field={file_field}, "
                            f"data_keys={list(data.keys())}: {response.text[:800]}"
                        )

                        if attempt < self.max_retries:
                            time.sleep(1.5 * (attempt + 1))
                            continue

                        break

                    raw = response.json()

                    return {
                        "ok": True,
                        "engine": "paddleocr_colab",
                        "image_path": str(image_path),
                        "request_variant": {
                            "file_field": file_field,
                            "data_keys": list(data.keys()),
                        },
                        "raw": raw,
                        "parsed": parse_paddle_response(raw),
                        "error": None,
                    }

                except Exception as exc:
                    errors.append(
                        f"exception with file_field={file_field}, data_keys={list(data.keys())}: {exc}"
                    )

                    if attempt < self.max_retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue

                    break

        return {
            "ok": False,
            "engine": "paddleocr_colab",
            "image_path": str(image_path),
            "raw": None,
            "parsed": {
                "text": "",
                "confidence": None,
                "lines": [],
            },
            "error": "\n".join(errors[-8:]),
        }

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers


def parse_paddle_response(raw: Any) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    _collect_text_lines(raw, lines)

    text_parts = [
        str(line["text"]).strip()
        for line in lines
        if str(line.get("text") or "").strip()
    ]

    confidences = [
        float(line["confidence"])
        for line in lines
        if isinstance(line.get("confidence"), (int, float))
    ]

    confidence = None
    if confidences:
        confidence = round(sum(confidences) / len(confidences), 4)

    return {
        "text": "\n".join(text_parts),
        "confidence": confidence,
        "lines": lines,
    }


def _collect_text_lines(value: Any, output: list[dict[str, Any]]) -> None:
    if value is None:
        return

    if isinstance(value, dict):
        for text_key in ("text", "transcription", "rec_text", "content"):
            if text_key in value and isinstance(value[text_key], str):
                output.append(
                    {
                        "text": value[text_key],
                        "confidence": _extract_confidence(value),
                        "source": f"dict.{text_key}",
                    }
                )
                return

        for key in (
            "result",
            "results",
            "ocr",
            "data",
            "res",
            "rec_texts",
            "rec_scores",
            "dt_polys",
        ):
            if key in value:
                _collect_text_lines(value[key], output)

        # PaddleOCR 3.x sometimes returns:
        # {"rec_texts": [...], "rec_scores": [...]}
        rec_texts = value.get("rec_texts")
        rec_scores = value.get("rec_scores")

        if isinstance(rec_texts, list):
            for index, text in enumerate(rec_texts):
                if not isinstance(text, str):
                    continue

                confidence = None
                if isinstance(rec_scores, list) and index < len(rec_scores):
                    score = rec_scores[index]
                    if isinstance(score, (int, float)):
                        confidence = float(score)

                output.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "source": "rec_texts",
                    }
                )

        for child in value.values():
            _collect_text_lines(child, output)

        return

    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], (int, float)):
            output.append(
                {
                    "text": value[0],
                    "confidence": float(value[1]),
                    "source": "list_text_score",
                }
            )
            return

        if len(value) >= 2 and isinstance(value[1], (list, tuple)):
            second = value[1]
            if len(second) >= 2 and isinstance(second[0], str):
                confidence = second[1] if isinstance(second[1], (int, float)) else None
                output.append(
                    {
                        "text": second[0],
                        "confidence": confidence,
                        "source": "paddle_bbox_text_score",
                    }
                )
                return

        for item in value:
            _collect_text_lines(item, output)

        return

    if isinstance(value, str):
        stripped = value.strip()

        if stripped:
            try:
                decoded = json.loads(stripped)
                _collect_text_lines(decoded, output)
            except Exception:
                output.append(
                    {
                        "text": stripped,
                        "confidence": None,
                        "source": "plain_string",
                    }
                )


def _extract_confidence(value: dict[str, Any]) -> float | None:
    for key in ("confidence", "score", "rec_score", "final_score"):
        score = value.get(key)
        if isinstance(score, (int, float)):
            return float(score)

    return None