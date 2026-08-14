from __future__ import annotations

import json
from typing import Any

import requests


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate_json(
        self,
        *,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        raw_text = data.get("response") or ""

        return parse_json_object(raw_text)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value

    raise ValueError(f"Could not parse JSON object from LLM response: {text[:500]}")