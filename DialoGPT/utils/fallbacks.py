"""
Fallbacks provider for DialoGPT.

Loads optional fallbacks.json and provides convenient selection methods.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional


class DialoGPTFallbacks:
    """Loads fallback responses and selects appropriate replies.

    Priority order when selecting:
    1) smart_fallbacks (list choices per keyword)
    2) simple_fallbacks (single string per keyword)
    3) fallback_responses (single string per keyword)
    4) default_responses (random choice)
    5) hard-coded ultimate default
    """

    def __init__(self, json_path: Optional[str] = None) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # default file next to repository root: <repo>/fallbacks.json
        default_path = os.path.join(base_dir, "..", "fallbacks.json")
        default_path = os.path.abspath(default_path)

        self.json_path = json_path or default_path
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def _pick(self, items: List[str]) -> str:
        try:
            return random.choice(items)
        except Exception:
            return "That's interesting! Could you tell me more about that?"

    def get_response(self, prompt: str) -> str:
        prompt_lower = (prompt or "").lower()

        smart: Dict[str, List[str]] = self.data.get("smart_fallbacks", {}) or {}
        simple: Dict[str, str] = self.data.get("simple_fallbacks", {}) or {}
        legacy: Dict[str, str] = self.data.get("fallback_responses", {}) or {}
        defaults: List[str] = self.data.get("default_responses", []) or []

        # 1) smart fallbacks
        for key, choices in smart.items():
            if key in prompt_lower and isinstance(choices, list) and choices:
                return self._pick(choices)

        # 2) simple fallbacks
        for key, text in simple.items():
            if key in prompt_lower and isinstance(text, str) and text:
                return text

        # 3) legacy fallback_responses
        for key, text in legacy.items():
            if key in prompt_lower and isinstance(text, str) and text:
                return text

        # 4) defaults list
        if defaults:
            return self._pick(defaults)

        # 5) ultimate default
        return "That's interesting! Could you tell me more about that?"
