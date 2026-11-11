"""Validated OpenAI-compatible JSON calls with resumable, content-addressed caching."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Protocol

from .io import canonical_json, fingerprint, read_json, write_json
from .prompts import SYSTEM_PROMPTS


class JsonBackend(Protocol):
    def complete(self, task: str, payload: dict, validate: Callable | None = None) -> dict: ...


class StructuredOutputError(RuntimeError):
    pass


def parse_json_output(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("\x60\x60\x60"):
        text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text).strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result


class OpenAIJsonLLM:
    def __init__(self, config: dict):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ.get(config["api_key_env"], "EMPTY"),
            timeout=config["timeout"],
            max_retries=0,
        )
        self.cache = Path(config["cache_dir"])

    def complete(self, task: str, payload: dict, validate: Callable | None = None) -> dict:
        system = SYSTEM_PROMPTS[task]
        options = {
            key: self.config[key]
            for key in ("model", "temperature", "max_tokens", "json_mode", "disable_thinking")
        }
        cache_key = fingerprint(
            {
                "system": system,
                "payload": payload,
                "options": options,
                "endpoint": self.config["base_url"],
                "schema": 1,
            }
        )
        cache_file = self.cache / cache_key[:2] / (cache_key + ".json")
        if cache_file.exists():
            result = read_json(cache_file)
            if validate:
                validate(result)
            return result
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical_json(payload)},
        ]
        last_error = None
        for attempt in range(self.config["retries"] + 1):
            try:
                kwargs = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": self.config["temperature"],
                    "max_tokens": self.config["max_tokens"],
                }
                if self.config["json_mode"]:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.config["disable_thinking"]:
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                response = self.client.chat.completions.create(**kwargs)
                if response.choices[0].finish_reason == "length":
                    raise ValueError("LLM response hit max_tokens; increase llm.max_tokens")
                result = parse_json_output(response.choices[0].message.content or "")
                if validate:
                    validate(result)
                write_json(cache_file, result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.config["retries"]:
                    if isinstance(exc, (ValueError, KeyError, TypeError)):
                        messages = messages[:2] + [
                            {
                                "role": "user",
                                "content": f"Your response failed schema validation: {exc}. Return corrected JSON.",
                            }
                        ]
                    time.sleep(min(2**attempt, 5))
        raise StructuredOutputError(f"{task} failed after retries: {last_error}") from last_error
