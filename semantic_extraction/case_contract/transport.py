"""Synchronous, loopback-only HTTP transport for the existing vLLM service."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class Reply:
    status: int
    body: bytes


class TransportError(RuntimeError):
    pass


def validate_base_url(base_url: str) -> str:
    url = urllib.parse.urlsplit(base_url)
    if (url.scheme != "http" or url.hostname != "127.0.0.1"
            or url.username or url.password or url.query or url.fragment
            or url.path.rstrip("/") != "/v1" or not url.port
            or not 1024 <= url.port <= 65535):
        raise ValueError("base_url must be http://127.0.0.1:<port>/v1")
    return base_url.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError("redirect_refused")


class VllmTransport:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 300):
        self.base_url = validate_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, endpoint: str, payload: dict | None = None) -> Reply:
        if endpoint not in {"models", "chat/completions"}:
            raise ValueError("unsupported endpoint")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/" + endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None,
            headers=headers, method="POST" if payload else "GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise TransportError("response_size_limit")
                return Reply(response.status, body)
        except urllib.error.HTTPError as exc:
            # Preserve bounded private error text; never print it to shared logs.
            return Reply(exc.code, exc.read(8 * 1024 * 1024))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(type(exc).__name__) from None


def strict_json(text: str | bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite_json_constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
