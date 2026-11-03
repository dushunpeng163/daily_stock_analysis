# -*- coding: utf-8 -*-
"""ntfy notification sender with automatic chunking to prevent HTTP 413."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse, urlunparse

import requests

from src.config import Config
from src.formatters import strip_hidden_markdown_metadata


logger = logging.getLogger(__name__)

_MAX_NTFY_CHUNK_BYTES = 3400


def resolve_ntfy_endpoint(ntfy_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split NTFY_URL into server root and topic from the final path segment."""
    raw_url = (ntfy_url or "").strip().rstrip("/")
    if not raw_url:
        return None, None

    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None, None

    path_segments = [segment for segment in parsed.path.split("/") if segment]
    if not path_segments:
        return None, None

    topic = unquote(path_segments[-1]).strip()
    if not topic:
        return None, None

    root_path = "/".join(path_segments[:-1])
    server_url = urlunparse(
        parsed._replace(
            path=f"/{root_path}" if root_path else "",
            params="",
            query="",
            fragment="",
        )
    ).rstrip("/")

    return server_url, topic


def _split_markdown_chunks(text: str, max_bytes: int = _MAX_NTFY_CHUNK_BYTES) -> List[str]:
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    chunks: List[str] = []
    current_lines: List[str] = []
    current_bytes = 0

    for line in text.split("\n"):
        line_bytes = len((line + "\n").encode("utf-8"))
        if line_bytes > max_bytes:
            if current_lines:
                chunks.append("\n".join(current_lines).strip())
                current_lines = []
                current_bytes = 0
            # Hard-split overly long single line by characters
            buf = ""
            for ch in line:
                if len((buf + ch).encode("utf-8")) > max_bytes:
                    chunks.append(buf)
                    buf = ch
                else:
                    buf += ch
            if buf:
                chunks.append(buf)
            continue

        if current_bytes + line_bytes > max_bytes and current_lines:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [line]
            current_bytes = line_bytes
        else:
            current_lines.append(line)
            current_bytes += line_bytes

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return [c for c in chunks if c]


class NtfySender:
    """Send Markdown text notifications through the ntfy JSON publish API."""

    def __init__(self, config: Config):
        self._ntfy_url = getattr(config, "ntfy_url", None)
        self._ntfy_token = getattr(config, "ntfy_token", None)
        self._webhook_verify_ssl = getattr(config, "webhook_verify_ssl", True)

    def _is_ntfy_configured(self) -> bool:
        return bool(self._ntfy_url)

    def _resolve_ntfy_endpoint(self) -> Tuple[Optional[str], Optional[str]]:
        return resolve_ntfy_endpoint(self._ntfy_url)

    def send_to_ntfy(
        self,
        content: str,
        title: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """Publish a notification to ntfy using a JSON body with UTF-8 text (auto-chunked)."""
        if not self._is_ntfy_configured():
            logger.warning("ntfy URL 未配置，跳过推送")
            return False

        server_url, topic = self._resolve_ntfy_endpoint()
        if not server_url or not topic:
            logger.error("NTFY_URL 必须是包含 topic path 的完整 endpoint，例如 https://ntfy.sh/my-topic")
            return False

        if title is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            title = f"📈 股票分析报告 - {date_str}"
        sanitized_content = strip_hidden_markdown_metadata(content).strip()
        chunks = _split_markdown_chunks(sanitized_content)
        total = len(chunks)

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "daily_stock_analysis",
        }
        token = (self._ntfy_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        all_ok = True
        for idx, chunk in enumerate(chunks, start=1):
            chunk_title = title if total == 1 else f"{title} ({idx}/{total})"
            payload = {
                "topic": topic,
                "title": chunk_title,
                "message": chunk,
                "markdown": True,
            }
            try:
                response = requests.post(
                    server_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_seconds or 10,
                    verify=self._webhook_verify_ssl,
                )
                if 200 <= response.status_code < 300:
                    logger.info("ntfy 消息发送成功 (%d/%d)", idx, total)
                else:
                    logger.error("ntfy 请求失败 (%d/%d): HTTP %s", idx, total, response.status_code)
                    all_ok = False
                if idx < total:
                    time.sleep(1.0)
            except Exception as exc:
                logger.error("发送 ntfy 消息失败 (%d/%d): %s", idx, total, exc)
                all_ok = False

        return all_ok
