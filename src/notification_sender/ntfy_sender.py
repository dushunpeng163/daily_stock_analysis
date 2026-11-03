# -*- coding: utf-8 -*-
"""ntfy notification sender with stock-boundary chunking and reverse-order delivery."""

from __future__ import annotations

import logging
import re
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


def _hard_split_by_lines(text: str, max_bytes: int = _MAX_NTFY_CHUNK_BYTES) -> List[str]:
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


def _split_into_named_cards(text: str, base_title: str) -> List[Tuple[str, str]]:
    """Split report by H2 sections (each stock/index) and extract clear card titles."""
    is_market_review = "大盘复盘" in text or ("一、" in text and "明日交易计划" in text)
    default_title = f"🏛️ A股大盘整体复盘 - {datetime.now().strftime('%m-%d')}" if is_market_review else base_title

    # Split at markdown H2 headings (`\n## `) so each stock gets its own card
    parts = re.split(r"(?=\n## )", text)
    if len(parts) <= 1:
        sub_chunks = _hard_split_by_lines(text)
        total = len(sub_chunks)
        return [
            (default_title if total == 1 else f"{default_title} ({i}/{total})", c)
            for i, c in enumerate(sub_chunks, 1)
        ]

    cards: List[Tuple[str, str]] = []
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        first_line = part_stripped.split("\n", 1)[0].lstrip("#").strip()
        if "结果摘要" in first_line or "决策仪表盘" in first_line:
            sec_title = f"📊 11只标的汇总总览"
        elif first_line:
            sec_title = f"📈 {first_line}"
        else:
            sec_title = default_title

        sub_chunks = _hard_split_by_lines(part_stripped)
        if len(sub_chunks) == 1:
            cards.append((sec_title, sub_chunks[0]))
        else:
            for idx, sc in enumerate(sub_chunks, 1):
                cards.append((f"{sec_title} ({idx}/{len(sub_chunks)})", sc))

    return cards


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
        """Publish a notification to ntfy in reverse order so item #1 appears at the top."""
        if not self._is_ntfy_configured():
            logger.warning("ntfy URL 未配置，跳过推送")
            return False

        server_url, topic = self._resolve_ntfy_endpoint()
        if not server_url or not topic:
            logger.error("NTFY_URL 必须是包含 topic path 的完整 endpoint，例如 https://ntfy.sh/my-topic")
            return False

        if title is None:
            date_str = datetime.now().strftime("%m-%d")
            title = f"📈 股票分析报告 - {date_str}"
        sanitized_content = strip_hidden_markdown_metadata(content).strip()
        cards = _split_into_named_cards(sanitized_content, title)
        total = len(cards)

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "daily_stock_analysis",
        }
        token = (self._ntfy_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        all_ok = True
        # Reverse send order so that in ntfy UI (newest on top), card 1 appears at the very top
        for send_idx, (card_title, card_body) in enumerate(reversed(cards), start=1):
            payload = {
                "topic": topic,
                "title": card_title,
                "message": card_body,
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
                    logger.info("ntfy 消息发送成功 (%d/%d): %s", send_idx, total, card_title)
                else:
                    logger.error("ntfy 请求失败 (%d/%d): HTTP %s", send_idx, total, response.status_code)
                    all_ok = False
                if send_idx < total:
                    time.sleep(0.8)
            except Exception as exc:
                logger.error("发送 ntfy 消息失败 (%d/%d): %s", send_idx, total, exc)
                all_ok = False

        return all_ok
