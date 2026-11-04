#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股发现：全市场量化初筛（DSA 选股引擎）→ 排除已在自选的 → ATR 风控点位 → 推送到 ntfy。

环境变量：
  SCREEN_STRATEGIES  逗号分隔的策略 id，默认 momentum_quality,quality_value,balanced_alpha
  SCREEN_TOP_N       每套策略取前 N（默认 10）
  SCREEN_MAX_CARDS   最终推送的候选数上限（默认 20）
  STOCK_LIST         已在自选里的代码，会被排除
  NTFY_URL / NTFY_TOKEN
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCREENING_ENABLED", "true")

import requests  # noqa: E402

from src.config import Config  # noqa: E402
from src.notification_sender.ntfy_sender import resolve_ntfy_endpoint  # noqa: E402
from src.risk_levels import compute_risk_plan, render_risk_plan_markdown  # noqa: E402
from src.services.screening.pipeline import screen  # noqa: E402
from src.stock_analyzer import StockTrendAnalyzer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("discover")

LABELS = {
    "momentum_quality": "趋势质量",
    "quality_value": "稳健价值",
    "balanced_alpha": "均衡多因子",
    "shrink_pullback": "缩量回踩",
    "low_volatility_quality": "低波质量",
    "oversold_reversal": "超跌反转",
    "blue_chip_income": "蓝筹收益",
    "dual_low": "双低选股",
    "capital_heat": "资金热度",
}
VERDICT_ORDER = {"ok": 0, "wait": 1, "weak": 2, "skip": 3}
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://dushunpeng163.github.io/TrendRadar/").rstrip("/") + "/"


def digits(code: str) -> str:
    return re.sub(r"\D", "", str(code or ""))[-6:]


def clip(text: str, n: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def run_strategies(strategies, top_n, extra, cfg):
    hits = {}
    for sid in strategies:
        res = None
        for use_llm in (True, False):
            try:
                res = screen(sid, market="cn", max_output=top_n + extra, use_llm=use_llm, daily_enrich=True, config=cfg)
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("策略 %s (use_llm=%s) 失败: %s", sid, use_llm, exc)
        if not res:
            continue
        log.info("策略 %s：快照 %s → 过滤后 %s → 入选 %s（%s）", sid, res.snapshot_count, res.after_filter_count, len(res.picks), res.ranking_mode)
        yield_rank = 0
        for p in res.picks:
            hits.setdefault(p.code, {"pick": p, "hits": []})
            hits[p.code]["hits"].append((sid, p.rank))
            if p.final_score > hits[p.code]["pick"].final_score:
                hits[p.code]["pick"] = p
            yield_rank += 1
    return hits


def build_plan(code, pick):
    try:
        from data_provider import DataFetcherManager

        df, _ = DataFetcherManager().get_daily_data(code, days=90)
        if df is None or df.empty:
            return None
        tr = StockTrendAnalyzer().analyze(df, code)
        price = float(pick.price or 0) or float(tr.current_price or 0)
        return compute_risk_plan(price, tr.ma5, tr.ma10, tr.atr14, tr.high_20, tr.high_60, tr.ma20)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s ATR 计划失败: %s", code, exc)
        return None


def mv_text(mv):
    try:
        mv = float(mv)
    except (TypeError, ValueError):
        return "—"
    return f"{mv / 1e8:.0f}" if mv > 1e7 else f"{mv:.0f}"


def card_body(item) -> str:
    p, plan = item["pick"], item["plan"]
    names = " · ".join(LABELS.get(s, s) for s, _ in item["hits"])
    lines = [
        f"## 🔎 {p.name} ({p.code})",
        "",
        f"**命中策略**: {names}",
        f"**综合评分**: {p.final_score:.1f}",
        f"**行业**: {p.industry or '—'}",
        f"**现价**: {p.price:.2f} | **涨跌**: {p.change_pct:+.2f}% | **PE**: {p.pe_ratio if p.pe_ratio is not None else '—'} | **PB**: {p.pb_ratio if p.pb_ratio is not None else '—'} | **市值亿**: {mv_text(p.total_mv)}",
        "",
        f"**入选理由**: {clip(p.ranking_reason)}",
        f"**风险提示**: {clip(p.risk_summary) or '—'}",
        "",
    ]
    if plan:
        lines += render_risk_plan_markdown(plan)
    return "\n".join(lines)


def post_all(cards, server_url, topic, token):
    headers = {"Content-Type": "application/json; charset=utf-8", "User-Agent": "daily_stock_analysis"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = int(time.time()) - 2
    payloads = []
    for title, body, link in cards:
        pl = {"topic": topic, "title": title, "message": body, "markdown": True,
              "click": link, "actions": [{"action": "view", "label": "打开看板", "url": link}]}
        payloads.append(pl)
    for i, pl in enumerate(reversed(payloads), 1):  # 倒序发送，ntfy 最新在上
        r = requests.post(server_url, json=pl, headers=headers, timeout=15)
        log.info("ntfy %d/%d %s -> %s", i, len(payloads), pl["title"], r.status_code)
        time.sleep(1.2)
    # 回查并补发
    poll = f"{server_url.rstrip('/')}/{topic}/json?poll=1&since={started}"
    need = Counter(p["title"] for p in payloads)
    for attempt in range(3):
        time.sleep(4)
        stored = Counter()
        for line in requests.get(poll, headers={k: v for k, v in headers.items() if k == "Authorization"}, timeout=20).text.splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("event") == "message":
                stored[m.get("title", "")] += 1
        missing = Counter({t: n - stored.get(t, 0) for t, n in need.items() if n - stored.get(t, 0) > 0})
        if not missing:
            log.info("送达校验通过：%d 条", len(payloads))
            return
        log.warning("缺失 %d 条，补发: %s", sum(missing.values()), list(missing)[:5])
        for pl in reversed(payloads):
            if missing.get(pl["title"], 0) > 0:
                missing[pl["title"]] -= 1
                requests.post(server_url, json=pl, headers=headers, timeout=15)
                time.sleep(2)


def main() -> int:
    strategies = [s.strip() for s in os.getenv("SCREEN_STRATEGIES", "momentum_quality,quality_value,balanced_alpha").split(",") if s.strip()]
    top_n = int(os.getenv("SCREEN_TOP_N", "10"))
    max_cards = int(os.getenv("SCREEN_MAX_CARDS", "20"))
    watch = {digits(c) for c in os.getenv("STOCK_LIST", "").split(",") if c.strip()}
    log.info("策略=%s top_n=%s 已有自选=%d 只", strategies, top_n, len(watch))

    cfg = Config.from_env()
    hits = run_strategies(strategies, top_n, extra=len(watch), cfg=cfg)
    fresh = {c: v for c, v in hits.items() if digits(c) not in watch}
    log.info("候选 %d 只（排除自选后 %d 只）", len(hits), len(fresh))
    if not fresh:
        log.error("没有候选，终止")
        return 1

    # 先按共识度和评分预排，只给前 N*1.5 只算 ATR，节省数据源调用
    pre = sorted(fresh.values(), key=lambda v: (-len(v["hits"]), -v["pick"].final_score))[: int(max_cards * 1.5)]
    for item in pre:
        item["plan"] = build_plan(item["pick"].code, item["pick"])
        time.sleep(0.5)
    pre.sort(key=lambda v: (-len(v["hits"]), VERDICT_ORDER.get((v["plan"] or {}).get("verdict"), 4), -v["pick"].final_score))
    final = pre[:max_cards]

    date = datetime.now().strftime("%m-%d")
    summary = [f"## 🔎 选股发现汇总 {date}", "", f"策略：{' · '.join(LABELS.get(s, s) for s in strategies)}（每套取前 {top_n}，已排除自选）", ""]
    for i, it in enumerate(final, 1):
        p, pl = it["pick"], it["plan"]
        v = {"ok": "✅", "wait": "⏳", "weak": "⚠️", "skip": "❌"}.get((pl or {}).get("verdict"), "·")
        summary.append(f"{i}. {v} **{p.name}({p.code})** 命中{len(it['hits'])}套 | 评分 {p.final_score:.1f}")
    cards = [(f"🔎 选股发现汇总 {date}", "\n".join(summary), DASHBOARD_URL + "#discover")]
    for it in final:
        p = it["pick"]
        cards.append((f"🔎 {p.name} ({p.code})", card_body(it), DASHBOARD_URL + "#discover"))

    server_url, topic = resolve_ntfy_endpoint(os.getenv("NTFY_URL"))
    if not server_url or not topic:
        log.error("NTFY_URL 未配置")
        return 1
    post_all(cards, server_url, topic, (os.getenv("NTFY_TOKEN") or "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
