#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选股发现：全市场量化初筛（DSA 选股引擎）→ 排除已在自选的 → ATR 风控点位 → 推送到 ntfy。

环境变量：
  SCREEN_STRATEGIES  逗号分隔的策略 id，默认 momentum_quality,balanced_alpha
  SCREEN_TOP_N       每套策略取前 N（默认 10）
  SCREEN_MAX_CARDS   最终推送的候选数上限，全部做完整 DSA 分析（默认 8）
  SCREEN_REVIEW      1=基本面过滤 + AI 复核（默认 1）；SCREEN_DEEP=1 做完整分析（默认 1）
  STOCK_LIST         已在自选里的代码，会被排除
  NTFY_URL / NTFY_TOKEN
"""
from __future__ import annotations

import json
import logging
import concurrent.futures as cf
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

from src.notification_sender.ntfy_sender import _hard_split_by_lines, resolve_ntfy_endpoint  # noqa: E402
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
    "volume_breakout": "放量突破",
}
DAILY_HEAVY = {"shrink_pullback", "volume_breakout", "low_volatility_quality", "oversold_reversal"}  # 硬条件依赖日线指标，需要多补算一些候选
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
        for use_llm in (False,):
            try:
                kw = {"daily_enrich_max_candidates": int(os.getenv("SCREEN_ENRICH_MAX", "160"))}
                res = screen(sid, market="cn", max_output=top_n + extra, use_llm=use_llm, daily_enrich=True, config=cfg, **kw)
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
            return None, {}
        tr = StockTrendAnalyzer().analyze(df, code)
        price = float(pick.price or 0) or float(tr.current_price or 0)
        plan = compute_risk_plan(price, tr.ma5, tr.ma10, tr.atr14, tr.high_20, tr.high_60, tr.ma20)
        # 与自选看板一致的「趋势 / 支撑位 / 压力位」
        if tr.ma5 and tr.ma10 and tr.ma20 and tr.ma5 > tr.ma10 > tr.ma20:
            trend = "看多"
        elif tr.ma5 and tr.ma10 and tr.ma20 and tr.ma5 < tr.ma10 < tr.ma20:
            trend = "看空"
        else:
            trend = "震荡"
        below = [v for v in (tr.ma5, tr.ma10, tr.ma20, tr.swing_low_10) if v and v < price]
        above = [v for v in (tr.high_20, tr.high_60) if v and v > price * 1.002]
        extras = {"trend": trend, "support": max(below) if below else None, "resistance": min(above) if above else None}
        return plan, extras
    except Exception as exc:  # noqa: BLE001
        log.warning("%s ATR 计划失败: %s", code, exc)
        return None, {}


def fnum(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def reason_for(p) -> str:
    """LLM 重排不可用时，用评分因子拼一句可读理由。"""
    if (p.ranking_reason or "").strip():
        return clip(p.ranking_reason)
    bits = []
    if p.pe_ratio is not None and p.pb_ratio is not None:
        bits.append(f"PE {fnum(p.pe_ratio)} / PB {fnum(p.pb_ratio, 2)}")
    if p.ma_bullish:
        bits.append("均线多头排列")
    elif p.price_above_ma20:
        bits.append("站上 MA20")
    if p.macd_status:
        bits.append(f"MACD {p.macd_status}")
    if p.rsi_status:
        bits.append(f"RSI {p.rsi_status}")
    if p.change_60d is not None:
        bits.append(f"近 60 日 {p.change_60d:+.1f}%")
    if p.breakout_20d_pct is not None:
        bits.append(f"距 20 日高点 {p.breakout_20d_pct:+.1f}%")
    if p.volume_ratio is not None:
        bits.append(f"量比 {fnum(p.volume_ratio, 2)}")
    return "因子入选：" + "；".join(bits) if bits else "因子综合评分靠前"


FIN_WORDS = ("银行", "证券", "保险", "信托", "期货", "海通", "中信", "平安", "人寿", "财险")


def sector_key(p) -> str:
    if p.industry:
        return p.industry
    return "金融" if any(w in p.name for w in FIN_WORDS) else f"other:{p.code}"


def mv_text(mv):
    try:
        mv = float(mv)
    except (TypeError, ValueError):
        return "—"
    return f"{mv / 1e8:.0f}" if mv > 1e7 else f"{mv:.0f}"


# ---------------- 基本面过滤 ----------------
def fetch_fundamentals(codes):
    out = {}
    try:
        from data_provider.fundamental_adapter import AkshareFundamentalAdapter

        adapter = AkshareFundamentalAdapter()
    except Exception as exc:  # noqa: BLE001
        log.warning("基本面适配器不可用: %s", exc)
        return out

    def one(code):
        b = adapter.get_fundamental_bundle(code) or {}
        g = b.get("growth") or {}
        fr = (b.get("earnings") or {}).get("financial_report") or {}
        return {
            "roe": g.get("roe"), "profit_yoy": g.get("net_profit_yoy"), "rev_yoy": g.get("revenue_yoy"),
            "gm": g.get("gross_margin"), "ocf": fr.get("operating_cash_flow"), "np": fr.get("net_profit_parent"),
            "date": fr.get("report_date"),
        }

    ex = cf.ThreadPoolExecutor(max_workers=4)
    futs = {ex.submit(one, c): c for c in codes}
    for f, c in futs.items():
        try:
            out[c] = f.result(timeout=75)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s 基本面获取失败: %s", c, exc)
    ex.shutdown(wait=False, cancel_futures=True)
    return out


def _f(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def fund_score(f):
    """返回 (分数, 是否剔除, 原因)。数据缺失不剔除，只标记未知。"""
    if not f:
        return 0, False, "基本面数据缺失"
    roe, py, ry, ocf = _f(f.get("roe")), _f(f.get("profit_yoy")), _f(f.get("rev_yoy")), _f(f.get("ocf"))
    q = 0
    if roe is not None:
        q += 2 if roe >= 10 else 1 if roe >= 5 else -1 if roe < 3 else 0
    if py is not None:
        q += 2 if py >= 10 else 1 if py >= 0 else -2 if py < -10 else -1
    if ry is not None:
        q += 1 if ry >= 5 else -1 if ry < -10 else 0
    if ocf is not None:
        q += 1 if ocf > 0 else -1
    if roe is not None and roe < 0:
        return q, True, f"ROE 为负({roe:.1f}%)，公司亏损"
    if py is not None and py < -20:
        return q, True, f"净利润同比 {py:.0f}%，业绩大幅下滑"
    if q <= -2:
        return q, True, "盈利/增长/现金流综合偏弱"
    return q, False, ""


def fund_line(f):
    if not f:
        return "基本面数据暂缺"
    parts = []
    for label, key, nd in (("ROE", "roe", 1), ("净利润同比", "profit_yoy", 1), ("营收同比", "rev_yoy", 1), ("毛利率", "gm", 1)):
        v = _f(f.get(key))
        if v is not None:
            parts.append(f"{label} {v:.{nd}f}%")
    ocf = _f(f.get("ocf"))
    if ocf is not None:
        parts.append("经营现金流" + ("为正" if ocf > 0 else "为负"))
    suffix = f"（报告期 {f['date']}）" if f.get("date") else ""
    return (" | ".join(parts) + suffix) if parts else "基本面数据暂缺"


# ---------------- AI 复核（Gemini，直接调用 REST） ----------------
def _gemini_keys():
    keys = [k.strip() for k in (os.getenv("GEMINI_API_KEYS") or "").split(",") if k.strip()]
    one = (os.getenv("GEMINI_API_KEY") or "").strip()
    if one:
        keys.insert(0, one)
    return keys


def _parse_json_array(text):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
    except ValueError:
        a, b = text.find("["), text.rfind("]")
        if a < 0 or b <= a:
            raise
        data = json.loads(text[a: b + 1])
    if isinstance(data, dict):
        data = data.get("items") or data.get("results") or list(data.values())[0]
    return data


def llm_review(items, funds):
    keys = _gemini_keys()
    if not keys or not items:
        log.warning("未配置 GEMINI_API_KEY，跳过 AI 复核")
        return {}
    rows = []
    for it in items:
        p, f = it["pick"], funds.get(it["pick"].code) or {}
        rows.append({
            "code": digits(p.code), "name": p.name, "industry": p.industry or "", "PE": fnum(p.pe_ratio), "PB": fnum(p.pb_ratio, 2),
            "市值亿": mv_text(p.total_mv), "ROE%": _f(f.get("roe")), "净利润同比%": _f(f.get("profit_yoy")), "营收同比%": _f(f.get("rev_yoy")),
            "毛利率%": _f(f.get("gm")), "经营现金流为正": (None if _f(f.get("ocf")) is None else _f(f.get("ocf")) > 0),
            "趋势": (it.get("extras") or {}).get("trend"), "近60日涨幅%": _f(getattr(p, "change_60d", None)),
            "命中策略": [LABELS.get(s, s) for s, _ in it["hits"]], "入场盈亏比(回踩)": (it.get("plan") or {}).get("rr_entry"),
        })
    prompt = (
        "你是严谨的 A 股买方研究员。下面是量化初筛得到的候选股及其客观数据。请只依据所给数据和你对这些公司的已知常识，"
        "判断每只是否值得进一步跟踪，重点看：盈利质量(ROE/现金流)、增长、估值是否合理、行业与周期风险、趋势是否支持。"
        "不要编造具体新闻或数字；数据缺失就写“数据不足”。宁缺毋滥：判为“跟踪”的不超过 40%。\n"
        "只输出 JSON 数组，每个元素：{\"code\":\"6位代码\",\"verdict\":\"跟踪|观望|回避\",\"score\":0-100整数,"
        "\"reason\":\"不超过60字，写清楚为什么\",\"risk\":\"不超过30字，最大风险\"}。\n候选：\n" + json.dumps(rows, ensure_ascii=False)
    )
    models = [m for m in (os.getenv("GEMINI_MODEL"), os.getenv("GEMINI_MODEL_FALLBACK")) if m]
    models = [m.split("/")[-1] for m in models] or ["gemini-2.5-flash"]
    for model in models:
        for key in keys[:2]:
            for attempt in range(2):
                try:
                    r = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": prompt}]}],
                              "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192, "responseMimeType": "application/json"}},
                        timeout=120,
                    )
                    if r.status_code != 200:
                        log.warning("AI 复核 %s HTTP %s: %s", model, r.status_code, r.text[:200])
                        if r.status_code in (429, 500, 503):
                            time.sleep(8 * (attempt + 1))
                            continue
                        break
                    parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                    data = _parse_json_array("".join(x.get("text", "") for x in parts if not x.get("thought")))
                    out = {}
                    for d in data:
                        c = digits(d.get("code"))
                        if c:
                            out[c] = {"verdict": str(d.get("verdict", "")).strip(), "score": d.get("score"),
                                      "reason": clip(d.get("reason"), 120), "risk": clip(d.get("risk"), 60)}
                    log.info("AI 复核完成 model=%s，%d/%d 只有结论", model, len(out), len(items))
                    if out:
                        return out
                except Exception as exc:  # noqa: BLE001
                    log.warning("AI 复核异常 %s: %s", model, exc)
                    time.sleep(3)
    return {}


# ---------------- 完整 DSA 分析 ----------------
def deep_analyze(items):
    """对入选标的跑和自选一致的完整分析，返回 {code6: (摘要行, 报告正文)}。"""
    out = {}
    if not items:
        return out
    try:
        from src.config import get_config
        from src.core.pipeline import StockAnalysisPipeline
        from src.formatters import strip_hidden_markdown_metadata

        pipeline = StockAnalysisPipeline(config=get_config(), max_workers=1, query_source="cli",
                                         daily_market_context_enabled=False, daily_market_context_allow_generate=False)
        codes = [digits(it["pick"].code) for it in items]
        results = pipeline.run(stock_codes=codes, send_notification=False) or []
        log.info("完整分析完成 %d/%d 只", len(results), len(codes))
        for r in results:
            rep = strip_hidden_markdown_metadata(pipeline.notifier.generate_dashboard_report([r])).strip()
            code = digits(getattr(r, "code", ""))
            body, summary = "", ""
            for part in re.split(r"(?=\n## )", "\n" + rep):
                first = part.strip().split("\n", 1)[0]
                if f"({getattr(r, 'code', '')})" in first and "摘要" not in first and "仪表盘" not in first:
                    lines = part.strip().split("\n", 1)
                    body = lines[1].strip() if len(lines) > 1 else ""
                    break
            for line in rep.split("\n"):
                if re.search(rf"\*\*.+?\({re.escape(str(getattr(r, 'code', '')))}\)\*\*", line) and "评分" in line:
                    summary = line.strip()
                    break
            if body:
                out[code] = (summary, body)
    except Exception:  # noqa: BLE001
        log.exception("完整 DSA 分析失败，退回轻量卡片")
    return out


def card_body(item, with_plan=True) -> str:
    p, plan = item["pick"], item["plan"]
    names = " · ".join(LABELS.get(s, s) for s, _ in item["hits"])
    lines = [
        f"## 🔎 {p.name} ({p.code})",
        "",
        f"**命中策略**: {names}",
        f"**综合评分**: {p.final_score:.1f}",
        f"**行业**: {p.industry or '—'}",
        f"**趋势**: {(item.get('extras') or {}).get('trend', '—')} | **支撑位**: {fnum((item.get('extras') or {}).get('support'), 3 if p.price < 20 else 2)} | **压力位**: {fnum((item.get('extras') or {}).get('resistance'), 3 if p.price < 20 else 2)}",
        f"**现价**: {p.price:.2f} | **涨跌**: {p.change_pct:+.2f}% | **PE**: {fnum(p.pe_ratio)} | **PB**: {fnum(p.pb_ratio, 2)} | **市值亿**: {mv_text(p.total_mv)}",
        "",
        f"**基本面**: {item.get('fund_line') or '—'}",
        *([f"**AI复核**: {item['ai']['verdict']} · {item['ai'].get('score', '—')}分 | {item['ai']['reason']} | **主要风险**: {item['ai']['risk'] or '—'}"] if item.get("ai") else []),
        f"**入选理由**: {reason_for(p)}",
        f"**风险提示**: {clip(p.risk_summary) or '—'}",
        "",
    ]
    if plan and with_plan:
        lines += render_risk_plan_markdown(plan)
    return "\n".join(lines)


def make_cards(it, deep):
    """轻量卡 or 完整分析卡（头部选股依据 + DSA 完整报告），按 3400 字节分段，标题带 (i/n)。"""
    p = it["pick"]
    d = deep.get(digits(p.code))
    if d:
        summary, body = d
        head = card_body(it, with_plan="风险收益测算" not in body)
        content = head + ("\n\n" + summary if summary else "") + "\n\n" + body
    else:
        content = card_body(it)
    chunks = _hard_split_by_lines(content)
    base = f"🔎 {p.name} ({p.code})"
    link = DASHBOARD_URL + "#discover"
    return [(base if len(chunks) == 1 else f"{base} ({i}/{len(chunks)})", c, link) for i, c in enumerate(chunks, 1)]


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
    strategies = [s.strip() for s in os.getenv("SCREEN_STRATEGIES", "momentum_quality,balanced_alpha").split(",") if s.strip()]
    top_n = int(os.getenv("SCREEN_TOP_N", "100"))
    max_cards = int(os.getenv("SCREEN_MAX_CARDS", "8"))
    watch = {digits(c) for c in os.getenv("STOCK_LIST", "").split(",") if c.strip()}
    log.info("策略=%s top_n=%s 已有自选=%d 只", strategies, top_n, len(watch))

    cfg = None  # screen() 默认加载选股引擎自己的配置
    hits = run_strategies(strategies, top_n, extra=len(watch), cfg=cfg)
    fresh = {c: v for c, v in hits.items() if digits(c) not in watch}
    log.info("候选 %d 只（排除自选后 %d 只）", len(hits), len(fresh))
    if not fresh:
        log.error("没有候选，终止")
        return 1

    # 先按共识度和评分预排，只给前 N*1.5 只算 ATR，节省数据源调用
    ranked = sorted(fresh.values(), key=lambda v: (-len(v["hits"]), -v["pick"].final_score))

    def maybe_up(pk):
        above, bull = getattr(pk, "price_above_ma20", None), getattr(pk, "ma_bullish", None)
        return not (above is False and not bull)  # 明确在 MA20 下方且非多头排列 → 直接跳过

    cand = [v for v in ranked if maybe_up(v["pick"])]
    log.info("日线特征预筛：%d → %d 只（剔除明显空头）", len(ranked), len(cand))
    pre = cand[:100]

    def _plan(item):
        item["plan"], item["extras"] = build_plan(item["pick"].code, item["pick"])
        return item

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(_plan, pre))
    # 能出手的优先；空头排列(skip)的先剔除，好标的太少时才补位
    good = [v for v in pre if (v["plan"] or {}).get("verdict") in ("ok", "wait", "weak")]
    rest = [v for v in pre if v not in good]
    key = lambda v: (VERDICT_ORDER.get((v["plan"] or {}).get("verdict"), 4), -len(v["hits"]), -v["pick"].final_score)
    good.sort(key=key)
    rest.sort(key=key)
    log.info("可入场/观察 %d 只，空头/不出手 %d 只", len(good), len(rest))
    pre = good + (rest[: max(0, 8 - len(good))] if len(good) < 8 else [])
    # 行业分散：同一行业（金融类合并）最多 2 只
    per_sector, final = Counter(), []
    for it in pre:
        k = sector_key(it["pick"])
        if per_sector[k] >= 2:
            continue
        per_sector[k] += 1
        final.append(it)
        if len(final) >= max(max_cards * 3, 20):
            break
    log.info("行业分散后 %d 只：%s", len(final), dict(per_sector.most_common(6)))

    # ---- 基本面过滤 + AI 复核 ----
    if os.getenv("SCREEN_REVIEW", "1") != "0":
        funds = fetch_fundamentals([it["pick"].code for it in final])
        kept = []
        for it in final:
            f = funds.get(it["pick"].code)
            q, drop, why = fund_score(f)
            it["fund_q"], it["fund_line"] = q, fund_line(f)
            if drop:
                log.info("基本面剔除 %s(%s)：%s", it["pick"].name, it["pick"].code, why)
                continue
            kept.append(it)
        log.info("基本面过滤：%d → %d 只", len(final), len(kept))
        if len(kept) >= 3:
            final = kept
        ai = llm_review(final, funds)
        for it in final:
            it["ai"] = ai.get(digits(it["pick"].code))
        kept = [it for it in final if not (it.get("ai") and it["ai"]["verdict"] == "回避")]
        log.info("AI 复核：%d → %d 只（剔除“回避”）", len(final), len(kept))
        if len(kept) >= 3:
            final = kept
        av = {"跟踪": 0, "观望": 1, "": 2}
        final.sort(key=lambda v: (av.get((v.get("ai") or {}).get("verdict", ""), 2), VERDICT_ORDER.get((v["plan"] or {}).get("verdict"), 4),
                                  -v.get("fund_q", 0), -len(v["hits"]), -v["pick"].final_score))
    final = final[:max_cards]
    deep = deep_analyze(final) if os.getenv("SCREEN_DEEP", "1") != "0" else {}

    date = datetime.now().strftime("%m-%d")
    summary = [f"## 🔎 选股发现汇总 {date}", "", f"策略：{' · '.join(LABELS.get(s, s) for s in strategies)}（每套取前 {top_n}，已排除自选）", ""]
    for i, it in enumerate(final, 1):
        p, pl = it["pick"], it["plan"]
        v = {"ok": "✅", "wait": "⏳", "weak": "⚠️", "skip": "❌"}.get((pl or {}).get("verdict"), "·")
        ai = (it.get("ai") or {}).get("verdict")
        summary.append(f"{i}. {v} **{p.name}({p.code})** {('AI ' + ai + ' | ') if ai else ''}命中{len(it['hits'])}套 | {it.get('fund_line') or ''}")
    cards = [(f"🔎 选股发现汇总 {date}", "\n".join(summary), DASHBOARD_URL + "#discover")]
    for it in final:
        cards.extend(make_cards(it, deep))

    server_url, topic = resolve_ntfy_endpoint(os.getenv("NTFY_URL"))
    if not server_url or not topic:
        log.error("NTFY_URL 未配置")
        return 1
    post_all(cards, server_url, topic, (os.getenv("NTFY_TOKEN") or "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
