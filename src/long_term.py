# -*- coding: utf-8 -*-
"""
长线体检（持仓周期：数月以上）

与 risk_levels.py 的交易计划 V2（波段级）互补：V2 回答「这几天该不该出手」，本模块回答
「这笔长线持仓现在健不健康、该不该加、该不该撤」。三档结论：

  healthy 🟢 健康   长线趋势完好，无事件风险，拿着不用动
  caution 🟡 注意   趋势走平 / 跌破中期均线 / 回撤偏大 / 出现中等事件，收紧观察
  alert   🔴 警报   长线趋势破坏（MA60 下穿 MA120 且双双向下、或跌破 MA250 且 MA250 走平向下）、
                   或重大事件（立案/退市/ST/业绩大幅下修）、或回撤超阈值 → 考虑减仓
  unknown ⚪ 数据不足 历史 K 线不足 130 根，无法计算 MA120

判定全部基于日线 MA60/MA120/MA250、250 日回撤、ATR、筹码快照与公告/新闻关键词。
环境变量：
  LONG_TERM_STOP_PCT     长线止损容忍度（默认 0.20，即 -20%）
  LONG_TERM_DD_CAUTION   回撤注意阈值（默认 0.25）
  LONG_TERM_DD_ALERT     回撤警报阈值（默认 0.35）
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LEVEL_ICON = {"healthy": "🟢", "caution": "🟡", "alert": "🔴", "unknown": "⚪", "na": "➖"}
LEVEL_LABEL = {"healthy": "健康", "caution": "注意", "alert": "警报", "unknown": "数据不足", "na": "不适用"}
TREND_LABEL = {"up": "向上", "flat": "走平", "down": "向下", "unknown": "未知"}

# 事件风险关键词分级：severe 直接警报；moderate 记为注意
SEVERE_EVENT_WORDS = ("立案", "退市", "ST", "暂停上市", "终止上市", "重大违法", "财务造假", "破产", "预亏", "大幅下修", "业绩变脸")
MODERATE_EVENT_WORDS = ("减持", "质押", "处罚", "警示", "问询", "关注函", "监管", "诉讼", "仲裁", "下修", "亏损", "解禁", "违规", "停牌")


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.getenv(name, "").strip() or default)
        return v if v > 0 else default
    except ValueError:
        return default


def _f(v: Any) -> float:
    try:
        x = float(v)
        return x if x == x else 0.0  # NaN -> 0
    except (TypeError, ValueError):
        return 0.0


def _fmt(p: float, ref: float) -> str:
    if p is None or p <= 0:
        return "—"
    return f"{p:.3f}" if ref < 10 else f"{p:.2f}"


def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if a > 0 and b > 0 else 0.0


def _spct(a: float, b: float) -> str:
    v = _pct(a, b)
    return f"{v:+.1f}%"


def _slope_pct(series, lookback: int) -> Optional[float]:
    """近 lookback 根 K 线均线的变化百分比；不足返回 None。"""
    s = series.dropna()
    if len(s) <= lookback:
        return None
    a, b = float(s.iloc[-1 - lookback]), float(s.iloc[-1])
    return _pct(b, a) if a > 0 else None


def _trend_from(ma60: float, ma120: float, s60: Optional[float], s120: Optional[float]) -> str:
    if ma60 <= 0 or ma120 <= 0:
        return "unknown"
    up_bias = ma60 > ma120
    s120 = 0.0 if s120 is None else s120
    s60 = 0.0 if s60 is None else s60
    if up_bias and s120 >= -0.3 and s60 >= 0:
        return "up"
    if (not up_bias) and s120 <= 0.3 and s60 <= 0:
        return "down"
    return "flat"


def _scan_events(text: Optional[str]) -> Tuple[List[str], List[str]]:
    """从公告/新闻文本中提取事件关键词。返回 (severe, moderate)。只看带 ⚠️ 或标题行，减少误伤。"""
    if not text:
        return [], []
    # 只扫描标题性质的行（以 - / * / 数字 / ⚠️ 开头或含「公告」），避免把分析正文里的泛泛提及算作事件
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cand = [ln for ln in lines if re.match(r"^(\d+[\.、)]|[-*•]|⚠️|【)", ln) or "公告" in ln or "⚠️" in ln]
    blob = "\n".join(cand) if cand else text
    severe = [w for w in SEVERE_EVENT_WORDS if w in blob]
    moderate = [w for w in MODERATE_EVENT_WORDS if w in blob and w not in severe]
    # 「ST」容易误匹配英文，要求独立出现
    if "ST" in severe and not re.search(r"(?<![A-Za-z])\*?ST(?![A-Za-z])", blob):
        severe.remove("ST")
    return severe, moderate


def compute_long_term_health(
    df,
    price: float,
    code: str = "",
    name: str = "",
    asset: str = "STOCK",
    chip: Any = None,
    news_context: Optional[str] = None,
    atr14: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    df:     日线 DataFrame（升序，需 close/high/low 列；250 根以上最佳，≥130 根可出 MA120）
    price:  现价
    asset:  risk_levels.classify_asset 的结果：STOCK / STOCK_ETF / INDEX / BOND_ETF / MONEY_ETF
    chip:   ChipDistribution 或 None
    """
    price = _f(price)
    out: Dict[str, Any] = {
        "version": 1, "code": code, "name": name, "price": price, "asset": asset,
        "level": "unknown", "level_label": LEVEL_LABEL["unknown"], "trend": "unknown", "trend_label": TREND_LABEL["unknown"],
        "ma60": 0.0, "ma120": 0.0, "ma250": 0.0, "slope60": None, "slope120": None, "slope250": None,
        "high250": 0.0, "drawdown": None, "bars": 0,
        "add_window": False, "add_text": "", "add_zone": None,
        "stop": None, "stop_basis": "", "stop_pct_cfg": _env_float("LONG_TERM_STOP_PCT", 0.20),
        "reduce": False, "reduce_text": "",
        "events_severe": [], "events_moderate": [],
        "chip_text": "", "checks": [], "notes": [], "reasons": [],
    }
    if asset in ("BOND_ETF", "MONEY_ETF"):
        out.update(level="na", level_label=LEVEL_LABEL["na"], reasons=["债券/货币类 ETF 不适用趋势体检"])
        return out
    if df is None or price <= 0:
        return out
    try:
        import pandas as pd  # noqa: F401

        d = df.copy()
        if "close" not in d.columns:
            return out
        close = d["close"].astype(float).reset_index(drop=True)
        high = d["high"].astype(float).reset_index(drop=True) if "high" in d.columns else close
        n = len(close)
        out["bars"] = n
        if n < 130:
            out["notes"].append(f"历史 K 线仅 {n} 根，不足 130，无法计算 MA120；请把 DAILY_HISTORY_DAYS 提到 300")
            return out

        ma60 = close.rolling(60).mean()
        ma120 = close.rolling(120).mean()
        ma250 = close.rolling(250).mean() if n >= 250 else None
        v60, v120 = float(ma60.iloc[-1]), float(ma120.iloc[-1])
        v250 = float(ma250.iloc[-1]) if ma250 is not None and ma250.notna().iloc[-1] else 0.0
        s60, s120 = _slope_pct(ma60, 20), _slope_pct(ma120, 20)
        s250 = _slope_pct(ma250, 20) if ma250 is not None else None
        h250 = float(high.iloc[-min(n, 250):].max())
        dd = _pct(price, h250)  # 负数

        out.update(ma60=v60, ma120=v120, ma250=v250, slope60=s60, slope120=s120, slope250=s250, high250=h250, drawdown=dd)
        trend = _trend_from(v60, v120, s60, s120)
        out.update(trend=trend, trend_label=TREND_LABEL[trend])

        dd_caution = _env_float("LONG_TERM_DD_CAUTION", 0.25) * 100
        dd_alert = _env_float("LONG_TERM_DD_ALERT", 0.35) * 100

        severe, moderate = _scan_events(news_context)
        out.update(events_severe=severe, events_moderate=moderate)

        # ---------- 体检项（固定 6 项，供看板解析） ----------
        checks: List[Dict[str, str]] = []

        def ck(label: str, ok: Optional[bool], detail: str) -> None:
            checks.append({"label": label, "status": "pass" if ok else ("na" if ok is None else "fail"), "detail": detail})

        ck("现价在 MA120 上方", price > v120, f"现价较 MA120 {_spct(price, v120)}")
        ck("MA60 高于 MA120（中期多头）", v60 > v120, f"MA60 {_fmt(v60, price)} / MA120 {_fmt(v120, price)}")
        ck("MA120 近 20 日向上", (s120 or 0) > 0, f"MA120 近 20 日 {s120:+.1f}%" if s120 is not None else "样本不足")
        if v250 > 0:
            ck("现价在 MA250 上方（年线）", price > v250, f"现价较 MA250 {_spct(price, v250)}")
        else:
            ck("现价在 MA250 上方（年线）", None, f"K 线 {n} 根不足 250，年线未计算")
        ck(f"距 250 日高点回撤 < {dd_caution:.0f}%", dd > -dd_caution, f"回撤 {dd:+.1f}%（高点 {_fmt(h250, price)}）")
        ev_ok = not severe and not moderate
        ev_detail = "公告/新闻未见风险关键词" if ev_ok else ("⚠️ " + "、".join(severe + moderate))
        if news_context is None:
            ck("无公告/新闻事件风险", None, "未获取到公告/新闻，无法排查")
        else:
            ck("无公告/新闻事件风险", ev_ok, ev_detail)
        out["checks"] = checks

        # ---------- 筹码（只做说明，不参与分级） ----------
        if chip is not None:
            pr, ac, c90 = _f(getattr(chip, "profit_ratio", 0)), _f(getattr(chip, "avg_cost", 0)), _f(getattr(chip, "concentration_90", 0))
            if ac > 0:
                rel = _spct(price, ac)
                tone = ("获利盘极多，上方缺乏套牢抛压但获利了结压力大" if pr >= 0.85 else
                        "多数持仓被套，若成交清淡属于被动锁仓而非派发" if pr <= 0.2 else "获利/套牢结构均衡")
                out["chip_text"] = f"获利比例 {pr * 100:.0f}%，平均成本 {_fmt(ac, price)}（现价 {rel}），90% 集中度 {c90 * 100:.1f}%；{tone}"

        # ---------- 分级 ----------
        reasons: List[str] = []
        level = "healthy"
        below250_broken = v250 > 0 and price < v250 and (s250 is not None and s250 <= 0)
        trend_broken = trend == "down" and v60 < v120 and (s60 or 0) < 0 and (s120 or 0) < 0
        if severe:
            level = "alert"; reasons.append(f"重大事件：{'、'.join(severe)}")
        if trend_broken:
            level = "alert"; reasons.append("MA60 已下穿 MA120 且双双向下，长线趋势破坏")
        if below250_broken:
            level = "alert"; reasons.append(f"跌破年线 MA250 {_fmt(v250, price)} 且年线走平/向下")
        if dd <= -dd_alert:
            level = "alert"; reasons.append(f"距 250 日高点回撤 {dd:+.1f}%，超过 {dd_alert:.0f}% 警报线")
        if level != "alert":
            if price < v120:
                level = "caution"; reasons.append(f"现价跌破 MA120（{_spct(price, v120)}）")
            if v60 < v120:
                level = "caution"; reasons.append("MA60 低于 MA120，中期转弱")
            if trend == "flat":
                level = "caution" if level != "healthy" or price < v60 else level
                if price < v60:
                    reasons.append("长线趋势走平且现价在 MA60 下方")
            if dd <= -dd_caution:
                level = "caution"; reasons.append(f"距 250 日高点回撤 {dd:+.1f}%，超过 {dd_caution:.0f}% 注意线")
            if moderate:
                level = "caution"; reasons.append(f"事件风险：{'、'.join(moderate)}")
        if level == "healthy":
            reasons.append("长线趋势向上、现价在 MA120 上方、回撤可控、未见事件风险")
        out.update(level=level, level_label=LEVEL_LABEL[level], reasons=reasons)

        # ---------- 长线止损 ----------
        stop_pct = out["stop_pct_cfg"]
        cand = [price * (1 - stop_pct)]
        basis = [f"现价 −{stop_pct * 100:.0f}%"]
        if v250 > 0 and v250 < price:
            cand.append(v250 * 0.97); basis.append("年线 MA250 下方 3%")
        stop = max(cand)  # 取更近的那个：年线有效时优先用结构位，否则用百分比
        out.update(stop=stop, stop_basis=basis[cand.index(stop)])

        # ---------- 加仓窗口 ----------
        add_ok = trend == "up" and level == "healthy" and v60 > 0
        lo, hi = (v120, v60 * 1.02) if v120 < v60 else (v60 * 0.97, v60 * 1.02)
        in_zone = lo <= price <= hi
        if add_ok and in_zone:
            out.update(add_window=True, add_zone=(lo, hi),
                       add_text=f"开启：现价回到 MA60～MA120 区间（{_fmt(lo, price)}～{_fmt(hi, price)}），趋势向上且体检健康，可分批加仓；止损 {_fmt(stop, price)}")
        elif add_ok:
            out.update(add_zone=(lo, hi),
                       add_text=f"关闭：等待回踩 {_fmt(lo, price)}～{_fmt(hi, price)}（MA60～MA120），现价 {_spct(price, hi) if price > hi else _spct(price, lo)}")
        else:
            out["add_text"] = "关闭：趋势未确认向上或体检非健康，不加仓"

        # ---------- 减仓信号 ----------
        if level == "alert":
            out.update(reduce=True, reduce_text="有：" + "；".join(reasons[:2]) + "。建议先减至轻仓，待月线趋势重新向上再回补")
        elif level == "caution" and price < v120 and (s120 or 0) < 0:
            out.update(reduce=True, reduce_text="有（弱）：跌破 MA120 且 MA120 转头向下，可先减一部分，跌破年线再减")
        else:
            out["reduce_text"] = "无"
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("[long_term] %s 长线体检计算失败: %s", code, exc)
        out["notes"].append(f"计算失败：{exc}")
        return out


def render_long_term_markdown(lt: Dict[str, Any]) -> List[str]:
    """渲染为 markdown（也是看板解析契约：键名与标点勿随意改）。"""
    if not lt:
        return []
    price = lt.get("price") or 0
    f = lambda v: _fmt(v, price)  # noqa: E731
    icon = LEVEL_ICON.get(lt["level"], "")
    lines = ["**🧭 长线体检**（持仓周期：数月以上）", ""]
    lines.append(f"- **体检结论**: {icon} {lt['level_label']}")
    if lt["level"] in ("na", "unknown"):
        for nt in lt.get("notes") or lt.get("reasons") or []:
            lines.append(f"- **说明**: {nt}")
        lines.append("")
        return lines
    s120 = lt.get("slope120")
    trend_detail = f"MA60 {'>' if lt['ma60'] > lt['ma120'] else '<'} MA120" + (f"，MA120 近 20 日 {s120:+.1f}%" if s120 is not None else "")
    lines.append(f"- **长线趋势**: {lt['trend_label']}（{trend_detail}）")
    pos = f"现价较 MA120 {_spct(price, lt['ma120'])}"
    if lt["ma250"] > 0:
        pos += f"，较 MA250 {_spct(price, lt['ma250'])}"
    if lt.get("drawdown") is not None:
        pos += f"；距 250 日高点回撤 {lt['drawdown']:+.1f}%"
    lines.append(f"- **位置**: {pos}")
    lines.append(f"- **加仓窗口**: {lt['add_text']}")
    if lt.get("stop"):
        lines.append(f"- **长线止损**: {f(lt['stop'])}（{lt['stop_basis']}，距现价{_spct(lt['stop'], price)}）")
    lines.append(f"- **减仓信号**: {lt['reduce_text']}")
    ev = lt.get("events_severe", []) + lt.get("events_moderate", [])
    lines.append(f"- **事件风险**: {('⚠️ ' + '、'.join(ev)) if ev else '无'}")
    if lt.get("chip_text"):
        lines.append(f"- **筹码**: {lt['chip_text']}")
    if lt.get("checks"):
        mark = {"pass": "✅", "fail": "❌", "na": "⚪"}
        lines.append("- **体检项**: " + "；".join(f"{mark[c['status']]} {c['label']}（{c['detail']}）" for c in lt["checks"]))
    if lt.get("reasons"):
        lines.append(f"- **结论依据**: {'；'.join(lt['reasons'])}")
    lines.append("")
    return lines


def apply_long_term_health(result: Any, df, chip: Any = None, news_context: Optional[str] = None, trend_result: Any = None) -> None:
    """计算长线体检并写入 result.dashboard['battle_plan']['long_term']。就地修改，失败静默。"""
    if not result or df is None:
        return
    try:
        from src.risk_levels import classify_asset

        code = str(getattr(result, "code", "") or "")
        name = str(getattr(result, "name", "") or "")
        price = getattr(result, "current_price", None)
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            tr = trend_result if isinstance(trend_result, dict) else getattr(trend_result, "__dict__", {}) or {}
            price = _f(tr.get("current_price")) or _f(df["close"].iloc[-1])
        atr = 0.0
        if trend_result is not None:
            tr = trend_result if isinstance(trend_result, dict) else getattr(trend_result, "__dict__", {}) or {}
            atr = _f(tr.get("atr14"))
        lt = compute_long_term_health(df, price, code=code, name=name, asset=classify_asset(code, name),
                                      chip=chip, news_context=news_context, atr14=atr)
        if not lt:
            return
        if not getattr(result, "dashboard", None):
            result.dashboard = {}
        battle = result.dashboard.get("battle_plan")
        if not isinstance(battle, dict):
            battle = {}
            result.dashboard["battle_plan"] = battle
        battle["long_term"] = lt
        logger.info("[long_term] %s level=%s trend=%s dd=%s add=%s reduce=%s bars=%s", code, lt["level"], lt["trend"],
                    None if lt.get("drawdown") is None else round(lt["drawdown"], 1), lt["add_window"], lt["reduce"], lt["bars"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("[long_term] %s 写入失败: %s", getattr(result, "code", "?"), exc)
