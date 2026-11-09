# -*- coding: utf-8 -*-
"""
策略 V2：确定性交易计划（价位由代码按波动率与市场结构算出，不让 LLM 编数字）。

核心原则
  - 风险与收益分开算：止损由波动与波段低点决定，目标由上方真实结构压力决定，盈亏比因此才有筛选意义。
  - 回踩 / 突破两种策略分开处理，"ok" 必须表示信号已触发且当前仍可执行。
  - 国债 / 货币 ETF 与指数不适用本模型（直接标记 na）。
  - 仓位按账户风险反推（ACCOUNT_SIZE 环境变量），并给出极端风险。

状态（state）
  ok      入场条件已触发，计划盈亏比 ≥ 2.0 且现价盈亏比 ≥ 2.0，且未超出计划入场价 0.5×ATR
  wait    计划盈亏比 ≥ 2.0，但现价盈亏比 < 2.0（或价格在入场区上方，等回踩）
  watch   条件基本满足，尚未进入入场区 / 尚未确认止跌
  expired 确认后价格已超出计划入场价 1×ATR，不再追入
  skip    趋势不满足 / 止损距离异常 / 盈亏比 < 1.5 / 仓位不足 100 股
  na      指数、国债 ETF、货币 ETF 等不适用
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STOP_ATR_MULT_PULLBACK = 1.5
STOP_LOW_BUFFER_ATR = 0.3
STOP_ATR_MULT_BREAKOUT = 1.25
ZONE_ATR = 0.25
TARGET_MIN_GAP_ATR = 0.5
RR_MIN_PLAN = 1.5
RR_WAIT_PLAN = 2.0
RR_OK_NOW = 2.0
STOP_RANGE = {"STOCK": (0.02, 0.08), "STOCK_ETF": (0.01, 0.05)}
RISK_PCT_BY_MARKET = {"strong": 0.75, "neutral": 0.35, "weak": 0.0, "unknown": 0.35}
MAX_POSITION_PCT = 0.10

_AUTO_MARKET_CACHE: Optional[str] = None

STATE_ICON = {"ok": "✅", "wait": "⏳", "watch": "👀", "expired": "⌛", "skip": "❌", "na": "➖"}
STATE_LABEL = {"ok": "可以出手", "wait": "等待出手", "watch": "观察", "expired": "已失效", "skip": "不出手", "na": "不适用"}
STRATEGY_LABEL = {"pullback": "趋势回踩", "breakout": "突破回踩", "none": "无"}
MARKET_LABEL = {"strong": "强势", "neutral": "中性", "weak": "弱势", "unknown": "未接入（按中性处理）"}

_INDEX_NAMES = ("上证指数", "深证成指", "创业板指", "科创50", "上证50", "沪深300", "中证500", "中证1000", "中证2000", "北证50", "科创综指", "中证A500")
_BOND_WORDS = ("国债", "政金债", "政策性金融债", "信用债", "公司债", "地方债", "城投债", "利率债", "短融", "中票", "国开", "债券ETF", "债ETF")
_MONEY_WORDS = ("货币", "现金", "添益", "保证金", "日利", "快线", "活期", "理财", "货基")


def _fmt(p: float, price_ref: float) -> str:
    return f"{p:.3f}" if price_ref < 20 else f"{p:.2f}"


def _pct(p: float, base: float) -> float:
    return (p / base - 1.0) * 100.0 if base else 0.0


def _spct(p: float, base: float) -> str:
    return f"{_pct(p, base):+.1f}%"


def _digits(code: str) -> str:
    m = re.search(r"(\d{6})", str(code or ""))
    return m.group(1) if m else ""


def classify_asset(code: str, name: str = "") -> str:
    """STOCK / STOCK_ETF / BOND_ETF / MONEY_ETF / INDEX"""
    c = str(code or "").lower().strip()
    n = str(name or "")
    d = _digits(c)
    if re.match(r"^(sh000|sz399|sh932|csi|bj899)", c) or c.endswith((".csi", ".idx")) or "指数" in n or n in _INDEX_NAMES:
        return "INDEX"
    if d.startswith(("899", "932")) and not re.match(r"^(sz|sh)", c):
        return "INDEX"
    is_etf = bool(re.match(r"^(51|52|56|58|15|16)\d{4}$", d)) or "ETF" in n.upper()
    if is_etf:
        if any(w in n for w in _MONEY_WORDS):
            return "MONEY_ETF"
        if any(w in n for w in _BOND_WORDS) and "转债" not in n:
            return "BOND_ETF"
        if d.startswith("511") and "转债" not in n:
            return "BOND_ETF" if not any(w in n for w in _MONEY_WORDS) else "MONEY_ETF"
        return "STOCK_ETF"
    return "STOCK"


def _limit_pct(code: str, name: str, asset: str) -> float:
    d = _digits(code)
    if asset == "STOCK_ETF":
        return 0.20 if any(w in str(name) for w in ("科创", "创业", "双创", "北证")) else 0.10
    if d.startswith(("30", "68")):
        return 0.20
    if d.startswith(("4", "8", "92")):
        return 0.30
    return 0.10


def _account_size(explicit: Optional[float]) -> float:
    if explicit and explicit > 0:
        return float(explicit)
    try:
        v = float(str(os.getenv("ACCOUNT_SIZE", "")).replace(",", "").strip() or 0)
    except ValueError:
        v = 0
    return v if v > 0 else 100000.0


def _auto_detect_market_state() -> str:
    """用沪深300(sh000300)与创业板指(sz399006)相对 MA20 自动判大盘强弱（单进程缓存一次）。"""
    global _AUTO_MARKET_CACHE
    if _AUTO_MARKET_CACHE is not None:
        return _AUTO_MARKET_CACHE
    try:
        from data_provider import DataFetcherManager

        mgr = DataFetcherManager()
        up, total = 0, 0
        for idx_code in ("sh000300", "sz399006"):
            try:
                df, _ = mgr.get_daily_data(idx_code, days=35)
                if df is None or df.empty or len(df) < 20 or "close" not in df.columns:
                    continue
                close = df["close"].astype(float).reset_index(drop=True)
                c_now = float(close.iloc[-1])
                ma20 = float(close.iloc[-20:].mean())
                ma20_prev = float(close.iloc[-25:-5].mean()) if len(close) >= 25 else ma20
                if c_now <= 0 or ma20 <= 0:
                    continue
                total += 1
                if c_now > ma20 and ma20 >= ma20_prev * 0.998:
                    up += 1
            except Exception:  # noqa: BLE001
                continue
        if total == 0:
            _AUTO_MARKET_CACHE = "unknown"
        elif up == total and total >= 2:
            _AUTO_MARKET_CACHE = "strong"
        elif up == 0:
            _AUTO_MARKET_CACHE = "weak"
        else:
            _AUTO_MARKET_CACHE = "neutral"
    except Exception:  # noqa: BLE001
        _AUTO_MARKET_CACHE = "unknown"
    logger.info("[risk_levels] 自动判定市场状态: %s", _AUTO_MARKET_CACHE)
    return _AUTO_MARKET_CACHE


def _market_state(explicit: Optional[str]) -> str:
    s = (explicit or os.getenv("MARKET_STATE", "") or "unknown").strip().lower()
    if s in ("strong", "neutral", "weak"):
        return s
    return _auto_detect_market_state()


def _f(v: Any) -> float:
    try:
        x = float(v)
        return x if x == x else 0.0
    except (TypeError, ValueError):
        return 0.0


def compute_risk_plan(
    price: float,
    tr: Any,
    code: str = "",
    name: str = "",
    account_size: Optional[float] = None,
    market_state: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    price: 现价（分析时点；收盘后运行即收盘价）
    tr:    TrendAnalysisResult 或其 dict（需 ma5/ma10/ma20/atr14/swing_low_10/high_20/high_60/
           swing_highs/ma20_prev5/breakout_level/breakout_age/last_open/last_low）
    """
    t = tr if isinstance(tr, dict) else getattr(tr, "__dict__", {}) or {}
    g = lambda k: _f(t.get(k))  # noqa: E731
    asset = classify_asset(code, name)
    market = _market_state(market_state)
    atr = g("atr14")
    price = _f(price)
    base: Dict[str, Any] = {
        "version": 2, "asset": asset, "code": code, "name": name, "price": price,
        "atr14": atr, "atr_pct": (atr / price * 100.0) if price and atr else 0.0,
        "market_state": market, "market_label": MARKET_LABEL[market],
        "strategy": "none", "strategy_label": STRATEGY_LABEL["none"],
        "state": "skip", "state_text": "", "reasons": [],
        "zone": None, "entry": None, "stop": None, "risk": None, "target": None, "target_label": "", "trailing": False,
        "rr_plan": None, "rr_now": None,
        "shares": 0, "position_value": 0.0, "risk_pct": RISK_PCT_BY_MARKET[market], "risk_amount": 0.0,
        "extreme": None, "invalidation": "", "bearish": False,
    }

    if asset in ("INDEX", "BOND_ETF", "MONEY_ETF"):
        why = {"INDEX": "指数不可直接交易，本模型只用于个股与股票型 ETF",
               "BOND_ETF": "国债/债券 ETF 波动极低，ATR 止损与目标间距小于交易成本与除息跳空，不适用本模型",
               "MONEY_ETF": "货币类 ETF 无价格波动意义，不适用本模型"}[asset]
        base.update(state="na", state_text=why, reasons=[why])
        return base

    ma5, ma10, ma20, ma20p = g("ma5"), g("ma10"), g("ma20"), g("ma20_prev5")
    if price <= 0 or atr <= 0 or not (ma5 and ma10 and ma20):
        return None

    bearish = ma5 < ma10 < ma20
    base["bearish"] = bearish
    low10 = g("swing_low_10") or g("low_10")
    h20, h60 = g("high_20"), g("high_60")
    swing_highs = [_f(x) for x in (t.get("swing_highs") or []) if _f(x) > 0]
    last_open, last_low = g("last_open"), g("last_low")
    b_level, b_age = g("breakout_level"), t.get("breakout_age")
    reasons: List[str] = []

    # ---------- 策略识别 ----------
    ma20_up = ma20p <= 0 or ma20 > ma20p
    trend_ok = price > ma20 and ma5 > ma10 > ma20 and ma20_up
    breakout_trend_ok = price > ma20 and ma5 >= ma10 * 0.995 and ma20_up and not bearish
    strategy = "none"
    if b_level > 0 and b_age is not None and 0 <= int(b_age) <= 5 and breakout_trend_ok and (price - ma20) < 3 * atr:
        strategy = "breakout"
    elif trend_ok and (price - ma20) <= 2 * atr:
        strategy = "pullback"
    else:
        if bearish:
            reasons.append("均线空头排列（MA5<MA10<MA20）")
        elif price <= ma20:
            reasons.append(f"收盘价在 MA20（{_fmt(ma20, price)}）下方")
        elif not (ma5 > ma10 > ma20):
            reasons.append("均线未形成多头排列")
        elif ma20p > 0 and ma20 <= ma20p:
            reasons.append("MA20 未向上（较 5 日前走平或下行）")
        elif (price - ma20) > 2 * atr:
            reasons.append(f"距 MA20 超过 2×ATR（{_fmt(price - ma20, price)}），追高风险")
            base.update(state="skip", state_text="趋势向上但离 MA20 过远，不追高，等回落到 MA5/MA10 附近再评估", reasons=reasons)
            return base
        else:
            reasons.append("趋势条件不满足")
        base.update(state="skip", state_text="趋势条件不满足，不出手", reasons=reasons)
        return base

    base.update(strategy=strategy, strategy_label=STRATEGY_LABEL[strategy])

    # ---------- 入场 / 止损 ----------
    if strategy == "pullback":
        entry = ma5
        low_ref = low10 if low10 > 0 else entry - atr
        stop = min(entry - STOP_ATR_MULT_PULLBACK * atr, low_ref - STOP_LOW_BUFFER_ATR * atr)
        zone_lo, zone_hi = ma10 - ZONE_ATR * atr, ma5 + ZONE_ATR * atr
        reasons.append("多头排列、MA20 向上、收盘在 MA20 上方 2×ATR 内")
    else:
        entry = b_level
        b_low = g("breakout_low")
        stop_atr = entry - STOP_ATR_MULT_BREAKOUT * atr
        stop = min(stop_atr, b_low - STOP_LOW_BUFFER_ATR * atr) if (0 < b_low <= entry) else stop_atr
        zone_lo, zone_hi = entry - 0.5 * atr, entry + 0.5 * atr
        reasons.append(f"近 {int(b_age) + 1} 日内放量突破前 20 日高点 {_fmt(entry, price)}，且 MA20 向上")
    if zone_lo > zone_hi:
        zone_lo, zone_hi = zone_hi, zone_lo
    stop = max(stop, 0.0001)
    risk = entry - stop
    base.update(entry=entry, stop=stop, risk=risk, zone=(zone_lo, zone_hi))
    if risk <= 0:
        base.update(state="skip", state_text="止损计算异常", reasons=reasons + ["止损价不低于入场价"])
        return base

    lo_lim, hi_lim = STOP_RANGE["STOCK_ETF" if asset == "STOCK_ETF" else "STOCK"]
    stop_pct = risk / entry
    if stop_pct < lo_lim:
        reasons.append(f"止损距离 {stop_pct * 100:.1f}% 低于下限 {lo_lim * 100:.0f}%，交易成本与噪声占比过高")
        base.update(state="skip", state_text="止损距离过窄，不出手", reasons=reasons)
        return base
    if stop_pct > hi_lim:
        reasons.append(f"止损距离 {stop_pct * 100:.1f}% 超过上限 {hi_lim * 100:.0f}%，止损过宽")
        base.update(state="skip", state_text="止损距离过宽，放弃交易", reasons=reasons)
        return base

    # ---------- 结构目标 ----------
    cands: List[Tuple[float, str]] = []
    if h20 > 0:
        cands.append((h20, "前20日高点"))
    if h60 > 0:
        cands.append((h60, "前60日高点"))
    cands += [(v, "120日内波段高点") for v in swing_highs]
    floor_ = entry + TARGET_MIN_GAP_ATR * atr
    if strategy == "breakout":
        # 突破后的新高不是压力；目标必须高于「突破日至今的最高价」，否则视为上方无结构（追踪止损）
        floor_ = max(floor_, g("breakout_high") or 0.0, price)
    cands = [(v, lb) for v, lb in cands if v > floor_]
    target, target_label = (min(cands, key=lambda x: x[0]) if cands else (None, ""))
    trailing = target is None
    rr_plan = (target - entry) / risk if target else None
    rr_now = ((target - price) / (price - stop) if price > stop else 0.0) if target else None
    base.update(target=target, target_label=target_label, trailing=trailing, rr_plan=rr_plan, rr_now=rr_now)
    if trailing:
        reasons.append("上方无结构压力：不人为设目标，按追踪止损退出（突破型）")

    # ---------- 状态判定 ----------
    # 实盘纪律：要给 ok（直接出手），计划盈亏比与现价盈亏比都必须 ≥ 2.0（无结构目标走追踪止损除外）。
    # 1.5 <= rr_plan < 2.0 空间偏鸡肋，一律只给 watch 观察，绝不进「可以出手」。
    plan_strong = rr_plan is None or rr_plan >= RR_WAIT_PLAN
    now_strong = rr_now is None or rr_now >= RR_OK_NOW
    state, text = "skip", ""
    if rr_plan is not None and rr_plan < RR_MIN_PLAN:
        state, text = "skip", f"计划盈亏比 1:{rr_plan:.2f} 低于 {RR_MIN_PLAN}，上方空间不足"
        reasons.append(f"结构目标 {_fmt(target, price)}（{target_label}）太近")
    elif price < stop:
        state, text = "skip", ("跌破突破止损位，突破失败" if strategy == "breakout" else "已跌破止损位，计划失效")
    elif strategy == "pullback":
        entered = last_low > 0 and last_low <= zone_hi or price <= zone_hi
        if not entered:
            state, text = "watch", f"尚未进入入场区 {_fmt(zone_lo, price)}～{_fmt(zone_hi, price)}，等回踩"
        elif price < zone_lo - ZONE_ATR * atr:
            state, text = "watch", "跌破入场区下沿，等重新站上 MA10 再确认"
        else:
            confirmed = price >= ma5 or (last_open > 0 and price > last_open)
            if not confirmed:
                state, text = "watch", "已进入入场区，等收盘站回 MA5 或收阳确认止跌"
            elif price > entry + 1.0 * atr:
                state, text = "expired", "确认后价格已超出计划入场价 1×ATR，不追"
            elif not plan_strong:
                state, text = "watch", f"计划盈亏比 1:{rr_plan:.2f} 不足 2.0（空间偏近），仅观察不出手"
            elif price <= entry + 0.5 * atr and now_strong:
                state, text = "ok", "止跌已确认，盈亏比 ≥ 2.0，价格仍在入场价 0.5×ATR 内，可按计划入场"
            else:
                state, text = "wait", (f"现价追入盈亏比 1:{rr_now:.2f} 不足 {RR_OK_NOW:.1f}，等回到入场区" if rr_now is not None
                                       else "价格偏离入场价，等回到入场区")
    else:  # breakout
        if price > zone_hi:
            if plan_strong:
                state, text = "wait", f"突破后未回踩，等回踩突破位 {_fmt(zone_lo, price)}～{_fmt(zone_hi, price)}"
            else:
                state, text = "watch", f"计划盈亏比 1:{rr_plan:.2f} 不足 2.0（上方压力偏近），回踩时仅观察"
        elif price >= zone_lo:
            # 突破日当天（b_age==0）放量阳线本身就是确认；次日及之后回踩突破位，必须收阳或收盘守住突破位才确认支撑有效，避免阴线暴跌接飞刀
            b_confirmed = (int(b_age or 0) == 0) or price >= entry or (last_open > 0 and price >= last_open)
            if not plan_strong:
                state, text = "watch", f"计划盈亏比 1:{rr_plan:.2f} 不足 2.0（上方压力偏近），仅观察不出手"
            elif not b_confirmed:
                state, text = "watch", f"回踩突破位 {_fmt(entry, price)} 但当日收阴未企稳，等收阳或站回突破位再出手"
            elif now_strong:
                state, text = "ok", "突破回踩企稳且盈亏比 ≥ 2.0，可按计划入场"
            else:
                state, text = "wait", f"现价追入盈亏比 1:{rr_now:.2f} 不足 {RR_OK_NOW:.1f}，等更靠近突破位"
        else:
            state, text = "watch", "跌回突破位下方，突破待重新确认"

    # ---------- 仓位 ----------
    acct = _account_size(account_size)
    risk_pct = RISK_PCT_BY_MARKET[market]
    risk_amt = acct * risk_pct / 100.0
    shares = int(math.floor(risk_amt / risk / 100.0) * 100) if risk > 0 and risk_pct > 0 else 0
    cap = int(math.floor(acct * MAX_POSITION_PCT / entry / 100.0) * 100) if entry > 0 else 0
    shares = max(0, min(shares, cap))
    lim = _limit_pct(code, name, asset)
    extreme = {
        "stop_loss": shares * risk,
        "gap_3atr": shares * 3 * atr,
        "limit_down": shares * entry * lim,
        "limit_pct": lim,
    }
    base.update(shares=shares, position_value=shares * entry, risk_pct=risk_pct, risk_amount=risk_amt, extreme=extreme, account_size=acct)
    if market == "weak" and state in ("ok", "wait", "watch"):
        state, text = "skip", "市场弱势，停止新增个股仓位，只保留观察"
    elif shares < 100 and state in ("ok", "wait", "watch"):
        reasons.append(f"按账户风险 {risk_pct:.2f}% 反推不足 100 股，账户规模与止损距离不匹配")
        state, text = "skip", "仓位不足 100 股，跳过"

    inval = (f"收盘跌破 MA20（{_fmt(ma20, price)}）或跳空超过 0.5×ATR（{_fmt(0.5 * atr, price)}）时重算"
             if strategy == "pullback" else f"收盘跌破 {_fmt(stop, price)}（突破止损位）视为突破失败")
    base.update(state=state, state_text=text, reasons=reasons, invalidation=inval)
    return base


# ---------------------------------------------------------------------------
def apply_atr_risk_levels(result: Any, trend_result: Any) -> None:
    """用 V2 计划覆盖 dashboard.battle_plan.sniper_points，并写入 battle_plan.risk_plan。就地修改。"""
    if not result or trend_result is None:
        return
    try:
        tr = trend_result if isinstance(trend_result, dict) else getattr(trend_result, "__dict__", {})
        live_price = getattr(result, "current_price", None)
        try:
            live_price = float(live_price)
        except (TypeError, ValueError):
            live_price = 0.0
        plan = compute_risk_plan(
            price=live_price if live_price > 0 else float(tr.get("current_price") or 0),
            tr=tr, code=str(getattr(result, "code", "") or ""), name=str(getattr(result, "name", "") or ""),
        )
        if not plan:
            return
        if not getattr(result, "dashboard", None):
            result.dashboard = {}
        battle = result.dashboard.get("battle_plan")
        if not isinstance(battle, dict):
            battle = {}
            result.dashboard["battle_plan"] = battle
        sniper = battle.get("sniper_points")
        if not isinstance(sniper, dict):
            sniper = {}
            battle["sniper_points"] = sniper

        price = plan["price"]
        f = lambda v: _fmt(v, price)  # noqa: E731
        if plan["state"] == "na" or plan.get("entry") is None:
            sniper["ideal_buy"] = f"不适用（{plan['state_text']}）" if plan["state"] == "na" else f"暂无（{plan['state_text']}）"
            sniper["secondary_buy"] = "—"
            sniper["stop_loss"] = "—"
            sniper["take_profit"] = "—"
        else:
            lo, hi = plan["zone"]
            sniper["ideal_buy"] = f"{f(lo)}～{f(hi)}元（{plan['strategy_label']}入场区，距现价{_spct(lo, price)}～{_spct(hi, price)}）"
            sniper["secondary_buy"] = f"{f(plan['entry'])}元（计划入场价）"
            sniper["stop_loss"] = f"{f(plan['stop'])}元（距入场{_spct(plan['stop'], plan['entry'])}，距现价{_spct(plan['stop'], price)}）"
            sniper["take_profit"] = (f"{f(plan['target'])}元（{plan['target_label']}，距现价{_spct(plan['target'], price)}）"
                                     if plan.get("target") else "无结构压力，按「最高价−2×ATR」追踪止损退出")
        battle["risk_plan"] = plan
        logger.info("[risk_levels] %s v2 strategy=%s state=%s entry=%s stop=%s target=%s rr_plan=%s rr_now=%s shares=%s",
                    getattr(result, "code", "?"), plan["strategy"], plan["state"], plan.get("entry"), plan.get("stop"),
                    plan.get("target"), plan.get("rr_plan"), plan.get("rr_now"), plan.get("shares"))
    except Exception as e:  # noqa: BLE001
        logger.warning("[risk_levels] 计算失败，保留原点位: %s", e)


def render_risk_plan_markdown(plan: Dict[str, Any]) -> list:
    """渲染为 markdown 行（供通知使用，同时是看板的解析契约，措辞与键名请勿随意改）。"""
    if not plan or plan.get("version") != 2:
        return _render_v1(plan) if plan else []
    price = plan["price"]
    f = lambda v: _fmt(v, price)  # noqa: E731
    icon = STATE_ICON.get(plan["state"], "")
    lines = [f"**⚖️ 交易计划 V2**（Wilder ATR14={f(plan['atr14'])}，{plan['atr_pct']:.1f}%）", ""]
    lines.append(f"- **市场状态**: {plan['market_label']}")
    lines.append(f"- **策略类型**: {plan['strategy_label']}")
    lines.append(f"- **当前状态**: {icon} {plan['state_text']}")
    if plan["state"] != "na" and plan.get("entry") is not None:
        lo, hi = plan["zone"]
        lines.append(f"- **计划入场区**: {f(lo)}～{f(hi)}")
        lines.append(f"- **计划入场价**: {f(plan['entry'])}")
        lines.append(f"- **止损**: {f(plan['stop'])}（距入场{_spct(plan['stop'], plan['entry'])}，距现价{_spct(plan['stop'], price)}）")
        if plan.get("target"):
            lines.append(f"- **结构目标**: {f(plan['target'])}（{plan['target_label']}，距现价{_spct(plan['target'], price)}）")
            lines.append(f"- **计划盈亏比**: 1:{plan['rr_plan']:.2f}")
            rn = plan["rr_now"]
            lines.append(f"- **现价盈亏比**: 1:{rn:.2f}{'，不追' if rn < RR_OK_NOW else ''}")
        else:
            lines.append("- **结构目标**: 无（上方无结构压力，按「最高价−2×ATR」追踪止损退出）")
            lines.append("- **计划盈亏比**: —")
            lines.append("- **现价盈亏比**: —")
        ex = plan.get("extreme") or {}
        if plan["shares"] >= 100:
            lines.append(f"- **建议仓位**: {plan['shares']}股（约 {plan['position_value']:.0f}元，账户风险 {plan['risk_pct']:.2f}% ≈ {plan['risk_amount']:.0f}元）")
            lines.append(f"- **极端风险**: 止损 −{ex.get('stop_loss', 0):.0f}元 · 低开3×ATR −{ex.get('gap_3atr', 0):.0f}元 · 跌停{ex.get('limit_pct', 0.1) * 100:.0f}% −{ex.get('limit_down', 0):.0f}元")
        else:
            lines.append(f"- **建议仓位**: 0股（账户风险 {plan['risk_pct']:.2f}% ≈ {plan['risk_amount']:.0f}元 不足 100 股）")
        lines.append(f"- **失效条件**: {plan['invalidation']}")
    if plan.get("reasons"):
        lines.append(f"- **判定依据**: {'；'.join(plan['reasons'])}")
    lines.append("")
    return lines


def _render_v1(plan: Dict[str, Any]) -> list:
    """兼容旧版 plan dict（无 version 字段）。"""
    try:
        price = plan["price"]
        f = lambda v: _fmt(v, price)  # noqa: E731
        t1, t2, t3 = plan["targets"]
        icon = {"ok": "✅", "wait": "⏳", "weak": "⚠️", "skip": "❌"}.get(plan["verdict"], "")
        return [
            f"**⚖️ 风险收益测算**（ATR14={f(plan['atr14'])}，约占现价{plan['atr_pct']:.1f}%）", "",
            "| 点位 | 价格 | 距现价 |", "|------|------|--------|",
            f"| 🎯 理想买入 | {f(plan['entry'])} | {_spct(plan['entry'], price)} |",
            f"| 🛑 止损(−2×ATR) | {f(plan['stop'])} | {_spct(plan['stop'], price)} |",
            f"| 🎊 目标1(+1R) | {f(t1)} | {_spct(t1, price)} |",
            f"| 🎊 目标2(+2R) | {f(t2)} | {_spct(t2, price)} |",
            f"| 🎊 目标3(+3R) | {f(t3)} | {_spct(t3, price)} |", "",
            f"- **盈亏比·回踩入场**: 1:{plan['rr_entry']:.2f}",
            f"- **盈亏比·现价追入**: 1:{plan['rr_now']:.2f}",
            f"- **风控结论**: {icon} {plan['verdict_text']}", "",
        ]
    except Exception:  # noqa: BLE001
        return []
