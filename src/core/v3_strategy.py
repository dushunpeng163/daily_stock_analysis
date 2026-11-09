# -*- coding: utf-8 -*-
"""策略 V3（实验版）：趋势缩量回踩 + 突破后的首次缩量回踩。

本模块只根据传入 DataFrame 最后一根及以前的数据生成信号，不读取未来数据。
在样本外验证通过前，不接管线上 V2 信号。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.risk_levels import _limit_pct, classify_asset
from src.stock_analyzer import StockTrendAnalyzer


RISK_PCT = 0.35
MAX_POSITION_PCT = 0.10
RR_REQUIRED = 2.0
VOLUME_SHRINK_MAX = 0.80
BREAKOUT_VOLUME_MIN = 1.50
BREAKOUT_LOOKBACK_DAYS = 8


def _base_plan(code: str, name: str, price: float, market_state: str) -> Dict[str, Any]:
    return {
        "version": 3,
        "code": code,
        "name": name,
        "price": price,
        "market_state": market_state,
        "strategy": "none",
        "state": "watch",
        "state_text": "",
        "reasons": [],
        "zone": None,
        "entry": None,
        "stop": None,
        "target": None,
        "atr14": None,
        "rr_plan": None,
        "rr_now": None,
        "risk_pct": 0.0 if market_state == "weak" else RISK_PCT,
        "risk_amount": 0.0,
        "shares": 0,
        "position_value": 0.0,
    }


def _nearest_target(trend: Any, floor_price: float) -> Tuple[Optional[float], str]:
    candidates = []
    if float(getattr(trend, "high_20", 0) or 0) > floor_price:
        candidates.append((float(trend.high_20), "前20日高点"))
    if float(getattr(trend, "high_60", 0) or 0) > floor_price:
        candidates.append((float(trend.high_60), "前60日高点"))
    for value in getattr(trend, "swing_highs", None) or []:
        value = float(value)
        if value > floor_price:
            candidates.append((value, "120日内确认波段高点"))
    return min(candidates, key=lambda item: item[0]) if candidates else (None, "")


def _recent_breakout(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    atr: float,
) -> Optional[Dict[str, Any]]:
    n = len(close)
    current = n - 1
    # 当日突破不买；只寻找 1～8 个交易日前的放量突破。
    for age in range(1, BREAKOUT_LOOKBACK_DAYS + 1):
        i = current - age
        if i < 20:
            break
        level = float(high.iloc[i - 20:i].max())
        volume_avg = float(volume.iloc[i - 20:i].mean())
        breakout_volume = float(volume.iloc[i])
        if not (
            float(close.iloc[i]) > level
            and volume_avg > 0
            and breakout_volume >= BREAKOUT_VOLUME_MIN * volume_avg
        ):
            continue
        retest_ceiling = level + 0.35 * atr
        # 当前必须是突破后的第一次触碰；曾经触碰过就不追第二次、第三次回踩。
        had_earlier_retest = any(
            float(low.iloc[j]) <= retest_ceiling
            for j in range(i + 1, current)
        )
        return {
            "age": age,
            "level": level,
            "volume": breakout_volume,
            "had_earlier_retest": had_earlier_retest,
        }
    return None


def compute_v3_plan(
    df: pd.DataFrame,
    *,
    code: str,
    name: str = "",
    market_state: str = "neutral",
    account_size: float = 100000.0,
) -> Optional[Dict[str, Any]]:
    """用最后一根收盘数据生成 V3 计划；仅 ``state=ok`` 可进入下一日执行。"""
    if df is None or len(df) < 130:
        return None
    required = {"date", "open", "high", "low", "close", "volume"}
    if required.difference(df.columns):
        return None
    bars = df.sort_values("date").reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    if bars[list(required - {"date"})].tail(130).isna().any().any():
        return None

    price = float(bars["close"].iloc[-1])
    plan = _base_plan(code, name, price, market_state)
    asset = classify_asset(code, name)
    if asset not in ("STOCK", "STOCK_ETF"):
        plan.update(state="na", state_text="V3 仅用于个股与股票型 ETF")
        return plan
    if market_state == "weak":
        plan.update(state="skip", state_text="大盘弱势，停止新开仓", reasons=["市场过滤"])
        return plan

    trend = StockTrendAnalyzer().analyze(bars, code)
    atr = float(getattr(trend, "atr14", 0) or 0)
    ma5 = float(getattr(trend, "ma5", 0) or 0)
    ma10 = float(getattr(trend, "ma10", 0) or 0)
    ma20 = float(getattr(trend, "ma20", 0) or 0)
    ma20_prev5 = float(getattr(trend, "ma20_prev5", 0) or 0)
    if min(price, atr, ma5, ma10, ma20) <= 0:
        return None

    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    volume = bars["volume"]
    current_open = float(open_.iloc[-1])
    current_low = float(low.iloc[-1])
    current_volume = float(volume.iloc[-1])
    avg_volume_5 = float(volume.iloc[-6:-1].mean())
    bullish = price > current_open
    ma20_up = ma20_prev5 <= 0 or ma20 > ma20_prev5
    trend_ok = ma5 > ma10 > ma20 and ma20_up and price > ma20
    volume_shrunk = avg_volume_5 > 0 and current_volume <= VOLUME_SHRINK_MAX * avg_volume_5

    strategy = "none"
    zone: Optional[Tuple[float, float]] = None
    stop = 0.0
    reasons = []

    # 主策略：多头趋势中缩量回踩 MA10，收阳且守住 MA10。
    pullback_touched = current_low <= ma10 + 0.25 * atr
    pullback_not_extended = price <= ma5 + 0.50 * atr
    if trend_ok and pullback_touched and price >= ma10 and pullback_not_extended and bullish and volume_shrunk:
        strategy = "trend_pullback"
        zone = (ma10 - 0.15 * atr, ma5 + 0.25 * atr)
        low10 = float(low.iloc[-10:].min())
        stop = min(ma10 - 1.25 * atr, low10 - 0.25 * atr)
        reasons = ["MA5>MA10>MA20 且 MA20 向上", "首次执行信号为缩量回踩 MA10 后收阳"]
    else:
        breakout = _recent_breakout(close, high, low, volume, atr)
        if breakout:
            level = float(breakout["level"])
            breakout_trend_ok = ma5 > ma10 > ma20 and ma20_up and price > ma20
            first_retest = not breakout["had_earlier_retest"] and current_low <= level + 0.35 * atr
            held_level = price >= level
            shrunk_vs_breakout = current_volume <= VOLUME_SHRINK_MAX * float(breakout["volume"])
            if breakout_trend_ok and first_retest and held_level and bullish and shrunk_vs_breakout:
                strategy = "breakout_retest"
                zone = (level, level + 0.50 * atr)
                stop = min(level - 1.25 * atr, current_low - 0.25 * atr)
                reasons = [
                    f"突破后第 {breakout['age']} 日首次回踩",
                    "收盘守住突破位且收阳",
                    "回踩量不超过突破日 80%",
                    "MA20 向上且均线多头排列",
                ]

    if strategy == "none" or zone is None:
        plan.update(state="watch", state_text="尚未出现 V3 有效缩量回踩确认")
        return plan

    entry = price
    stop = max(0.0001, stop)
    risk = entry - stop
    if risk <= 0:
        plan.update(state="skip", state_text="止损计算异常")
        return plan
    stop_pct = risk / entry
    stop_min, stop_max = ((0.01, 0.05) if asset == "STOCK_ETF" else (0.02, 0.08))
    if not (stop_min <= stop_pct <= stop_max):
        plan.update(
            strategy=strategy,
            state="skip",
            state_text=f"止损距离 {stop_pct * 100:.1f}% 不在 {stop_min * 100:.0f}%～{stop_max * 100:.0f}% 范围",
            reasons=reasons,
        )
        return plan

    target_floor = entry + RR_REQUIRED * risk
    target, target_label = _nearest_target(trend, target_floor)
    if target is None:
        plan.update(
            strategy=strategy,
            state="watch",
            state_text="上方没有满足 2R 的已确认结构目标，仅观察",
            reasons=reasons,
        )
        return plan
    rr = (target - entry) / risk
    risk_amount = float(account_size) * RISK_PCT / 100.0
    risk_shares = math.floor(risk_amount / risk / 100.0) * 100
    cap_shares = math.floor(float(account_size) * MAX_POSITION_PCT / entry / 100.0) * 100
    shares = max(0, min(risk_shares, cap_shares))
    if shares < 100:
        plan.update(
            strategy=strategy,
            state="skip",
            state_text="风险仓位不足 100 股",
            reasons=reasons,
        )
        return plan

    plan.update(
        strategy=strategy,
        state="ok",
        state_text="V3 缩量回踩确认，下一交易日开盘仍在入场区才执行",
        reasons=reasons,
        zone=zone,
        entry=entry,
        stop=stop,
        target=target,
        target_label=target_label,
        atr14=atr,
        rr_plan=rr,
        rr_now=rr,
        risk_pct=RISK_PCT,
        risk_amount=risk_amount,
        shares=shares,
        position_value=shares * entry,
        limit_pct=_limit_pct(code, name, asset),
    )
    return plan


def v3_plan_provider(prefix: pd.DataFrame, code: str, name: str, market_state: str) -> Optional[Dict[str, Any]]:
    return compute_v3_plan(
        prefix,
        code=code,
        name=name,
        market_state=market_state,
    )


def make_v3_plan_provider(account_size: float):
    def provider(prefix: pd.DataFrame, code: str, name: str, market_state: str) -> Optional[Dict[str, Any]]:
        return compute_v3_plan(
            prefix,
            code=code,
            name=name,
            market_state=market_state,
            account_size=account_size,
        )

    return provider
