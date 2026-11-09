# -*- coding: utf-8 -*-
from datetime import date, timedelta

import pandas as pd

from src.core.v3_strategy import compute_v3_plan


def _uptrend() -> pd.DataFrame:
    rows = []
    for i in range(140):
        close = 10 + i * 0.03
        rows.append(
            {
                "date": date(2025, 1, 1) + timedelta(days=i),
                "open": close - 0.01,
                "high": close + 0.08,
                "low": close - 0.08,
                "close": close,
                "volume": 1000,
            }
        )
    frame = pd.DataFrame(rows)
    # 已确认的上方结构目标；位于当前日 39 根之前，不含未来数据。
    frame.loc[100, "high"] = 15.5
    return frame


def test_v3_trend_pullback_requires_shrink_volume_and_bullish_close() -> None:
    frame = _uptrend()
    frame.loc[139, ["open", "high", "low", "close", "volume"]] = [14.08, 14.20, 14.00, 14.15, 600]
    plan = compute_v3_plan(frame, code="600001", market_state="neutral")
    assert plan["state"] == "ok"
    assert plan["strategy"] == "trend_pullback"
    assert plan["rr_plan"] >= 2

    high_volume = frame.copy()
    high_volume.loc[139, "volume"] = 1200
    rejected = compute_v3_plan(high_volume, code="600001", market_state="neutral")
    assert rejected["state"] != "ok"

    bearish = frame.copy()
    bearish.loc[139, ["open", "close"]] = [14.18, 14.15]
    rejected = compute_v3_plan(bearish, code="600001", market_state="neutral")
    assert rejected["state"] != "ok"


def _breakout_retest_frame() -> tuple[pd.DataFrame, float]:
    frame = _uptrend()
    level = float(frame["high"].iloc[117:137].max())
    frame.loc[137, ["open", "high", "low", "close", "volume"]] = [
        level - 0.02,
        level + 0.30,
        level - 0.03,
        level + 0.20,
        2000,
    ]
    frame.loc[138, ["open", "high", "low", "close", "volume"]] = [
        level + 0.22,
        level + 0.28,
        level + 0.15,
        level + 0.20,
        1000,
    ]
    frame.loc[139, ["open", "high", "low", "close", "volume"]] = [
        level + 0.03,
        level + 0.15,
        level - 0.02,
        level + 0.10,
        1000,
    ]
    return frame, level


def test_v3_breakout_day_does_not_buy_and_first_retest_can_signal() -> None:
    frame, _ = _breakout_retest_frame()
    breakout_day = compute_v3_plan(frame.iloc[:138], code="600001", market_state="neutral")
    assert breakout_day["state"] != "ok"

    first_retest = compute_v3_plan(frame, code="600001", market_state="neutral")
    assert first_retest["state"] == "ok"
    assert first_retest["strategy"] == "breakout_retest"
    assert first_retest["rr_plan"] >= 2


def test_v3_rejects_second_breakout_retest() -> None:
    frame, level = _breakout_retest_frame()
    frame.loc[138, "low"] = level
    plan = compute_v3_plan(frame, code="600001", market_state="neutral")
    assert plan["state"] != "ok"


def test_v3_market_only_filters_entries_and_never_doubles_risk() -> None:
    frame = _uptrend()
    frame.loc[139, ["open", "high", "low", "close", "volume"]] = [14.08, 14.20, 14.00, 14.15, 600]
    neutral = compute_v3_plan(frame, code="600001", market_state="neutral")
    strong = compute_v3_plan(frame, code="600001", market_state="strong")
    weak = compute_v3_plan(frame, code="600001", market_state="weak")
    assert neutral["risk_pct"] == strong["risk_pct"] == 0.35
    assert neutral["risk_amount"] == strong["risk_amount"] == 350
    assert weak["state"] == "skip"
    assert weak["risk_pct"] == 0
