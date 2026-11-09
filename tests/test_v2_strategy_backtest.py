# -*- coding: utf-8 -*-
from datetime import date, timedelta

import pandas as pd
import pytest

from src.core.v2_strategy_backtest import (
    V2BacktestConfig,
    infer_historical_market_state,
    run_v2_symbol_backtest,
    select_v2_portfolio_trades,
    summarise_v2_backtest,
)
from src.risk_levels import _limit_pct


def _bars(rows):
    start = date(2026, 1, 1)
    return pd.DataFrame(
        [
            {
                "date": start + timedelta(days=i),
                "open": row[0],
                "high": row[1],
                "low": row[2],
                "close": row[3],
                "volume": 100000,
            }
            for i, row in enumerate(rows)
        ]
    )


def _config():
    return V2BacktestConfig(
        account_size=100000,
        commission_rate=0,
        min_commission=0,
        stamp_duty_rate=0,
        transfer_fee_rate=0,
        slippage_rate=0,
        max_holding_days=20,
        min_history_bars=2,
    )


def _provider(*, shares=100, target=12.0, stop=9.0):
    risk_amount = shares * (10.0 - stop)

    def build(prefix, code, name, market_state):
        if len(prefix) != 2:
            return {"state": "watch"}
        return {
            "state": "ok",
            "strategy": "breakout",
            "market_state": market_state,
            "zone": (9.5, 10.5),
            "stop": stop,
            "target": target,
            "atr14": 1.0,
            "risk_amount": risk_amount,
        }

    return build


def test_t1_stop_on_entry_day_exits_next_open() -> None:
    frame = _bars(
        [
            (10, 10.2, 9.8, 10),
            (10, 10.2, 9.8, 10),  # signal
            (10, 10.4, 8.5, 9.2),  # entry; stop touched but T+1 locked
            (8.0, 9.0, 7.8, 8.5),  # execute at next open
            (8.5, 8.8, 8.2, 8.6),
        ]
    )
    result = run_v2_symbol_backtest(
        frame,
        code="600001",
        config=_config(),
        plan_provider=_provider(shares=100),
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "t1_locked_stop"
    assert trade["entry_price"] == 10
    assert trade["average_exit_price"] == 8
    assert trade["net_pnl"] == -200


def test_one_lot_target_exits_all() -> None:
    frame = _bars(
        [
            (10, 10.2, 9.8, 10),
            (10, 10.2, 9.8, 10),
            (10, 10.5, 9.5, 10.2),
            (10.2, 12.2, 10.0, 12.0),
            (12.0, 12.3, 11.8, 12.1),
        ]
    )
    result = run_v2_symbol_backtest(
        frame,
        code="600001",
        config=_config(),
        plan_provider=_provider(shares=100),
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "target"
    assert trade["shares"] == 100
    assert trade["average_exit_price"] == 12
    assert trade["net_pnl"] == 200


def test_two_lots_take_half_then_trailing_stop() -> None:
    frame = _bars(
        [
            (10, 10.2, 9.8, 10),
            (10, 10.2, 9.8, 10),
            (10, 10.5, 9.5, 10.2),
            (10.2, 12.5, 9.5, 12.0),  # 100 shares sold at 12; next trailing stop = 10.5
            (11.0, 11.2, 10.0, 10.6),  # remaining 100 shares sold at 10.5
            (10.6, 10.8, 10.2, 10.4),
        ]
    )
    result = run_v2_symbol_backtest(
        frame,
        code="600001",
        config=_config(),
        plan_provider=_provider(shares=200),
    )
    trade = result["trades"][0]
    assert trade["exit_reason"] == "target_partial_then_stop"
    assert trade["shares"] == 200
    assert trade["average_exit_price"] == pytest.approx(11.25)
    assert trade["net_pnl"] == pytest.approx(250)


def test_locked_limit_up_skips_entry() -> None:
    frame = _bars(
        [
            (10, 10.2, 9.8, 10),
            (10, 10.2, 9.8, 10),
            (11, 11, 11, 11),  # one-price 10% limit-up
            (11, 11.2, 10.8, 11),
        ]
    )
    result = run_v2_symbol_backtest(
        frame,
        code="600001",
        config=_config(),
        plan_provider=_provider(shares=100),
    )
    assert result["trades"] == []
    assert result["skipped"]["limit_up_locked"] == 1


def test_st_stock_uses_five_percent_price_limit() -> None:
    assert _limit_pct("600001", "*ST测试", "STOCK") == 0.05
    assert _limit_pct("600001", "普通股票", "STOCK") == 0.10


def test_stock_etf_does_not_charge_stamp_duty() -> None:
    frame = _bars(
        [
            (10, 10.2, 9.8, 10),
            (10, 10.2, 9.8, 10),
            (10, 10.5, 9.5, 10.2),
            (10.2, 12.2, 10.0, 12.0),
            (12.0, 12.3, 11.8, 12.1),
        ]
    )
    cfg = _config()
    cfg = V2BacktestConfig(**{**cfg.__dict__, "stamp_duty_rate": 0.01})
    result = run_v2_symbol_backtest(
        frame,
        code="510300",
        name="沪深300ETF",
        config=cfg,
        plan_provider=_provider(shares=100),
    )
    assert result["trades"][0]["costs"] == 0


def test_market_state_uses_only_data_available_as_of_date() -> None:
    rows_up = [(10 + i * 0.1, 10 + i * 0.1, 10 + i * 0.1, 10 + i * 0.1) for i in range(30)]
    rows_down = [(20 - i * 0.1, 20 - i * 0.1, 20 - i * 0.1, 20 - i * 0.1) for i in range(30)]
    up = _bars(rows_up)
    down = _bars(rows_down)
    as_of = up["date"].iloc[-1]
    assert infer_historical_market_state([up, up])(as_of) == "strong"
    assert infer_historical_market_state([down, down])(as_of) == "weak"
    assert infer_historical_market_state([up, down])(as_of) == "neutral"


def test_summary_includes_cost_adjusted_validation() -> None:
    summary = summarise_v2_backtest(
        [
            {
                "trades": [
                    {"code": "A", "entry_date": "2026-01-01", "exit_date": "2026-01-02", "net_pnl": 100, "costs": 10, "exit_reason": "target"},
                    {"code": "B", "entry_date": "2026-01-03", "exit_date": "2026-01-04", "net_pnl": -50, "costs": 10, "exit_reason": "stop"},
                ]
            }
        ],
        100000,
    )
    assert summary["trade_count"] == 2
    assert summary["win_rate_pct"] == 50
    assert summary["expectancy_per_trade"] == 25
    assert summary["profit_factor"] == 2
    assert summary["candidate_trade_count"] == 2
    assert summary["rejected_by_max_positions"] == 0
    assert summary["validation"]["positive_expectancy"] is True
    assert summary["validation"]["sample_size_100"] is False


def test_portfolio_selection_enforces_three_concurrent_positions() -> None:
    results = [
        {
            "trades": [
                {
                    "code": code,
                    "entry_date": "2026-01-02",
                    "exit_date": "2026-01-10",
                    "net_pnl": 10,
                    "costs": 1,
                    "exit_reason": "target",
                }
            ]
        }
        for code in ("A", "B", "C", "D")
    ]
    accepted, rejected = select_v2_portfolio_trades(results, max_positions=3)
    assert [trade["code"] for trade in accepted] == ["A", "B", "C"]
    assert rejected == 1
    summary = summarise_v2_backtest(results, 100000, max_positions=3)
    assert summary["trade_count"] == 3
    assert summary["candidate_trade_count"] == 4
    assert summary["rejected_by_max_positions"] == 1
