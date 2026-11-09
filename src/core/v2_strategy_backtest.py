# -*- coding: utf-8 -*-
"""策略 V2 的无未来函数日线回测。

信号只使用当日及以前的数据；信号在收盘后产生，最早于下一交易日开盘成交。
执行模型覆盖 A 股一手制、T+1、涨跌停无法成交、滑点、佣金和卖出印花税。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import math
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from src.risk_levels import _limit_pct, classify_asset, compute_risk_plan
from src.stock_analyzer import StockTrendAnalyzer


@dataclass(frozen=True)
class V2BacktestConfig:
    account_size: float = 100000.0
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.0005
    max_holding_days: int = 40
    min_history_bars: int = 130


@dataclass
class V2Trade:
    code: str
    name: str
    signal_date: date
    entry_date: date
    exit_date: date
    strategy: str
    market_state: str
    shares: int
    entry_price: float
    average_exit_price: float
    stop: float
    target: Optional[float]
    exit_reason: str
    holding_days: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float


PlanProvider = Callable[[pd.DataFrame, str, str, str], Optional[Dict[str, Any]]]
MarketStateProvider = Callable[[date], str]


def _normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"日线缺少字段: {', '.join(sorted(missing))}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _default_plan_provider(prefix: pd.DataFrame, code: str, name: str, market_state: str) -> Optional[Dict[str, Any]]:
    trend = StockTrendAnalyzer().analyze(prefix, code)
    return compute_risk_plan(
        float(prefix["close"].iloc[-1]),
        trend,
        code=code,
        name=name,
        market_state=market_state,
    )


def _commission(notional: float, cfg: V2BacktestConfig) -> float:
    return max(cfg.min_commission, notional * cfg.commission_rate)


def _locked_limit(bar: pd.Series, previous_close: float, limit_pct: float, direction: str) -> bool:
    """仅把一字涨/跌停视为不可成交；有价格区间时按可成交处理。"""
    if previous_close <= 0:
        return False
    prices = [float(bar[k]) for k in ("open", "high", "low", "close")]
    one_price = max(prices) - min(prices) <= max(0.001, previous_close * 0.0002)
    if not one_price:
        return False
    ratio = prices[0] / previous_close - 1.0
    tolerance = 0.003
    return ratio >= limit_pct - tolerance if direction == "up" else ratio <= -limit_pct + tolerance


def _round_lot_shares(risk_amount: float, entry: float, stop: float, account_size: float) -> int:
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return 0
    risk_shares = math.floor(risk_amount / risk_per_share / 100.0) * 100
    cap_shares = math.floor(account_size * 0.10 / entry / 100.0) * 100
    return max(0, min(risk_shares, cap_shares))


def run_v2_symbol_backtest(
    df: pd.DataFrame,
    *,
    code: str,
    name: str = "",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    config: Optional[V2BacktestConfig] = None,
    plan_provider: Optional[PlanProvider] = None,
    market_state_provider: Optional[MarketStateProvider] = None,
) -> Dict[str, Any]:
    """回测单只标的；同一标的同一时间最多持有一笔。"""
    cfg = config or V2BacktestConfig()
    bars = _normalise_bars(df)
    provider = plan_provider or _default_plan_provider
    market_provider = market_state_provider or (lambda _d: "neutral")
    trades: List[V2Trade] = []
    skipped = {
        "not_ok": 0,
        "next_open_outside_zone": 0,
        "next_open_below_stop": 0,
        "limit_up_locked": 0,
        "insufficient_lot": 0,
        "end_of_data": 0,
    }
    asset = classify_asset(code, name)
    limit_pct = _limit_pct(code, name, asset)
    stamp_rate = 0.0 if asset == "STOCK_ETF" else cfg.stamp_duty_rate
    transfer_rate = cfg.transfer_fee_rate if asset == "STOCK" else 0.0
    i = max(1, cfg.min_history_bars - 1)

    while i < len(bars) - 1:
        signal_bar = bars.iloc[i]
        signal_date = signal_bar["date"]
        if start_date and signal_date < start_date:
            i += 1
            continue
        if end_date and signal_date > end_date:
            break

        market_state = market_provider(signal_date)
        prefix = bars.iloc[: i + 1].copy()
        plan = provider(prefix, code, name, market_state)
        if not plan or plan.get("state") != "ok":
            skipped["not_ok"] += 1
            i += 1
            continue

        entry_index = i + 1
        entry_bar = bars.iloc[entry_index]
        previous_close = float(signal_bar["close"])
        if _locked_limit(entry_bar, previous_close, limit_pct, "up"):
            skipped["limit_up_locked"] += 1
            i += 1
            continue

        raw_open = float(entry_bar["open"])
        stop = float(plan.get("stop") or 0)
        zone = plan.get("zone") or (None, None)
        zone_lo, zone_hi = zone
        if raw_open <= stop:
            skipped["next_open_below_stop"] += 1
            i += 1
            continue
        if zone_lo is None or zone_hi is None or not (float(zone_lo) <= raw_open <= float(zone_hi)):
            skipped["next_open_outside_zone"] += 1
            i += 1
            continue

        entry_price = raw_open * (1.0 + cfg.slippage_rate)
        shares = _round_lot_shares(
            float(plan.get("risk_amount") or 0),
            entry_price,
            stop,
            cfg.account_size,
        )
        if shares < 100:
            skipped["insufficient_lot"] += 1
            i += 1
            continue

        target = float(plan["target"]) if plan.get("target") else None
        atr = float(plan.get("atr14") or 0)
        initial_shares = shares
        remaining = shares
        highest = float(entry_bar["high"])
        trailing_stop = stop
        buy_notional = entry_price * shares
        total_costs = _commission(buy_notional, cfg) + buy_notional * transfer_rate
        gross_proceeds = 0.0
        weighted_exit = 0.0
        sold_shares = 0
        exit_reason = "end_of_data"
        exit_date = entry_bar["date"]
        pending_reason: Optional[str] = None
        partial_taken = False

        # 买入当日受 T+1 约束；触发止损/目标只能下一交易日开盘处理。
        entry_stop_hit = float(entry_bar["low"]) <= stop
        entry_target_hit = target is not None and float(entry_bar["high"]) >= target
        if entry_stop_hit:
            pending_reason = "t1_locked_stop"
        elif entry_target_hit:
            pending_reason = "t1_locked_target"
        elif target is None and atr > 0:
            trailing_stop = max(stop, highest - 2.0 * atr)

        j = entry_index + 1
        while j < len(bars) and remaining > 0:
            bar = bars.iloc[j]
            prev_close = float(bars.iloc[j - 1]["close"])
            locked_down = _locked_limit(bar, prev_close, limit_pct, "down")
            raw_bar_open = float(bar["open"])
            raw_low = float(bar["low"])
            raw_high = float(bar["high"])

            def sell(qty: int, raw_price: float, reason: str) -> None:
                nonlocal remaining, gross_proceeds, total_costs, weighted_exit
                nonlocal sold_shares, exit_reason, exit_date
                qty = min(qty, remaining)
                if qty <= 0:
                    return
                closing_remainder = qty == remaining
                fill = max(0.0001, raw_price * (1.0 - cfg.slippage_rate))
                notional = fill * qty
                gross_proceeds += notional
                weighted_exit += fill * qty
                sold_shares += qty
                total_costs += _commission(notional, cfg) + notional * (stamp_rate + transfer_rate)
                remaining -= qty
                exit_reason = (
                    f"target_partial_then_{reason}"
                    if partial_taken and closing_remainder and not reason.endswith("_partial")
                    else reason
                )
                exit_date = bar["date"]

            if pending_reason:
                if locked_down:
                    j += 1
                    continue
                if "target" in pending_reason and initial_shares >= 200 and not partial_taken:
                    qty = max(100, math.floor(initial_shares / 2 / 100.0) * 100)
                    sell(qty, raw_bar_open, pending_reason)
                    partial_taken = True
                    pending_reason = None
                else:
                    sell(remaining, raw_bar_open, pending_reason)
                    pending_reason = None
                    if remaining == 0:
                        break

            active_stop = max(stop, trailing_stop) if partial_taken or target is None else stop
            if raw_bar_open <= active_stop:
                if locked_down:
                    pending_reason = "limit_down_deferred_stop"
                    j += 1
                    continue
                sell(remaining, raw_bar_open, "gap_stop")
                break

            if target is not None and not partial_taken and raw_bar_open >= target:
                if initial_shares < 200:
                    sell(remaining, raw_bar_open, "gap_target")
                    break
                qty = max(100, math.floor(initial_shares / 2 / 100.0) * 100)
                sell(qty, raw_bar_open, "gap_target_partial")
                partial_taken = True

            stop_hit = raw_low <= active_stop
            target_hit = target is not None and not partial_taken and raw_high >= target
            if stop_hit and target_hit:
                sell(remaining, active_stop, "ambiguous_stop_first")
                break
            if stop_hit:
                if locked_down:
                    pending_reason = "limit_down_deferred_stop"
                else:
                    sell(remaining, active_stop, "stop")
                    break
            elif target_hit:
                if initial_shares < 200:
                    sell(remaining, target, "target")
                    break
                qty = max(100, math.floor(initial_shares / 2 / 100.0) * 100)
                sell(qty, target, "target_partial")
                partial_taken = True

            highest = max(highest, raw_high)
            if atr > 0 and (partial_taken or target is None):
                # 当日最高价只用于下一交易日止损，避免日线内先高后低的未来函数。
                trailing_stop = max(stop, highest - 2.0 * atr)

            holding_days = j - entry_index + 1
            if holding_days >= cfg.max_holding_days and remaining > 0:
                if locked_down:
                    pending_reason = "limit_down_deferred_timeout"
                else:
                    sell(remaining, float(bar["close"]), "max_holding_days")
                    break
            j += 1

        if remaining > 0:
            last = bars.iloc[-1]
            fill = float(last["close"]) * (1.0 - cfg.slippage_rate)
            notional = fill * remaining
            gross_proceeds += notional
            weighted_exit += fill * remaining
            sold_shares += remaining
            total_costs += _commission(notional, cfg) + notional * (stamp_rate + transfer_rate)
            remaining = 0
            exit_date = last["date"]
            exit_reason = "end_of_data"
            skipped["end_of_data"] += 1

        gross_pnl = gross_proceeds - buy_notional
        net_pnl = gross_pnl - total_costs
        trade = V2Trade(
            code=code,
            name=name,
            signal_date=signal_date,
            entry_date=entry_bar["date"],
            exit_date=exit_date,
            strategy=str(plan.get("strategy") or ""),
            market_state=str(plan.get("market_state") or market_state),
            shares=initial_shares,
            entry_price=round(entry_price, 4),
            average_exit_price=round(weighted_exit / sold_shares, 4),
            stop=round(stop, 4),
            target=round(target, 4) if target is not None else None,
            exit_reason=exit_reason,
            holding_days=max(1, (bars["date"] <= exit_date).sum() - entry_index),
            gross_pnl=round(gross_pnl, 2),
            costs=round(total_costs, 2),
            net_pnl=round(net_pnl, 2),
            return_pct=round(net_pnl / (buy_notional + _commission(buy_notional, cfg)) * 100.0, 4),
        )
        trades.append(trade)
        # 同一标的持仓期间不重复开仓；从退出日下一根开始重新找信号。
        exit_matches = bars.index[bars["date"] == exit_date].tolist()
        i = (exit_matches[-1] + 1) if exit_matches else (j + 1)

    return {
        "code": code,
        "name": name,
        "config": asdict(cfg),
        "trades": [asdict(t) for t in trades],
        "skipped": skipped,
    }


def infer_historical_market_state(
    benchmark_frames: Iterable[pd.DataFrame],
) -> MarketStateProvider:
    """构造只使用目标日期及以前指数数据的市场状态函数。"""
    frames = [_normalise_bars(df)[["date", "close"]] for df in benchmark_frames if df is not None and not df.empty]

    def provider(as_of: date) -> str:
        up, total = 0, 0
        for frame in frames:
            hist = frame[frame["date"] <= as_of]
            if len(hist) < 25:
                continue
            close = hist["close"].astype(float)
            ma20 = float(close.iloc[-20:].mean())
            ma20_prev = float(close.iloc[-25:-5].mean())
            total += 1
            if float(close.iloc[-1]) > ma20 and ma20 >= ma20_prev * 0.998:
                up += 1
        if total == 0:
            return "neutral"
        if total >= 2 and up == total:
            return "strong"
        if up == 0:
            return "weak"
        return "neutral"

    return provider


def summarise_v2_backtest(results: Iterable[Dict[str, Any]], account_size: float) -> Dict[str, Any]:
    trades = [trade for result in results for trade in result.get("trades", [])]
    trades.sort(key=lambda t: (str(t["exit_date"]), str(t["entry_date"]), t["code"]))
    pnls = [float(t["net_pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    equity = float(account_size)
    peak = equity
    max_drawdown = 0.0
    loss_streak = max_loss_streak = 0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100.0)
        if pnl < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    total = len(trades)
    return {
        "trade_count": total,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": round(len(wins) / total * 100.0, 2) if total else None,
        "net_pnl": round(sum(pnls), 2),
        "net_return_pct": round(sum(pnls) / account_size * 100.0, 2) if account_size > 0 else None,
        "expectancy_per_trade": round(sum(pnls) / total, 2) if total else None,
        "average_win": round(gross_profit / len(wins), 2) if wins else None,
        "average_loss": round(gross_loss / len(losses), 2) if losses else None,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "max_realized_drawdown_pct": round(max_drawdown, 2),
        "max_consecutive_losses": max_loss_streak,
        "total_costs": round(sum(float(t["costs"]) for t in trades), 2),
        "exit_reasons": {
            reason: sum(1 for t in trades if t["exit_reason"] == reason)
            for reason in sorted({str(t["exit_reason"]) for t in trades})
        },
        "validation": {
            "sample_size_100": total >= 100,
            "positive_expectancy": bool(total and sum(pnls) > 0),
            "win_rate_at_least_40": bool(total and len(wins) / total >= 0.40),
        },
    }
