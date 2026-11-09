# -*- coding: utf-8 -*-
"""Qlib 模型样本外上线门禁（纯逻辑，不依赖 Qlib）。"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


DEFAULT_THRESHOLDS = {
    "rank_ic": 0.02,
    "information_ratio": 0.50,
    "annualized_excess_return": 0.0,
    "max_drawdown": 0.10,
    "test_days": 252,
    "trading_days": 100,
}


def _number(metrics: Mapping[str, Any], suffix: str) -> Optional[float]:
    for key, value in metrics.items():
        if str(key).lower().replace(" ", "_").endswith(suffix.lower().replace(" ", "_")):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def evaluate_qlib_gate(
    metrics: Mapping[str, Any],
    *,
    test_days: int,
    trading_days: int,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    rank_ic = _number(metrics, "rank_ic")
    annualized_excess = _number(metrics, "excess_return_with_cost.annualized_return")
    information_ratio = _number(metrics, "excess_return_with_cost.information_ratio")
    max_drawdown = _number(metrics, "excess_return_with_cost.max_drawdown")
    values = {
        "rank_ic": rank_ic,
        "annualized_excess_return": annualized_excess,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
        "test_days": int(test_days),
        "trading_days": int(trading_days),
    }
    checks = {
        "rank_ic": rank_ic is not None and rank_ic >= limits["rank_ic"],
        "annualized_excess_return": annualized_excess is not None
        and annualized_excess > limits["annualized_excess_return"],
        "information_ratio": information_ratio is not None
        and information_ratio >= limits["information_ratio"],
        "max_drawdown": max_drawdown is not None and abs(max_drawdown) <= limits["max_drawdown"],
        "test_days": test_days >= limits["test_days"],
        "trading_days": trading_days >= limits["trading_days"],
    }
    return {
        "passed": all(checks.values()),
        "values": values,
        "thresholds": limits,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
