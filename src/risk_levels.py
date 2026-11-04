# -*- coding: utf-8 -*-
"""
基于 ATR 的确定性风控点位计算。

原则：价位由代码按波动率算出，而不是让 LLM 凭均线“编”数字。
  - 入场：回踩 MA5（价格已在 MA5 下方则取现价）
  - 止损：入场价 − 2×ATR14
  - 目标：入场价 + 1R / 2R / 3R（R = 入场价 − 止损价）
  - 盈亏比：同时给出「回踩入场」与「现价追入」两种，且目标会被上方真实压力位封顶
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ATR_STOP_MULT = 2.0


def _fmt(p: float, price_ref: float) -> str:
    return f"{p:.3f}" if price_ref < 20 else f"{p:.2f}"


def _pct(p: float, price: float) -> float:
    return (p / price - 1.0) * 100.0


def _spct(p: float, price: float) -> str:
    return f"{_pct(p, price):+.1f}%"


def compute_risk_plan(
    price: float,
    ma5: float,
    ma10: float,
    atr: float,
    high_20: float = 0.0,
    high_60: float = 0.0,
    ma20: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """返回风控计划字典；输入不足时返回 None。"""
    if not price or price <= 0 or not atr or atr <= 0:
        return None

    entry = min(price, ma5) if ma5 and ma5 > 0 else price
    if ma10 and 0 < ma10 < entry:
        secondary = ma10
    else:
        secondary = entry - atr
    secondary = max(secondary, 0.0001)

    stop = max(entry - ATR_STOP_MULT * atr, 0.0001)
    risk = entry - stop
    if risk <= 0:
        return None

    t1, t2, t3 = entry + risk, entry + 2 * risk, entry + 3 * risk

    # 上方最近的真实压力位（需高于现价至少 0.2%，否则视为已突破）
    resistance = None
    resistance_label = ""
    for val, label in ((high_20, "近20日高点"), (high_60, "近60日高点")):
        if val and val > price * 1.002 and (resistance is None or val < resistance):
            resistance, resistance_label = val, label

    cap = min(t2, resistance) if resistance else t2

    def _rr(target: float, base: float) -> float:
        denom = base - stop
        return (target - base) / denom if denom > 0 else 0.0

    rr_entry = _rr(cap, entry)
    rr_now = _rr(cap, price)

    bearish = bool(ma5 and ma10 and ma20 and ma5 < ma10 < ma20)

    if bearish:
        verdict, verdict_text = "skip", "均线空头排列，逆势入场风险高，等趋势转强再考虑"
    elif rr_now >= 2.0:
        verdict, verdict_text = "ok", "盈亏比合格，现价可按计划分批入场"
    elif rr_entry >= 2.0:
        verdict, verdict_text = "wait", f"现价追入盈亏比不足，等回踩 {_fmt(entry, price)} 附近再出手"
    elif max(rr_now, rr_entry) >= 1.2:
        verdict, verdict_text = "weak", "盈亏比一般，只适合小仓位试探"
    else:
        verdict, verdict_text = "skip", "上方空间不足以覆盖风险，不值得出手"

    return {
        "price": price,
        "atr14": atr,
        "atr_pct": atr / price * 100.0,
        "entry": entry,
        "secondary": secondary,
        "stop": stop,
        "targets": [t1, t2, t3],
        "risk_pct_of_entry": risk / entry * 100.0,
        "resistance": resistance,
        "resistance_label": resistance_label,
        "rr_entry": rr_entry,
        "rr_now": rr_now,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "bearish": bearish,
    }


def apply_atr_risk_levels(result: Any, trend_result: Any) -> None:
    """用 ATR 计划覆盖 dashboard.battle_plan.sniper_points，并写入 battle_plan.risk_plan。就地修改。"""
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
            ma5=float(tr.get("ma5") or 0),
            ma10=float(tr.get("ma10") or 0),
            ma20=float(tr.get("ma20") or 0),
            atr=float(tr.get("atr14") or 0),
            high_20=float(tr.get("high_20") or 0),
            high_60=float(tr.get("high_60") or 0),
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
        t1, t2, t3 = plan["targets"]
        sniper["ideal_buy"] = f"{f(plan['entry'])}元（回踩MA5/现价，距现价{_spct(plan['entry'], price)}）"
        sniper["secondary_buy"] = f"{f(plan['secondary'])}元（MA10或再回落1×ATR，距现价{_spct(plan['secondary'], price)}）"
        sniper["stop_loss"] = (
            f"{f(plan['stop'])}元（入场价−{ATR_STOP_MULT:g}×ATR14，距现价{_spct(plan['stop'], price)}，"
            f"单笔风险约{plan['risk_pct_of_entry']:.1f}%）"
        )
        sniper["take_profit"] = (
            f"{f(t2)}元（目标2=+2R，距现价{_spct(t2, price)}；目标1 {f(t1)} / 目标3 {f(t3)}）"
        )
        battle["risk_plan"] = plan
        logger.info(
            "[risk_levels] %s ATR14=%.3f entry=%.3f stop=%.3f rr_now=%.2f rr_entry=%.2f verdict=%s",
            getattr(result, "code", "?"), plan["atr14"], plan["entry"], plan["stop"],
            plan["rr_now"], plan["rr_entry"], plan["verdict"],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[risk_levels] 计算失败，保留原点位: %s", e)


def render_risk_plan_markdown(plan: Dict[str, Any]) -> list:
    """渲染为 markdown 行（供通知使用，同时是看板的解析契约，措辞请勿随意改）。"""
    price = plan["price"]
    f = lambda v: _fmt(v, price)  # noqa: E731
    t1, t2, t3 = plan["targets"]
    icon = {"ok": "✅", "wait": "⏳", "weak": "⚠️", "skip": "❌"}.get(plan["verdict"], "")
    lines = [
        f"**⚖️ 风险收益测算**（ATR14={f(plan['atr14'])}，约占现价{plan['atr_pct']:.1f}%）",
        "",
        "| 点位 | 价格 | 距现价 |",
        "|------|------|--------|",
        f"| 🎯 理想买入 | {f(plan['entry'])} | {_spct(plan['entry'], price)} |",
        f"| 🛑 止损(−2×ATR) | {f(plan['stop'])} | {_spct(plan['stop'], price)} |",
        f"| 🎊 目标1(+1R) | {f(t1)} | {_spct(t1, price)} |",
        f"| 🎊 目标2(+2R) | {f(t2)} | {_spct(t2, price)} |",
        f"| 🎊 目标3(+3R) | {f(t3)} | {_spct(t3, price)} |",
        "",
        f"- **盈亏比·回踩入场**: 1:{plan['rr_entry']:.2f}",
        f"- **盈亏比·现价追入**: 1:{plan['rr_now']:.2f}",
    ]
    if plan.get("resistance"):
        lines.append(
            f"- **上方压力**: {f(plan['resistance'])}（{plan['resistance_label']}，距现价{_spct(plan['resistance'], price)}），目标已按此封顶"
        )
    lines.append(f"- **风控结论**: {icon} {plan['verdict_text']}")
    lines.append("")
    return lines
