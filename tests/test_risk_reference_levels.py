# -*- coding: utf-8 -*-
"""没有合格买点时的「参考价位」：只供查看，不能变成买入信号，也不能被下游当成真实买点入库。"""

from types import SimpleNamespace

from src.risk_levels import (
    REF_NOTE,
    apply_atr_risk_levels,
    compute_risk_plan,
    render_risk_plan_markdown,
)
from src.utils.sniper_points import SNIPER_KEYS, parse_sniper_value

BASE = dict(atr14=0.5, swing_low_10=17.2, high_20=19.4, high_60=20.8, swing_highs=[21.5],
            ma20_prev5=17.9, last_open=18.2, last_low=18.0)
# 趋势向上但离 MA20 超过 2×ATR
EXTENDED = (20.6, dict(BASE, ma5=20.2, ma10=19.6, ma20=18.9, swing_low_10=19.0, high_20=20.9, high_60=20.9,
                       swing_highs=[23.0], ma20_prev5=18.5))
# 空头排列、现价在 MA20 下方
BELOW_MA20 = (17.0, dict(BASE, ma5=17.3, ma10=17.8, ma20=18.4, swing_low_10=16.8, ma20_prev5=18.6))
# 现价在 MA20 上方但均线未多头排列
NOT_ALIGNED = (18.6, dict(BASE, ma5=18.3, ma10=18.5, ma20=18.1))
# 正常趋势回踩：有正式入场价
PULLBACK = (18.5, dict(BASE, ma5=18.5, ma10=18.3, ma20=18.0))


def _plan(case, market="neutral"):
    price, tr = case
    return compute_risk_plan(price, tr, code="600000", name="测试", account_size=100000, market_state=market)


def _lines(plan, key):
    return [ln for ln in render_risk_plan_markdown(plan) if f"**{key}**" in ln]


def test_extended_gives_pullback_zone_below_price_and_stays_skip():
    price, tr = EXTENDED
    plan = _plan(EXTENDED)
    assert plan["state"] == "skip" and plan.get("entry") is None
    ref = plan["ref"]
    lo, hi = ref["zone"]
    assert lo < hi < price
    assert hi <= tr["ma20"] + 2 * tr["atr14"] + 1e-9      # 回到 MA20 上方 2×ATR 以内才算不追高
    assert ref["stop"] < lo
    assert ref["resistance"] == 20.9 and ref["resistance"] > price


def test_below_ma20_gives_reclaim_level_and_resistance_above_ma20():
    price, tr = BELOW_MA20
    ref = _plan(BELOW_MA20)["ref"]
    assert ref["zone"] is None and ref["level"] == tr["ma20"]
    assert ref["stop"] < price
    # 压力不能和「站回 MA20」是同一个数
    assert ref["resistance"] > tr["ma20"]


def test_not_aligned_zone_never_below_ma20():
    _, tr = NOT_ALIGNED
    ref = _plan(NOT_ALIGNED)["ref"]
    lo, hi = ref["zone"]
    assert tr["ma20"] <= lo < hi


def test_real_plan_has_no_reference_block():
    plan = _plan(PULLBACK)
    assert plan.get("entry") is not None and not plan.get("ref")
    assert _lines(plan, "计划入场区") and not _lines(plan, "参考买入")


def test_stop_too_wide_still_reports_structural_target():
    """因止损过宽放弃的计划，上方有前高时不能渲染成「无结构压力 / 追踪止损」。"""
    tr = dict(atr14=4.69, ma5=109.26, ma10=107.9, ma20=104.0, ma20_prev5=102.0, swing_low_10=96.4,
              high_20=113.84, high_60=137.48, swing_highs=[], last_open=108.5, last_low=107.0)
    plan = compute_risk_plan(107.75, tr, code="688981", name="测试", account_size=100000, market_state="neutral")
    assert plan["state"] == "skip" and "止损距离过宽" in plan["state_text"]
    assert plan["target"] == 113.84 and plan["trailing"] is False
    line = _lines(plan, "结构目标")[0]
    assert "113.84" in line and "前20日高点" in line and "追踪止损" not in line


def test_markdown_contract_and_not_recommended_note():
    for case in (EXTENDED, BELOW_MA20, NOT_ALIGNED):
        plan = _plan(case)
        buy, stop, res = _lines(plan, "参考买入"), _lines(plan, "参考止损"), _lines(plan, "上方压力")
        assert len(buy) == 1 and len(stop) == 1 and len(res) == 1
        assert REF_NOTE in buy[0]
        # 参考位不能冒充正式计划：不输出入场区 / 止损 / 仓位行
        for key in ("计划入场区", "计划入场价", "止损", "建议仓位", "计划盈亏比"):
            assert not _lines(plan, key)


def test_reference_prices_never_leak_into_sniper_points(monkeypatch):
    """操作点位表会被 extract_sniper_points 解析成数值入库并生成决策信号，参考位的数字不能出现在里面。"""
    monkeypatch.setenv("MARKET_STATE", "neutral")   # 不走联网的市场状态自动判定
    for case in (EXTENDED, BELOW_MA20, NOT_ALIGNED):
        price, tr = case
        result = SimpleNamespace(code="600000", name="测试", current_price=price, dashboard={})
        apply_atr_risk_levels(result, tr)
        battle = result.dashboard["battle_plan"]
        assert battle["risk_plan"]["ref"]
        sniper = battle["sniper_points"]
        assert {k: parse_sniper_value(sniper.get(k)) for k in SNIPER_KEYS} == dict.fromkeys(SNIPER_KEYS)
