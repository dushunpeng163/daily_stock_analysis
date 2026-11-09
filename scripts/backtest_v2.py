#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实日线重建策略 V2 信号并执行 T+1 回测。"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.v2_strategy_backtest import (  # noqa: E402
    V2BacktestConfig,
    infer_historical_market_state,
    run_v2_symbol_backtest,
    summarise_v2_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("backtest_v2")


def _parse_codes(raw: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for token in re.split(r"[,;，；\n]+", raw or ""):
        token = token.strip()
        if not token:
            continue
        name = ""
        if "|" in token:
            token, name = [part.strip() for part in token.split("|", 1)]
        elif ":" in token and not token.lower().startswith(("sh:", "sz:", "bj:")):
            left, right = [part.strip() for part in token.split(":", 1)]
            if re.search(r"\d{6}", left):
                token, name = left, right
            elif re.search(r"\d{6}", right):
                token, name = right, left
        if re.search(r"\d{6}", token):
            items.append((token, name))
    dedup: Dict[str, str] = {}
    for code, name in items:
        dedup.setdefault(code, name)
    return list(dedup.items())


def _fetch(manager, code: str, start: date, end: date):
    df, source = manager.get_daily_data(
        code,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        days=max(30, (end - start).days),
    )
    logger.info("%s 日线完成: source=%s rows=%s", code, source, 0 if df is None else len(df))
    return df


def _render_markdown(payload: Dict) -> str:
    s = payload["summary"]
    lines = [
        "# 策略 V2 现实执行回测",
        "",
        f"- 区间：{payload['period']['start']} ～ {payload['period']['end']}",
        f"- 标的：{', '.join(payload['codes'])}",
        f"- 成交笔数：{s['trade_count']}",
        f"- 胜率：{s['win_rate_pct'] if s['win_rate_pct'] is not None else '—'}%",
        f"- 净收益：{s['net_pnl']:.2f} 元（{s['net_return_pct']}%）",
        f"- 单笔期望：{s['expectancy_per_trade'] if s['expectancy_per_trade'] is not None else '—'} 元",
        f"- 盈利因子：{s['profit_factor'] if s['profit_factor'] is not None else '—'}",
        f"- 已实现资金曲线最大回撤：{s['max_realized_drawdown_pct']}%",
        f"- 最大连续亏损：{s['max_consecutive_losses']} 笔",
        f"- 总交易成本：{s['total_costs']:.2f} 元",
        "",
        "## 验证结论",
        "",
        f"- 样本至少 100 笔：{'通过' if s['validation']['sample_size_100'] else '未通过'}",
        f"- 扣除成本后正期望：{'通过' if s['validation']['positive_expectancy'] else '未通过'}",
        f"- 胜率至少 40%：{'通过' if s['validation']['win_rate_at_least_40'] else '未通过'}",
        "",
        "## 执行假设",
        "",
        "- 信号只使用当日及以前数据，收盘后产生，下一交易日开盘才允许买入。",
        "- 次日开盘必须仍在计划入场区；跳空跌破止损或一字涨停均跳过。",
        "- A 股按 100 股一手、T+1；一字跌停时卖出顺延。",
        "- 普通股票按代码对应板块涨跌停；ST 需用“代码|ST名称”传入，否则按普通股票处理。",
        "- 日线同日同时碰到止损和目标时，保守按止损先发生。",
        "- 佣金、最低 5 元、过户费、股票卖出印花税和双边滑点均已扣除（ETF 不收印花税）。",
        "- 1 手到目标全部卖出；至少 2 手到目标卖一半，剩余按最高价减 2×入场 ATR 退出。",
        "",
        "## 逐笔交易",
        "",
    ]
    for result in payload["results"]:
        for trade in result["trades"]:
            lines.append(
                f"- {trade['code']} {trade['entry_date']}→{trade['exit_date']} "
                f"{trade['shares']}股｜{trade['exit_reason']}｜净盈亏 {trade['net_pnl']:+.2f}元 "
                f"({trade['return_pct']:+.2f}%)"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes",
        default=os.getenv("BACKTEST_CODES") or os.getenv("STOCK_LIST") or "",
        help="逗号分隔代码；可用 600519|贵州茅台 携带名称",
    )
    parser.add_argument("--start", required=True, help="信号开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="信号结束日期 YYYY-MM-DD")
    parser.add_argument("--account-size", type=float, default=float(os.getenv("ACCOUNT_SIZE") or 100000))
    parser.add_argument("--max-holding-days", type=int, default=40)
    parser.add_argument("--commission-rate", type=float, default=0.00025)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--stamp-duty-rate", type=float, default=0.0005)
    parser.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--output-dir", default="reports/v2_backtest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codes = _parse_codes(args.codes)
    if not codes:
        raise SystemExit("未提供有效代码：请设置 --codes 或 BACKTEST_CODES")
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start >= end:
        raise SystemExit("--start 必须早于 --end")

    # 延迟导入：--help 与纯逻辑测试不应要求安装全部行情源 SDK。
    from data_provider import DataFetcherManager

    cfg = V2BacktestConfig(
        account_size=args.account_size,
        commission_rate=args.commission_rate,
        min_commission=args.min_commission,
        stamp_duty_rate=args.stamp_duty_rate,
        transfer_fee_rate=args.transfer_fee_rate,
        slippage_rate=args.slippage_rate,
        max_holding_days=args.max_holding_days,
    )
    # 130 根预热需要约 200 个自然日；多取 260 天覆盖节假日和停牌。
    fetch_start = start - timedelta(days=260)
    manager = DataFetcherManager()
    benchmark_frames = []
    for index_code in ("sh000300", "sz399006"):
        try:
            benchmark_frames.append(_fetch(manager, index_code, fetch_start, end))
        except Exception as exc:  # noqa: BLE001
            logger.warning("基准指数 %s 获取失败，缺失部分按中性市场处理: %s", index_code, exc)
    market_provider = infer_historical_market_state(benchmark_frames)

    results = []
    errors = []
    for code, name in codes:
        try:
            frame = _fetch(manager, code, fetch_start, end)
            results.append(
                run_v2_symbol_backtest(
                    frame,
                    code=code,
                    name=name,
                    start_date=start,
                    end_date=end,
                    config=cfg,
                    market_state_provider=market_provider,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s 回测失败", code)
            errors.append({"code": code, "error": f"{type(exc).__name__}: {exc}"})

    summary = summarise_v2_backtest(results, cfg.account_size)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "codes": [code for code, _ in codes],
        "config": cfg.__dict__,
        "summary": summary,
        "results": results,
        "errors": errors,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"v2_backtest_{start.isoformat()}_{end.isoformat()}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告：{json_path}\n摘要：{md_path}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
