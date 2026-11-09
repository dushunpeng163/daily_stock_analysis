#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qlib Alpha158 + LightGBM 训练、样本外门禁和每日 Top3 信号。"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quant.qlib_gate import evaluate_qlib_gate  # noqa: E402


EXPERIMENT = "alpha158-lgb-top3"
LABEL_5D = (["Ref($close, -6) / Ref($close, -1) - 1"], ["LABEL0"])


def _init_qlib(data_dir: Path, mlruns_dir: Path) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(
        provider_uri=str(data_dir.resolve()),
        region=REG_CN,
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": mlruns_dir.resolve().as_uri(),
                "default_exp_name": EXPERIMENT,
            },
        },
    )


def _model():
    from qlib.contrib.model.gbdt import LGBModel

    return LGBModel(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.05,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=2,
        seed=42,
        feature_fraction_seed=42,
        bagging_seed=42,
        data_random_seed=42,
        verbosity=-1,
    )


def _dataset(test_end: str):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH

    handler = Alpha158(
        instruments="csi300",
        start_time="2014-01-01",
        end_time=test_end,
        fit_start_time="2014-01-01",
        fit_end_time="2022-12-31",
        label=LABEL_5D,
    )
    return DatasetH(
        handler=handler,
        segments={
            "train": ("2014-01-01", "2022-12-31"),
            "valid": ("2023-01-01", "2024-12-31"),
            "test": ("2025-01-01", test_end),
        },
    )


def _portfolio_config(test_end: str, account_size: float) -> Dict[str, Any]:
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 3,
                "n_drop": 1,
                "hold_thresh": 5,
                "risk_degree": 0.30,
                "only_tradable": True,
                "forbid_all_trade_at_limit": True,
            },
        },
        "backtest": {
            "start_time": "2025-01-01",
            "end_time": test_end,
            "account": float(account_size),
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "open",
                # 佣金/滑点/过户费合并进开仓，平仓另含印花税。
                "open_cost": 0.00076,
                "close_cost": 0.00126,
                "min_cost": 5,
            },
        },
    }


def _top3(pred: pd.DataFrame) -> tuple[str, list[Dict[str, Any]]]:
    if pred is None or pred.empty:
        raise RuntimeError("Qlib pred.pkl 为空")
    latest = pd.Timestamp(pred.index.get_level_values("datetime").max())
    daily = pred.xs(latest, level="datetime")
    score_col = "score" if "score" in daily.columns else daily.columns[0]
    ranked = daily.nlargest(3, score_col)
    rows = [
        {"instrument": str(instrument).upper(), "score": round(float(row[score_col]), 8), "rank": rank}
        for rank, (instrument, row) in enumerate(ranked.iterrows(), start=1)
    ]
    return latest.date().isoformat(), rows


def train(args: argparse.Namespace) -> Dict[str, Any]:
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

    dataset = _dataset(args.data_date)
    model = _model()
    with R.start(experiment_name=EXPERIMENT, recorder_name=f"fixed-split-{args.data_date}"):
        recorder = R.get_recorder()
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})
        dataset.config(dump_all=False, recursive=True)
        R.save_objects(**{"dataset": dataset})
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder, ana_long_short=False).generate()
        PortAnaRecord(
            recorder,
            _portfolio_config(args.data_date, args.account_size),
            risk_analysis_freq="day",
            indicator_analysis_freq="day",
        ).generate()
        metrics = dict(recorder.list_metrics())
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        trading_days = int((report["turnover"].abs() > 0).sum()) if "turnover" in report else 0
        gate = evaluate_qlib_gate(metrics, test_days=len(report), trading_days=trading_days)
        pred = recorder.load_object("pred.pkl")
        signal_date, top3 = _top3(pred)
        state = {
            "version": 1,
            "experiment": EXPERIMENT,
            "recorder_id": recorder.id,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "data_date": args.data_date,
            "splits": {
                "train": ["2014-01-01", "2022-12-31"],
                "valid": ["2023-01-01", "2024-12-31"],
                "test": ["2025-01-01", args.data_date],
            },
            "metrics": metrics,
            "gate": gate,
            "signal_date": signal_date,
            "top3": top3,
        }
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return state


def daily(args: argparse.Namespace) -> Dict[str, Any]:
    from qlib.workflow import R
    from qlib.workflow.online.update import PredUpdater

    state_path = args.state_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    recorder = R.get_recorder(
        experiment_name=state["experiment"],
        recorder_id=state["recorder_id"],
    )
    PredUpdater(recorder, to_date=args.data_date, trusted=True).update()
    pred = recorder.load_object("pred.pkl")
    signal_date, top3 = _top3(pred)
    state.update(
        {
            "last_predicted_at": datetime.now().isoformat(timespec="seconds"),
            "data_date": args.data_date,
            "signal_date": signal_date,
            "top3": top3,
        }
    )
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return state


def _publish(state: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    research_passed = bool((state.get("gate") or {}).get("passed"))
    live_enabled = os.getenv("QLIB_LIVE_ENABLED", "0").strip() == "1"
    status = "validated_live" if research_passed and live_enabled else "simulation"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "Qlib Alpha158 + LightGBM",
        "label": "未来5交易日收益",
        "universe": "CSI300历史成分",
        "signal_date": state.get("signal_date"),
        "status": status,
        "research_gate_passed": research_passed,
        "live_enabled": live_enabled,
        "gate": state.get("gate"),
        "top3": state.get("top3") or [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qlib_signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        f"## 🤖 Qlib Top3 · {payload['signal_date']}",
        "",
        f"状态：{'✅ 已验证实盘候选' if status == 'validated_live' else '🧪 模拟观察（禁止作为买入建议）'}",
        f"样本外门禁：{'通过' if research_passed else '未通过'}",
        "",
    ]
    for item in payload["top3"]:
        lines.append(f"{item['rank']}. **{item['instrument']}** · score {item['score']:+.6f}")
    if not research_passed:
        failed = "、".join((payload.get("gate") or {}).get("failed_checks") or [])
        lines.extend(["", f"未通过项：{failed or '暂无有效验证结果'}"])
    message = "\n".join(lines)
    (args.output_dir / "qlib_signal.md").write_text(message + "\n", encoding="utf-8")
    ntfy_url = (os.getenv("NTFY_URL") or "").strip()
    if ntfy_url:
        response = requests.post(
            ntfy_url,
            data=message.encode("utf-8"),
            headers={
                "Title": "Qlib Top3 模型信号",
                "Tags": "robot,chart_with_upwards_trend",
                **({"Authorization": f"Bearer {os.getenv('NTFY_TOKEN')}"} if os.getenv("NTFY_TOKEN") else {}),
            },
            timeout=15,
        )
        response.raise_for_status()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "train", "daily"), default="auto")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-date", required=True)
    parser.add_argument("--state-dir", type=Path, default=Path(".qlib_model"))
    parser.add_argument("--mlruns-dir", type=Path, default=Path(".qlib_mlruns"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qlib"))
    parser.add_argument("--account-size", type=float, default=float(os.getenv("ACCOUNT_SIZE") or 100000))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if date.fromisoformat(args.data_date) < date(2025, 12, 31):
        raise SystemExit("Qlib 数据过旧，无法覆盖完整样本外区间")
    _init_qlib(args.data_dir, args.mlruns_dir)
    state_exists = (args.state_dir / "state.json").exists()
    mode = ("daily" if state_exists else "train") if args.mode == "auto" else args.mode
    state = train(args) if mode == "train" else daily(args)
    payload = _publish(state, args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
