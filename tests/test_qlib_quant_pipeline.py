# -*- coding: utf-8 -*-
import io
import json
from pathlib import Path
import tarfile

import pandas as pd

from scripts.prepare_qlib_data import _find_cn_data, _safe_extract, _sha256
from scripts.qlib_quant_pipeline import _merge_portfolio_metrics, _publish, _top3
from src.quant.qlib_gate import evaluate_qlib_gate


def _metrics(**overrides):
    values = {
        "Rank IC": 0.035,
        "1day.excess_return_with_cost.annualized_return": 0.08,
        "1day.excess_return_with_cost.information_ratio": 0.9,
        "1day.excess_return_with_cost.max_drawdown": -0.07,
    }
    values.update(overrides)
    return values


def test_gate_requires_all_out_of_sample_checks() -> None:
    passed = evaluate_qlib_gate(_metrics(), test_days=300, trading_days=120)
    assert passed["passed"] is True
    failed = evaluate_qlib_gate(
        _metrics(**{"Rank IC": 0.005}),
        test_days=300,
        trading_days=120,
    )
    assert failed["passed"] is False
    assert failed["failed_checks"] == ["rank_ic"]


def test_gate_fails_closed_when_metrics_are_missing() -> None:
    result = evaluate_qlib_gate({}, test_days=0, trading_days=0)
    assert result["passed"] is False
    assert set(result["failed_checks"]) == {
        "rank_ic",
        "annualized_excess_return",
        "information_ratio",
        "max_drawdown",
        "test_days",
        "trading_days",
    }


def test_top3_uses_latest_prediction_date() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-11-05"), "SH600001"),
            (pd.Timestamp("2026-11-06"), "SH600001"),
            (pd.Timestamp("2026-11-06"), "SZ000001"),
            (pd.Timestamp("2026-11-06"), "SZ000002"),
            (pd.Timestamp("2026-11-06"), "SZ000003"),
        ],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [99, 0.2, 0.5, -0.1, 0.3]}, index=index)
    signal_date, rows = _top3(pred)
    assert signal_date == "2026-11-06"
    assert [row["instrument"] for row in rows] == ["SZ000001", "SZ000003", "SH600001"]


def test_portfolio_metrics_are_read_from_analysis_artifact() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("excess_return_with_cost", "annualized_return"),
            ("excess_return_with_cost", "information_ratio"),
            ("excess_return_with_cost", "max_drawdown"),
        ]
    )
    analysis = pd.DataFrame({"risk": [0.13, 0.75, -0.29]}, index=index)
    metrics = _merge_portfolio_metrics({"Rank IC": 0.04}, analysis)
    assert metrics["1day.excess_return_with_cost.annualized_return"] == 0.13
    assert metrics["1day.excess_return_with_cost.information_ratio"] == 0.75
    assert metrics["1day.excess_return_with_cost.max_drawdown"] == -0.29


def test_publish_never_marks_live_when_gate_failed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NTFY_URL", raising=False)
    monkeypatch.setenv("QLIB_LIVE_ENABLED", "1")
    args = type("Args", (), {"output_dir": tmp_path})()
    payload = _publish(
        {
            "signal_date": "2026-11-06",
            "gate": {"passed": False, "failed_checks": ["rank_ic"]},
            "top3": [{"rank": 1, "instrument": "SH600001", "score": 0.1}],
        },
        args,
    )
    assert payload["status"] == "simulation"
    saved = json.loads((tmp_path / "qlib_signal.json").read_text(encoding="utf-8"))
    assert saved["research_gate_passed"] is False


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../../escape")
        content = b"bad"
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    try:
        _safe_extract(archive, tmp_path / "out")
        assert False, "expected traversal rejection"
    except RuntimeError:
        pass


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("cn_data/features")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        tar.addfile(info)
    try:
        _safe_extract(archive, tmp_path / "out")
        assert False, "expected symlink rejection"
    except RuntimeError:
        pass


def test_find_cn_data_and_sha256(tmp_path: Path) -> None:
    data = tmp_path / "nested" / "cn_data"
    for child in ("calendars", "instruments", "features"):
        (data / child).mkdir(parents=True, exist_ok=True)
    assert _find_cn_data(tmp_path) == data
    payload = tmp_path / "payload"
    payload.write_bytes(b"abc")
    assert _sha256(payload) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
