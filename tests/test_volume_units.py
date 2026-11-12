# -*- coding: utf-8 -*-
"""实时成交量（股）与历史日线成交量（手）的单位对齐。"""

from src.utils.volume_units import align_volume_to_reference, infer_volume_lot_size


def test_infer_lot_size():
    # 4,000,000 股 × 1400 元 ≈ 56 亿
    assert infer_volume_lot_size(4_000_000, 5.6e9, 1402.3) == 1
    # 同一天用「手」表示
    assert infer_volume_lot_size(40_000, 5.6e9, 1402.3) == 100
    # 指数：点位不是价格，对不上 → 不判断
    assert infer_volume_lot_size(3.2e8, 4.1e11, 4600.0) is None
    for bad in ((None, 1, 1), (0, 1, 1), (1, None, 1), (1, 1, 0), ("x", 1, 1)):
        assert infer_volume_lot_size(*bad) is None


def test_shares_realtime_aligned_to_lots_history():
    # 实时 4,000,000 股，昨日日线 36,000 手：真实倍数约 1.11，而不是 111
    aligned = align_volume_to_reference(4_000_000, 5.6e9, 1402.3, 36_000, 5.1e9, 1439.0)
    assert aligned == 40_000
    assert round(aligned / 36_000, 2) == 1.11


def test_same_unit_or_unknown_is_untouched():
    assert align_volume_to_reference(40_000, 5.6e9, 1402.3, 36_000, 5.1e9, 1439.0) == 40_000
    assert align_volume_to_reference(4_000_000, 5.6e9, 1402.3, 3_600_000, 5.1e9, 1439.0) == 4_000_000
    # 历史行缺成交额：无法判断，保持原值
    assert align_volume_to_reference(4_000_000, 5.6e9, 1402.3, 36_000, None, 1439.0) == 4_000_000
    assert align_volume_to_reference(None, 5.6e9, 1402.3, 36_000, 5.1e9, 1439.0) is None
