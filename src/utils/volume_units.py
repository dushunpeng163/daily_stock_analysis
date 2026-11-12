# -*- coding: utf-8 -*-
"""成交量单位对齐。

不同数据源的成交量单位不同：东财系（efinance / akshare）日线是「手」，腾讯实时行情被归一成「股」。
把实时成交量直接除以历史日线成交量，会得到放大 100 倍的「较昨日 N 倍」，进而触发「量能异常、降权解读」。
这里用 成交额 ≈ 成交量 × 价格 × 每单位股数 反推各自的单位，再把实时成交量换算到历史口径。
"""

from __future__ import annotations

from typing import Any, Optional


def _pos(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def infer_volume_lot_size(volume: Any, amount: Any, price: Any) -> Optional[int]:
    """返回成交量单位对应的股数：1（股）或 100（手）；数据不足或对不上（如指数）时返回 None。"""
    v, a, p = _pos(volume), _pos(amount), _pos(price)
    if v is None or a is None or p is None:
        return None
    ratio = a / (v * p)   # 日内均价与收盘价的偏差远小于 2 倍，区间足够宽
    if 0.5 <= ratio <= 2:
        return 1
    if 50 <= ratio <= 200:
        return 100
    return None


def align_volume_to_reference(
    volume: Any, amount: Any, price: Any,
    ref_volume: Any, ref_amount: Any, ref_price: Any,
) -> Any:
    """把 volume 换算成与参考行（历史日线）相同的单位；任一方单位无法判断时原样返回。"""
    lot = infer_volume_lot_size(volume, amount, price)
    ref_lot = infer_volume_lot_size(ref_volume, ref_amount, ref_price)
    if lot is None or ref_lot is None or lot == ref_lot:
        return volume
    return float(volume) * lot / ref_lot
