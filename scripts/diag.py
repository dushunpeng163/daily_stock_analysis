#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时诊断：GitHub runner 上各数据源的连通性 + 缩量回踩被哪条筛选条件挡掉。"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCREENING_ENABLED", "true")

import requests  # noqa: E402


def probe_http(name, url, **kw):
    t = time.time()
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, **kw)
        print(f"[HTTP] {name:28s} status={r.status_code} len={len(r.content)} {time.time()-t:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[HTTP] {name:28s} FAIL {type(exc).__name__}: {str(exc)[:90]} {time.time()-t:.1f}s", flush=True)


def probe_ak(name, fn, **kw):
    t = time.time()
    try:
        df = fn(**kw)
        shape = getattr(df, "shape", None)
        cols = list(getattr(df, "columns", []))[:8]
        print(f"[AK]   {name:34s} OK shape={shape} cols={cols} {time.time()-t:.1f}s", flush=True)
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[AK]   {name:34s} FAIL {type(exc).__name__}: {str(exc)[:100]} {time.time()-t:.1f}s", flush=True)
        return None


print("===== 1. 连通性 =====", flush=True)
probe_http("eastmoney push2his(K线)", "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600519&fields1=f1&fields2=f51&klt=101&fqt=0&end=20500101&lmt=2")
probe_http("eastmoney push2(行情)", "https://push2.eastmoney.com/api/qt/stock/get?secid=1.600519&fields=f43")
probe_http("eastmoney datacenter", "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_LICO_FN_CPD&columns=ALL&pageSize=1")
probe_http("sina 行情", "https://hq.sinajs.cn/list=sh600519", headers={"Referer": "https://finance.sina.com.cn"})
probe_http("sina 财务摘要", "https://vip.stock.finance.sina.com.cn/corp/go.php/vFD_FinanceSummary/stockid/600519.phtml")
probe_http("tencent 行情", "https://qt.gtimg.cn/q=sh600519")
probe_http("10jqka 资金流", "https://data.10jqka.com.cn/funds/ggzjl/")
probe_http("xueqiu", "https://stock.xueqiu.com/v5/stock/batch/quote.json?symbol=SH600519")

print("\n===== 2. akshare 资金流 / 财务接口 =====", flush=True)
import akshare as ak  # noqa: E402

probe_ak("stock_individual_fund_flow(东财)", ak.stock_individual_fund_flow, stock="600519", market="sh")
probe_ak("stock_fund_flow_individual(同花顺)", ak.stock_fund_flow_individual, symbol="即时")
probe_ak("stock_financial_abstract(新浪)", ak.stock_financial_abstract, symbol="600519")
probe_ak("stock_financial_abstract_ths(同花顺)", ak.stock_financial_abstract_ths, symbol="600519", indicator="按报告期")
probe_ak("stock_yjyg_em(东财业绩预告)", ak.stock_yjyg_em, date="20250930")
probe_ak("stock_zh_a_hist_tx(腾讯K线)", ak.stock_zh_a_hist_tx, symbol="sh600519", start_date="20260901", end_date="20261104")

print("\n===== 3. 缩量回踩：哪条筛选条件挡掉了所有人 =====", flush=True)
try:
    from src.services.screening.pipeline import screen

    res = screen("shrink_pullback", market="cn", max_output=30, use_llm=False, daily_enrich=True,
                 daily_enrich_max_candidates=120, explain_filters=True, config=None)
    print("snapshot", res.snapshot_count, "after_filter", res.after_filter_count, "picks", len(res.picks), "enriched", res.daily_enrich_count)
    for d in res.degradation:
        print("  *", d[:900])
except Exception as exc:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    print("screen failed:", exc)
