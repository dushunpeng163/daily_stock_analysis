"""
免费情报兜底：未配置任何搜索引擎（Tavily/Bocha/…）时，用 akshare 的公开接口
拉取 A 股个股新闻（东方财富）与公司公告（巨潮资讯），拼成一段 Markdown 情报，
喂给 LLM 作为 news_context。

- 只对 A 股个股 / ETF 生效；指数与美股直接返回 None。
- 这些接口是公开网页抓取，海外 IP（如 GitHub 托管 runner）经常被限流；
  在国内网络（自建 runner）下基本可用。失败只记 warning，不影响主流程。
- 受 FREE_INTEL_FALLBACK 环境变量控制，默认开启；设为 false 关闭。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NEWS_LIMIT = 8
ANN_LIMIT = 8
ANN_LOOKBACK_DAYS = 30
# 公告里最值得 LLM 注意的关键词，命中时在报告中加 ⚠️ 前缀
ANN_RISK_KEYWORDS = (
    "减持", "质押", "立案", "调查", "处罚", "警示", "问询", "关注函", "监管", "诉讼", "仲裁",
    "业绩预告", "业绩快报", "预亏", "亏损", "下修", "退市", "ST", "停牌", "解禁", "违规",
)


def is_enabled() -> bool:
    return os.getenv("FREE_INTEL_FALLBACK", "true").strip().lower() not in ("0", "false", "no", "off")


def _first(row: Any, keys: List[str]) -> str:
    for k in keys:
        try:
            v = row.get(k) if hasattr(row, "get") else None
        except Exception:  # noqa: BLE001
            v = None
        if v is None or (isinstance(v, float) and v != v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""


def _six(code: str) -> Optional[str]:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else None


def fetch_news(code: str, limit: int = NEWS_LIMIT) -> List[Dict[str, str]]:
    import akshare as ak  # 延迟导入，避免未安装时影响模块加载

    sym = _six(code)
    if not sym:
        return []
    df = ak.stock_news_em(symbol=sym)
    if df is None or df.empty:
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for _, row in df.iterrows():
        title = _first(row, ["新闻标题", "标题", "title"])
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({
            "title": title,
            "time": _first(row, ["发布时间", "时间", "date"]),
            "source": _first(row, ["文章来源", "来源", "source"]),
            "summary": _first(row, ["新闻内容", "内容", "content"])[:160],
        })
        if len(out) >= limit:
            break
    return out


def fetch_announcements(code: str, limit: int = ANN_LIMIT, lookback_days: int = ANN_LOOKBACK_DAYS) -> List[Dict[str, str]]:
    import akshare as ak

    sym = _six(code)
    if not sym:
        return []
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=sym, market="沪深京",
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return []
    out: List[Dict[str, str]] = []
    seen = set()
    for _, row in df.iterrows():
        title = _first(row, ["公告标题", "标题", "announcementTitle", "title"])
        if not title or title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "time": _first(row, ["公告时间", "公告日期", "date"])})
        if len(out) >= limit:
            break
    return out


def build_free_intel_context(code: str, stock_name: str, *, is_index: bool = False) -> Optional[Dict[str, Any]]:
    """返回 {"context": markdown, "count": n} 或 None（不适用 / 全部失败）。"""
    if not is_enabled() or is_index:
        return None
    sym = _six(code)
    if not sym:
        return None
    # 美股/港股代码不走这里
    if any(ch.isalpha() for ch in str(code).replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "").replace("bj", "").replace("BJ", "")):
        return None

    news: List[Dict[str, str]] = []
    anns: List[Dict[str, str]] = []
    errors: List[str] = []
    try:
        news = fetch_news(code)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"东财个股新闻: {exc}")
    try:
        anns = fetch_announcements(code)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"巨潮公告: {exc}")

    if errors:
        logger.warning("%s(%s) 免费情报兜底部分失败: %s", stock_name, code, "; ".join(errors))
    if not news and not anns:
        return None

    lines: List[str] = [f"## 📰 {stock_name}({code}) 公开情报（免费源兜底）", ""]
    lines.append("> 来源：东方财富个股新闻、巨潮资讯公告。未配置搜索引擎，以下为公开接口直接抓取，未经筛选，请自行判断重要性。")
    lines.append("")
    if anns:
        lines.append(f"### 📑 近 {ANN_LOOKBACK_DAYS} 日公司公告（{len(anns)} 条）")
        for a in anns:
            flag = "⚠️ " if any(k in a["title"] for k in ANN_RISK_KEYWORDS) else ""
            lines.append(f"- {flag}{a['time']} {a['title']}".rstrip())
        lines.append("")
    if news:
        lines.append(f"### 🗞️ 最新个股新闻（{len(news)} 条）")
        for n in news:
            head = " ".join(x for x in [n["time"], n["source"]] if x)
            lines.append(f"- {head}｜{n['title']}" if head else f"- {n['title']}")
            if n["summary"]:
                lines.append(f"  {n['summary']}")
        lines.append("")
    return {"context": "\n".join(lines).strip(), "count": len(news) + len(anns)}
