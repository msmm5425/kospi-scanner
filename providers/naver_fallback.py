from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import KST, settings
from .base import is_eligible_common_stock

logger = logging.getLogger(__name__)

_NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://m.stock.naver.com/",
}


def _parse_num(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").strip()
    if not s or s.upper() == "N/A":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_naver_kospi_universe(
    min_cap: int | None = None,
    max_cap: int | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Naver 모바일 시가총액 API로 KOSPI 보통주 유니버스를 조회한다 (KRX 차단 시 안전 대체)."""
    lo_cap = settings.min_marcap_krw if min_cap is None else int(min_cap)
    hi_cap = settings.max_marcap_krw if max_cap is None else int(max_cap)
    sess = session or requests.Session()

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    page_size = 100

    for page in range(1, 32):
        url = f"https://m.stock.naver.com/api/stocks/marketValue/KOSPI?page={page}&pageSize={page_size}"
        try:
            resp = sess.get(url, headers=_NAVER_HEADERS, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("Naver marketValue HTTP %s (page=%s)", resp.status_code, page)
                break
            payload = resp.json() or {}
        except Exception as e:
            logger.warning("Naver marketValue fetch failed (page=%s): %s", page, e)
            break

        stocks = payload.get("stocks") or []
        if not stocks:
            break

        page_max_cap = 0.0
        for row in stocks:
            stock_end_type = str(row.get("stockEndType") or "").strip().lower()
            if stock_end_type and stock_end_type != "stock":
                continue
            code = str(row.get("itemCode") or row.get("reutersCode") or "").zfill(6)
            name = str(row.get("stockName") or "").strip()
            marcap = _parse_num(row.get("marketValueRaw"))
            if marcap <= 0:
                # marketValue(억원 단위 표기) 폴백
                mv_uk = _parse_num(row.get("marketValue"))
                if mv_uk > 0:
                    marcap = mv_uk * 1e8
            if marcap > page_max_cap:
                page_max_cap = marcap

            if code in seen or not is_eligible_common_stock(code, name):
                continue
            if marcap <= 0 or not (lo_cap <= marcap <= hi_cap):
                continue

            close_p = _parse_num(row.get("closePriceRaw") or row.get("closePrice"))
            if close_p <= 0:
                continue
            vol = _parse_num(row.get("accumulatedTradingVolumeRaw") or row.get("accumulatedTradingVolume"))
            tval = _parse_num(row.get("accumulatedTradingValueRaw"))
            if tval <= 0:
                tv_mil = _parse_num(row.get("accumulatedTradingValue"))
                if tv_mil > 0:
                    tval = tv_mil * 1_000_000.0
            if tval <= 0 and close_p > 0 and vol > 0:
                tval = close_p * vol

            chg = _parse_num(row.get("fluctuationsRatio"))
            seen.add(code)
            result.append(
                {
                    "code": code,
                    "name": name,
                    "marcap": int(marcap),
                    "trading_val": float(tval),
                    "price": float(close_p),
                    "chg_pct": round(chg, 2),
                }
            )

        # 시총 내림차순 페이지에서 최대 시총이 하한 미만으로 떨어지면 조기 종료
        if page_max_cap > 0 and page_max_cap < lo_cap:
            break
        if len(stocks) < page_size:
            break

    result.sort(key=lambda x: x["trading_val"], reverse=True)
    return result


def fetch_naver_top_gainers(
    top_n: int = 20,
    min_trading_val: int | None = None,
    marcap_map: dict[str, int] | None = None,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Naver 모바일 상승률 상위 API로 KOSPI 보통주 TOP N을 조회한다."""
    min_tval = settings.top20_min_trading_val_krw if min_trading_val is None else int(min_trading_val)
    sess = session or requests.Session()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for page in (1, 2):
        url = f"https://m.stock.naver.com/api/stocks/up/KOSPI?page={page}&pageSize=100"
        try:
            resp = sess.get(url, headers=_NAVER_HEADERS, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("Naver up/KOSPI HTTP %s (page=%s)", resp.status_code, page)
                break
            payload = resp.json() or {}
        except Exception as e:
            logger.warning("Naver up/KOSPI fetch failed (page=%s): %s", page, e)
            break

        stocks = payload.get("stocks") or []
        if not stocks:
            break

        for row in stocks:
            stock_end_type = str(row.get("stockEndType") or "").strip().lower()
            if stock_end_type and stock_end_type != "stock":
                continue
            code = str(row.get("itemCode") or row.get("reutersCode") or "").zfill(6)
            name = str(row.get("stockName") or "").strip()
            if code in seen or not is_eligible_common_stock(code, name):
                continue

            stop_info = row.get("tradeStopType") or {}
            stop_name = str(stop_info.get("name") or "").upper()
            if stop_name and stop_name != "TRADING":
                continue

            close_p = _parse_num(row.get("closePriceRaw") or row.get("closePrice"))
            vol = _parse_num(row.get("accumulatedTradingVolumeRaw") or row.get("accumulatedTradingVolume"))
            tval = _parse_num(row.get("accumulatedTradingValueRaw"))
            if tval <= 0:
                tv_mil = _parse_num(row.get("accumulatedTradingValue"))
                if tv_mil > 0:
                    tval = tv_mil * 1_000_000.0
            if tval <= 0 and close_p > 0 and vol > 0:
                tval = close_p * vol

            if close_p <= 0 or tval < min_tval:
                continue

            chg = _parse_num(row.get("fluctuationsRatio"))
            marcap = int(_parse_num(row.get("marketValueRaw")))
            if marcap <= 0 and marcap_map:
                marcap = int(marcap_map.get(code, 0))

            seen.add(code)
            out.append(
                {
                    "code": code,
                    "name": name,
                    "close": float(close_p),
                    "chg_pct": round(chg, 2),
                    "trading_val": float(tval),
                    "marcap": marcap,
                }
            )
        if len(out) >= max(1, int(top_n)):
            break

    out.sort(key=lambda x: x["chg_pct"], reverse=True)
    return out[: max(1, int(top_n))]


def parse_naver_fchart_xml(
    xml_text: str,
    bars: int = 150,
    end_date: str | None = None,
    is_index: bool = False,
) -> pd.DataFrame:
    """Naver fchart XML 응답을 표준 OHLCV 또는 지수 DataFrame으로 파싱."""
    cols = ["date", "close", "chg_pct"] if is_index else ["date", "open", "high", "low", "close", "volume", "amount"]
    if not xml_text or "<item" not in xml_text:
        return pd.DataFrame(columns=cols)

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame(columns=cols)

    eff_end = end_date[:10] if end_date else ""
    records: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        raw = str(item.attrib.get("data") or "")
        parts = raw.split("|")
        if len(parts) < 6:
            continue
        dt_raw = parts[0].strip()
        if len(dt_raw) != 8 or not dt_raw.isdigit():
            continue
        dt_iso = f"{dt_raw[:4]}-{dt_raw[4:6]}-{dt_raw[6:8]}"
        if eff_end and dt_iso > eff_end:
            continue

        o = _parse_num(parts[1])
        h = _parse_num(parts[2])
        l = _parse_num(parts[3])
        c = _parse_num(parts[4])
        v = _parse_num(parts[5])
        if c <= 0:
            continue
        if is_index:
            records.append({"date": dt_iso, "close": c})
        else:
            records.append(
                {
                    "date": dt_iso,
                    "open": o if o > 0 else c,
                    "high": h if h > 0 else c,
                    "low": l if l > 0 else c,
                    "close": c,
                    "volume": v,
                    "amount": c * v if v > 0 else 0.0,
                }
            )

    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if is_index:
        df["chg_pct"] = (df["close"].pct_change().fillna(0.0) * 100.0).round(2)
    return df.tail(bars).reset_index(drop=True)


def fetch_naver_fchart_ohlcv(
    symbol: str,
    bars: int = 150,
    end_date: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Naver fchart에서 개별 종목 수정 일봉을 조회한다."""
    sess = session or requests.Session()
    count = max(bars + 40, 200)
    if end_date:
        try:
            end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
            days_diff = max(0, (datetime.now(tz=KST).date() - end_dt).days)
            count = max(count, bars + days_diff + 40)
        except ValueError:
            pass
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol={symbol}&timeframe=day&count={count}&requestType=0"
    try:
        resp = sess.get(url, headers=_NAVER_HEADERS, timeout=10.0)
        if resp.status_code != 200:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
        text = resp.content.decode("euc-kr", errors="replace")
        return parse_naver_fchart_xml(text, bars=bars, end_date=end_date, is_index=False)
    except Exception as e:
        logger.debug("Naver fchart OHLCV (%s) failed: %s", symbol, e)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])


def fetch_naver_index_ohlcv(
    bars: int = 100,
    end_date: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Naver fchart에서 KOSPI 지수 일봉을 조회한다."""
    sess = session or requests.Session()
    count = max(bars + 40, 150)
    if end_date:
        try:
            end_dt = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
            days_diff = max(0, (datetime.now(tz=KST).date() - end_dt).days)
            count = max(count, bars + days_diff + 40)
        except ValueError:
            pass
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count={count}&requestType=0"
    try:
        resp = sess.get(url, headers=_NAVER_HEADERS, timeout=10.0)
        if resp.status_code != 200:
            return pd.DataFrame(columns=["date", "close", "chg_pct"])
        text = resp.content.decode("euc-kr", errors="replace")
        return parse_naver_fchart_xml(text, bars=bars, end_date=end_date, is_index=True)
    except Exception as e:
        logger.debug("Naver fchart KOSPI failed: %s", e)
        return pd.DataFrame(columns=["date", "close", "chg_pct"])


def parse_kind_sector_html(html_text: str) -> dict[str, str]:
    """KIND 상장법인목록 HTML 테이블에서 {종목코드: 업종} 매핑을 추출한다."""
    if not html_text:
        return {}
    soup = BeautifulSoup(html_text, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return {}

    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    code_idx = next((i for i, h in enumerate(headers) if "종목코드" in h), 2)
    sec_idx = next((i for i, h in enumerate(headers) if "업종" in h), 3)

    mapping: dict[str, str] = {}
    for tr in rows[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) <= max(code_idx, sec_idx):
            continue
        code = re.sub(r"\s+", "", cells[code_idx]).zfill(6)
        sec = cells[sec_idx].strip()
        if len(code) == 6 and code.isdigit() and sec and sec != "nan":
            mapping[code] = sec
    return mapping


def fetch_kind_sector_map(session: requests.Session | None = None) -> dict[str, str]:
    """KIND 상장법인목록에서 유가증권시장(KOSPI) 종목별 업종 매핑을 조회한다."""
    sess = session or requests.Session()
    url = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=stockMkt"
    try:
        resp = sess.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12.0)
        if resp.status_code != 200 or not resp.content:
            return {}
        html_text = resp.content.decode("euc-kr", errors="replace")
        return parse_kind_sector_html(html_text)
    except Exception as e:
        logger.debug("KIND sector map fetch failed: %s", e)
        return {}


def parse_naver_integration_supply(
    payload: dict[str, Any] | None,
    days: int = 5,
    end_date: str | None = None,
) -> dict[str, Any] | None:
    """Naver integration API의 dealTrendInfos에서 외인·기관 순매수(주)를 추출한다."""
    rows = (payload or {}).get("dealTrendInfos") or []
    if not rows:
        return None

    eff_end = (end_date or "").replace("-", "").strip()
    parsed: list[dict[str, Any]] = []
    for r in rows:
        dt = str(r.get("bizdate") or "").strip()
        if len(dt) != 8 or not dt.isdigit():
            continue
        if eff_end and dt > eff_end:
            continue
        frgn = int(round(_parse_num(r.get("foreignerPureBuyQuant"))))
        orgn = int(round(_parse_num(r.get("organPureBuyQuant"))))
        parsed.append({"date": dt, "frgn": frgn, "orgn": orgn})

    if not parsed:
        return None

    parsed.sort(key=lambda x: x["date"])
    recent = parsed[-max(days, 5) :]
    w3 = recent[-3:]
    w5 = recent[-days:]
    return {
        "frgn_today": recent[-1]["frgn"],
        "orgn_today": recent[-1]["orgn"],
        "frgn_3d": sum(x["frgn"] for x in w3),
        "orgn_3d": sum(x["orgn"] for x in w3),
        "total_5d": sum(x["frgn"] + x["orgn"] for x in w5),
        "pos_days_5d": sum(1 for x in w5 if (x["frgn"] + x["orgn"]) > 0),
        "intraday_pending": False,
        "orgn_series": [x["orgn"] for x in w5],
    }


def fetch_naver_investor_supply(
    code: str,
    days: int = 5,
    end_date: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """Naver 모바일 통합 API에서 최근 외인·기관 순매수를 조회한다."""
    norm_code = str(code).zfill(6)
    sess = session or requests.Session()
    url = f"https://m.stock.naver.com/api/stock/{norm_code}/integration"
    try:
        resp = sess.get(url, headers=_NAVER_HEADERS, timeout=8.0)
        if resp.status_code != 200:
            return None
        return parse_naver_integration_supply(resp.json(), days=days, end_date=end_date)
    except Exception as e:
        logger.debug("Naver integration supply (%s) failed: %s", norm_code, e)
        return None
