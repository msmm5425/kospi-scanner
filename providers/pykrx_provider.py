from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from config.settings import KST, settings
from .base import MarketDataProvider, is_eligible_common_stock
from .naver_fallback import (
    fetch_kind_sector_map,
    fetch_naver_fchart_ohlcv,
    fetch_naver_index_ohlcv,
    fetch_naver_kospi_universe,
    fetch_naver_top_gainers,
)

try:
    from pykrx import stock as pykrx_stock
except ImportError:  # pragma: no cover
    pykrx_stock = None

try:
    import FinanceDataReader as fdr
except ImportError:  # pragma: no cover
    fdr = None

logger = logging.getLogger(__name__)


def _to_yyyymmdd(date_str: str) -> str:
    return str(date_str).replace("-", "").strip()


def _to_iso_date(val: Any) -> str:
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """pykrx / FDR 일봉 DataFrame을 표준 컬럼 ['date','open','high','low','close','volume','amount']으로 정규화."""
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    if "date" not in work.columns:
        work = work.reset_index()

    rename_map = {
        "날짜": "date",
        "Date": "date",
        "index": "date",
        "시가": "open",
        "Open": "open",
        "고가": "high",
        "High": "high",
        "저가": "low",
        "Low": "low",
        "종가": "close",
        "Close": "close",
        "거래량": "volume",
        "Volume": "volume",
        "거래대금": "amount",
        "Amount": "amount",
    }
    work = work.rename(columns={k: v for k, v in rename_map.items() if k in work.columns})

    if "date" not in work.columns or "close" not in work.columns:
        return pd.DataFrame(columns=cols)

    work["date"] = work["date"].map(_to_iso_date)
    for num_col in ("open", "high", "low", "close", "volume", "amount"):
        if num_col not in work.columns:
            work[num_col] = 0.0
        work[num_col] = pd.to_numeric(work[num_col], errors="coerce").fillna(0.0)

    # amount가 없는 FDR 데이터 등은 종가 * 거래량으로 근사 보완
    missing_amt = (work["amount"] <= 0) & (work["close"] > 0) & (work["volume"] > 0)
    work.loc[missing_amt, "amount"] = work.loc[missing_amt, "close"] * work.loc[missing_amt, "volume"]

    work = work[work["close"] > 0].copy()
    work = work.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return work[cols]


class PyKRXProvider(MarketDataProvider):
    """[Phase 1 개발·검증용] pykrx + FinanceDataReader 무료 코스피 데이터 프로바이더."""

    provider_name = "pykrx"

    def __init__(self, as_of_date: str | None = None, request_delay_sec: float = 0.04):
        self.as_of_date = as_of_date
        self.request_delay_sec = max(0.0, float(request_delay_sec))
        self._lock = threading.RLock()
        self._resolved_date_cache: str | None = None
        self._snapshot_cache: dict[str, pd.DataFrame] = {}
        self._ticker_name_cache: dict[str, dict[str, str]] = {}
        self._sector_map_cache: dict[str, str] | None = None
        self._risk_codes_cache: tuple[float, dict[str, str]] | None = None
        self._ohlcv_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._index_cache: dict[str, pd.DataFrame] = {}

    def _sleep(self) -> None:
        if self.request_delay_sec > 0:
            time.sleep(self.request_delay_sec)

    def _effective_end_date(self, end_date: str | None = None) -> str:
        if end_date:
            return self._resolve_latest_trading_day(end_date)
        if self.as_of_date:
            return self._resolve_latest_trading_day(self.as_of_date)
        with self._lock:
            if self._resolved_date_cache:
                return self._resolved_date_cache
            today_str = datetime.now(tz=KST).strftime("%Y-%m-%d")
            resolved = self._resolve_latest_trading_day(today_str)
            self._resolved_date_cache = resolved
            return resolved

    def _resolve_latest_trading_day(self, anchor_date: str) -> str:
        """주말·공휴일·장 시작 전 실행 시 직전 유효 KRX 거래일을 안전하게 탐색."""
        anchor_dt = datetime.strptime(_to_iso_date(anchor_date), "%Y-%m-%d").date()
        start_dt = anchor_dt - timedelta(days=20)
        if pykrx_stock is not None:
            try:
                idx_df = pykrx_stock.get_index_ohlcv_by_date(
                    start_dt.strftime("%Y%m%d"),
                    anchor_dt.strftime("%Y%m%d"),
                    "1001",  # KOSPI 지수
                )
                if idx_df is not None and not idx_df.empty:
                    # 장 시작 직후(09:00 이전) 전부 0인 당일 행 제외
                    close_col = "종가" if "종가" in idx_df.columns else idx_df.columns[3]
                    valid = idx_df[pd.to_numeric(idx_df[close_col], errors="coerce").fillna(0) > 0]
                    if not valid.empty:
                        return _to_iso_date(valid.index[-1])
            except Exception as e:
                logger.debug("pykrx index trading-day lookup failed: %s", e)

            # 백업: 삼성전자(005930) 일봉으로 확인
            try:
                s_df = pykrx_stock.get_market_ohlcv_by_date(
                    start_dt.strftime("%Y%m%d"),
                    anchor_dt.strftime("%Y%m%d"),
                    "005930",
                )
                if s_df is not None and not s_df.empty:
                    c_col = "종가" if "종가" in s_df.columns else "Close"
                    valid = s_df[pd.to_numeric(s_df[c_col], errors="coerce").fillna(0) > 0]
                    if not valid.empty:
                        return _to_iso_date(valid.index[-1])
            except Exception as e:
                logger.debug("pykrx 005930 trading-day lookup failed: %s", e)

        if fdr is not None:
            try:
                ks11 = fdr.DataReader("KS11", start_dt.strftime("%Y-%m-%d"), anchor_dt.strftime("%Y-%m-%d"))
                if ks11 is not None and not ks11.empty:
                    return _to_iso_date(ks11.index[-1])
            except Exception as e:
                logger.debug("FDR KS11 trading-day lookup failed: %s", e)

        try:
            naver_idx = fetch_naver_index_ohlcv(bars=15, end_date=anchor_dt.strftime("%Y-%m-%d"))
            if not naver_idx.empty:
                return str(naver_idx["date"].iloc[-1])
        except Exception as e:
            logger.debug("Naver fchart KS11 trading-day lookup failed: %s", e)

        # 최후 폴백: 평일 보정
        d = anchor_dt
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d.strftime("%Y-%m-%d")

    def _get_ticker_names(self, date_iso: str) -> dict[str, str]:
        with self._lock:
            if date_iso in self._ticker_name_cache:
                return self._ticker_name_cache[date_iso]

        names: dict[str, str] = {}
        if pykrx_stock is not None:
            try:
                tickers = pykrx_stock.get_market_ticker_list(_to_yyyymmdd(date_iso), market="KOSPI")
                for t in tickers or []:
                    code = str(t).zfill(6)
                    try:
                        nm = pykrx_stock.get_market_ticker_name(code)
                        if isinstance(nm, str) and nm:
                            names[code] = nm.strip()
                    except Exception:
                        continue
            except Exception as e:
                logger.debug("pykrx ticker list failed: %s", e)

        if not names and fdr is not None:
            try:
                listing = fdr.StockListing("KOSPI")
                if listing is not None and not listing.empty:
                    for _, row in listing.iterrows():
                        code = str(row.get("Code") or row.get("Symbol") or "").zfill(6)
                        nm = str(row.get("Name") or "").strip()
                        if len(code) == 6 and nm:
                            names[code] = nm
            except Exception as e:
                logger.debug("FDR StockListing fallback failed: %s", e)

        with self._lock:
            self._ticker_name_cache[date_iso] = names
        return names

    def _get_market_snapshot(self, date_iso: str) -> pd.DataFrame:
        """해당 거래일 KOSPI 전 종목 시세 + 시가총액 통합 스냅샷을 1회 벌크 조회·캐시."""
        with self._lock:
            if date_iso in self._snapshot_cache:
                return self._snapshot_cache[date_iso].copy()

        ymd = _to_yyyymmdd(date_iso)
        ohlcv_df = pd.DataFrame()
        cap_df = pd.DataFrame()

        if pykrx_stock is not None:
            try:
                ohlcv_df = pykrx_stock.get_market_ohlcv_by_ticker(ymd, market="KOSPI")
                self._sleep()
                cap_df = pykrx_stock.get_market_cap_by_ticker(ymd, market="KOSPI")
            except Exception as e:
                logger.warning("pykrx bulk snapshot failed on %s: %s", date_iso, e)

        rows: list[dict[str, Any]] = []
        if ohlcv_df is not None and not ohlcv_df.empty:
            names = self._get_ticker_names(date_iso)
            for idx, row in ohlcv_df.iterrows():
                code = str(idx).zfill(6)
                close_p = float(pd.to_numeric(row.get("종가", row.get("Close", 0)), errors="coerce") or 0)
                open_p = float(pd.to_numeric(row.get("시가", row.get("Open", 0)), errors="coerce") or 0)
                high_p = float(pd.to_numeric(row.get("고가", row.get("High", 0)), errors="coerce") or 0)
                low_p = float(pd.to_numeric(row.get("저가", row.get("Low", 0)), errors="coerce") or 0)
                vol = float(pd.to_numeric(row.get("거래량", row.get("Volume", 0)), errors="coerce") or 0)
                amt = float(pd.to_numeric(row.get("거래대금", row.get("Amount", 0)), errors="coerce") or 0)
                chg = float(pd.to_numeric(row.get("등락률", 0), errors="coerce") or 0)
                marcap = 0.0
                if cap_df is not None and not cap_df.empty and idx in cap_df.index:
                    cap_row = cap_df.loc[idx]
                    marcap = float(pd.to_numeric(cap_row.get("시가총액", 0), errors="coerce") or 0)
                if amt <= 0 and close_p > 0 and vol > 0:
                    amt = close_p * vol
                rows.append(
                    {
                        "code": code,
                        "name": names.get(code, code),
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": vol,
                        "trading_val": amt,
                        "chg_pct": chg,
                        "marcap": marcap,
                    }
                )
        elif fdr is not None and not self.as_of_date:
            # pykrx 장애 시 당일 스냅샷 FDR 폴백
            try:
                listing = fdr.StockListing("KOSPI")
                if listing is not None and not listing.empty:
                    for _, row in listing.iterrows():
                        code = str(row.get("Code") or row.get("Symbol") or "").zfill(6)
                        close_p = float(pd.to_numeric(row.get("Close", 0), errors="coerce") or 0)
                        open_p = float(pd.to_numeric(row.get("Open", 0), errors="coerce") or 0)
                        high_p = float(pd.to_numeric(row.get("High", 0), errors="coerce") or 0)
                        low_p = float(pd.to_numeric(row.get("Low", 0), errors="coerce") or 0)
                        vol = float(pd.to_numeric(row.get("Volume", 0), errors="coerce") or 0)
                        amt = float(pd.to_numeric(row.get("Amount", 0), errors="coerce") or 0)
                        chg = float(pd.to_numeric(row.get("ChagesRatio", 0), errors="coerce") or 0)
                        marcap = float(pd.to_numeric(row.get("Marcap", 0), errors="coerce") or 0)
                        if amt <= 0 and close_p > 0 and vol > 0:
                            amt = close_p * vol
                        rows.append(
                            {
                                "code": code,
                                "name": str(row.get("Name") or code).strip(),
                                "open": open_p,
                                "high": high_p,
                                "low": low_p,
                                "close": close_p,
                                "volume": vol,
                                "trading_val": amt,
                                "chg_pct": chg,
                                "marcap": marcap,
                            }
                        )
            except Exception as e:
                logger.warning("FDR StockListing fallback failed: %s", e)

        if not rows and not self.as_of_date:
            try:
                naver_items = fetch_naver_kospi_universe(min_cap=0, max_cap=10_000_000_000_000_000)
                for item in naver_items:
                    close_p = float(item.get("price") or 0.0)
                    tval = float(item.get("trading_val") or 0.0)
                    vol = (tval / close_p) if (close_p > 0 and tval > 0) else 0.0
                    rows.append(
                        {
                            "code": item["code"],
                            "name": item["name"],
                            "open": close_p,
                            "high": close_p,
                            "low": close_p,
                            "close": close_p,
                            "volume": vol,
                            "trading_val": tval,
                            "chg_pct": float(item.get("chg_pct") or 0.0),
                            "marcap": float(item.get("marcap") or 0.0),
                        }
                    )
            except Exception as e:
                logger.warning("Naver KOSPI snapshot fallback failed: %s", e)

        snap_df = pd.DataFrame(rows)
        with self._lock:
            self._snapshot_cache[date_iso] = snap_df
        return snap_df.copy()

    def _get_krx_risk_codes(self) -> dict[str, str]:
        """KIND 공개 엔드포인트에서 관리종목/투자주의·경고·위험/정리매매 코드 목록을 조회(1시간 캐시)."""
        now = time.time()
        with self._lock:
            if self._risk_codes_cache and (now - self._risk_codes_cache[0]) < 3600:
                return self._risk_codes_cache[1]

        risk_map: dict[str, str] = {}
        endpoints = (
            ("https://kind.krx.co.kr/investwarn/adminissue.do?method=searchAdminIssueSub&currentPageSize=300&pageIndex=1&searchMode=&marketType=1", "관리종목"),
            ("https://kind.krx.co.kr/investwarn/investattent.do?method=searchInvestAttentSub&currentPageSize=300&pageIndex=1&marketType=1", "투자주의"),
            ("https://kind.krx.co.kr/investwarn/investwarning.do?method=searchInvestWarningSub&currentPageSize=300&pageIndex=1&marketType=1", "투자경고"),
            ("https://kind.krx.co.kr/investwarn/investrisk.do?method=searchInvestRiskSub&currentPageSize=300&pageIndex=1&marketType=1", "투자위험"),
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; KOSPI-Swing-Scanner/1.0)",
            "Referer": "https://kind.krx.co.kr/",
        }
        for url, label in endpoints:
            try:
                resp = requests.get(url, headers=headers, timeout=4.0)
                if resp.status_code == 200 and resp.text:
                    # KIND 테이블 내 javascript 함수 호출 또는 6자리 종목코드 패턴 추출
                    for code in re.findall(r"companysummary_open\(['\"](\d{6})['\"]\)", resp.text):
                        risk_map.setdefault(code, label)
            except Exception:
                continue

        with self._lock:
            self._risk_codes_cache = (now, risk_map)
        return risk_map

    def get_universe(
        self,
        min_cap: int | None = None,
        max_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        lo_cap = settings.min_marcap_krw if min_cap is None else int(min_cap)
        hi_cap = settings.max_marcap_krw if max_cap is None else int(max_cap)

        date_iso = self._effective_end_date()
        snap = self._get_market_snapshot(date_iso)
        if snap.empty:
            return []

        result: list[dict[str, Any]] = []
        for _, row in snap.iterrows():
            code = str(row["code"]).zfill(6)
            name = str(row["name"]).strip()
            if not is_eligible_common_stock(code, name):
                continue
            marcap = float(row.get("marcap", 0.0))
            if marcap <= 0 or not (lo_cap <= marcap <= hi_cap):
                continue
            if float(row.get("close", 0.0)) <= 0:
                continue
            result.append(
                {
                    "code": code,
                    "name": name,
                    "marcap": int(marcap),
                    "trading_val": float(row.get("trading_val", 0.0)),
                    "price": float(row.get("close", 0.0)),
                    "chg_pct": float(row.get("chg_pct", 0.0)),
                }
            )

        result.sort(key=lambda x: x["trading_val"], reverse=True)
        return result

    def get_safety_status(self, code: str) -> dict[str, Any]:
        norm_code = str(code).zfill(6)
        date_iso = self._effective_end_date()
        snap = self._get_market_snapshot(date_iso)

        price = 0.0
        chg_pct = 0.0
        trading_val = 0.0
        name = norm_code
        open_p = 0.0
        vol = 0.0

        if not snap.empty:
            hit = snap[snap["code"] == norm_code]
            if not hit.empty:
                r = hit.iloc[0]
                name = str(r.get("name", norm_code))
                price = float(r.get("close", 0.0))
                open_p = float(r.get("open", 0.0))
                vol = float(r.get("volume", 0.0))
                chg_pct = float(r.get("chg_pct", 0.0))
                trading_val = float(r.get("trading_val", 0.0))

        if not is_eligible_common_stock(norm_code, name):
            return {
                "is_safe": False,
                "reject_reason": "보통주/제외키워드 부적격",
                "price": price,
                "chg_pct": chg_pct,
                "trading_val": trading_val,
            }

        risk_map = self._get_krx_risk_codes()
        if norm_code in risk_map:
            return {
                "is_safe": False,
                "reject_reason": f"KRX 지정({risk_map[norm_code]})",
                "price": price,
                "chg_pct": chg_pct,
                "trading_val": trading_val,
            }

        # 거래정지(당일 거래량 0 또는 시가 0) 차단
        if price <= 0 or vol <= 0 or open_p <= 0:
            return {
                "is_safe": False,
                "reject_reason": "거래정지 또는 당일 미체결",
                "price": price,
                "chg_pct": chg_pct,
                "trading_val": trading_val,
            }

        return {
            "is_safe": True,
            "reject_reason": "",
            "price": price,
            "chg_pct": chg_pct,
            "trading_val": trading_val,
        }

    def get_daily_ohlcv(
        self,
        code: str,
        bars: int = 150,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        norm_code = str(code).zfill(6)
        eff_end = self._effective_end_date(end_date)
        cache_key = (norm_code, eff_end)

        with self._lock:
            cached = self._ohlcv_cache.get(cache_key)
        if cached is not None and len(cached) >= bars:
            return cached.tail(bars).reset_index(drop=True)

        end_dt = datetime.strptime(eff_end, "%Y-%m-%d").date()
        # 휴장일 감안하여 약 1.75배 달력일 조회
        cal_days = max(240, int(bars * 1.75) + 30)
        start_dt = end_dt - timedelta(days=cal_days)

        raw_df = pd.DataFrame()
        if pykrx_stock is not None:
            for adj in (True, False):
                try:
                    raw_df = pykrx_stock.get_market_ohlcv_by_date(
                        start_dt.strftime("%Y%m%d"),
                        end_dt.strftime("%Y%m%d"),
                        norm_code,
                        adjusted=adj,
                    )
                    self._sleep()
                    if raw_df is not None and not raw_df.empty:
                        break
                except Exception as e:
                    logger.debug("pykrx OHLCV (%s, adj=%s) failed: %s", norm_code, adj, e)

        if (raw_df is None or raw_df.empty) and fdr is not None:
            try:
                raw_df = fdr.DataReader(
                    norm_code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
                )
            except Exception as e:
                logger.debug("FDR OHLCV (%s) failed: %s", norm_code, e)

        norm = _normalize_ohlcv_df(raw_df)
        if norm.empty:
            norm = fetch_naver_fchart_ohlcv(norm_code, bars=bars, end_date=eff_end)
        if not norm.empty:
            # 과거 거래정지일(open==0, close>0) 시가·고가·저가를 종가로 보정하여 지표 NaN 방지
            halted_bars = (norm["open"] <= 0) & (norm["close"] > 0)
            for col in ("open", "high", "low"):
                norm.loc[halted_bars, col] = norm.loc[halted_bars, "close"]

            with self._lock:
                self._ohlcv_cache[cache_key] = norm

        return norm.tail(bars).reset_index(drop=True)

    def get_investor_supply(
        self,
        code: str,
        days: int = 5,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        norm_code = str(code).zfill(6)
        eff_end = self._effective_end_date(end_date)
        end_dt = datetime.strptime(eff_end, "%Y-%m-%d").date()
        start_dt = end_dt - timedelta(days=max(20, days * 4))

        default_resp = {
            "frgn_today": 0,
            "orgn_today": 0,
            "frgn_3d": 0,
            "orgn_3d": 0,
            "total_5d": 0,
            "pos_days_5d": 0,
            "intraday_pending": False,
        }
        if pykrx_stock is None:
            return default_resp

        try:
            df = pykrx_stock.get_market_trading_volume_by_date(
                start_dt.strftime("%Y%m%d"),
                end_dt.strftime("%Y%m%d"),
                norm_code,
                detail=False,
            )
            self._sleep()
        except Exception as e:
            logger.debug("pykrx supply (%s) failed: %s", norm_code, e)
            return default_resp

        if df is None or df.empty:
            return default_resp

        frgn_col = "외국인합계" if "외국인합계" in df.columns else ("외국인" if "외국인" in df.columns else None)
        orgn_col = "기관합계" if "기관합계" in df.columns else ("기관" if "기관" in df.columns else None)
        if not frgn_col or not orgn_col:
            return default_resp

        work = df[[frgn_col, orgn_col]].copy()
        work[frgn_col] = pd.to_numeric(work[frgn_col], errors="coerce").fillna(0).astype(int)
        work[orgn_col] = pd.to_numeric(work[orgn_col], errors="coerce").fillna(0).astype(int)
        work["date_iso"] = [_to_iso_date(ix) for ix in work.index]
        work = work[work["date_iso"] <= eff_end].tail(max(days, 5)).reset_index(drop=True)
        if work.empty:
            return default_resp

        latest_row_date = str(work["date_iso"].iloc[-1])
        intraday_pending = latest_row_date != eff_end

        if intraday_pending:
            frgn_today = 0
            orgn_today = 0
        else:
            frgn_today = int(work[frgn_col].iloc[-1])
            orgn_today = int(work[orgn_col].iloc[-1])

        w3 = work.tail(3)
        w5 = work.tail(days)
        frgn_3d = int(w3[frgn_col].sum())
        orgn_3d = int(w3[orgn_col].sum())
        combined_5 = w5[frgn_col] + w5[orgn_col]
        total_5d = int(combined_5.sum())
        pos_days_5d = int((combined_5 > 0).sum())

        return {
            "frgn_today": frgn_today,
            "orgn_today": orgn_today,
            "frgn_3d": frgn_3d,
            "orgn_3d": orgn_3d,
            "total_5d": total_5d,
            "pos_days_5d": pos_days_5d,
            "intraday_pending": intraday_pending,
            "orgn_series": [int(x) for x in w5[orgn_col].tolist()],
        }

    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        eff_end = self._effective_end_date(end_date)
        with self._lock:
            cached = self._index_cache.get(eff_end)
        if cached is not None and len(cached) >= bars:
            return cached.tail(bars).reset_index(drop=True)

        end_dt = datetime.strptime(eff_end, "%Y-%m-%d").date()
        cal_days = max(180, int(bars * 1.75) + 20)
        start_dt = end_dt - timedelta(days=cal_days)

        cols = ["date", "close", "chg_pct"]
        raw = pd.DataFrame()
        if pykrx_stock is not None:
            try:
                raw = pykrx_stock.get_index_ohlcv_by_date(
                    start_dt.strftime("%Y%m%d"),
                    end_dt.strftime("%Y%m%d"),
                    "1001",
                )
                self._sleep()
            except Exception as e:
                logger.debug("pykrx index failed: %s", e)

        if (raw is None or raw.empty) and fdr is not None:
            try:
                raw = fdr.DataReader("KS11", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
            except Exception as e:
                logger.debug("FDR KS11 index failed: %s", e)

        if raw is None or raw.empty:
            naver_idx = fetch_naver_index_ohlcv(bars=bars, end_date=eff_end)
            if not naver_idx.empty:
                with self._lock:
                    self._index_cache[eff_end] = naver_idx
            return naver_idx

        work = raw.reset_index().copy()
        date_col = next((c for c in ("날짜", "Date", "index") if c in work.columns), work.columns[0])
        close_col = next((c for c in ("종가", "Close") if c in work.columns), None)
        if close_col is None:
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame()
        out["date"] = work[date_col].map(_to_iso_date)
        out["close"] = pd.to_numeric(work[close_col], errors="coerce").fillna(0.0)
        out = out[out["close"] > 0].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        out["chg_pct"] = (out["close"].pct_change().fillna(0.0) * 100.0).round(2)

        with self._lock:
            self._index_cache[eff_end] = out
        return out.tail(bars).reset_index(drop=True)

    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        eff_end = self._effective_end_date(end_date)
        snap = self._get_market_snapshot(eff_end)
        if snap.empty:
            if not end_date and not self.as_of_date:
                return fetch_naver_top_gainers(
                    top_n=top_n,
                    min_trading_val=settings.top20_min_trading_val_krw,
                )
            return []

        risk_map = self._get_krx_risk_codes()
        rows: list[dict[str, Any]] = []
        for _, r in snap.iterrows():
            code = str(r["code"]).zfill(6)
            name = str(r["name"]).strip()
            if not is_eligible_common_stock(code, name):
                continue
            if code in risk_map:
                continue
            close_p = float(r.get("close", 0.0))
            open_p = float(r.get("open", 0.0))
            vol = float(r.get("volume", 0.0))
            tval = float(r.get("trading_val", 0.0))
            chg = float(r.get("chg_pct", 0.0))
            marcap = int(float(r.get("marcap", 0.0)))
            if close_p <= 0 or open_p <= 0 or vol <= 0:
                continue
            if tval < settings.top20_min_trading_val_krw:
                continue
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "close": close_p,
                    "chg_pct": round(chg, 2),
                    "trading_val": tval,
                    "marcap": marcap,
                }
            )

        rows.sort(key=lambda x: x["chg_pct"], reverse=True)
        return rows[: max(1, int(top_n))]

    def get_sector_map(self) -> dict[str, str]:
        with self._lock:
            if self._sector_map_cache is not None:
                return dict(self._sector_map_cache)

        mapping: dict[str, str] = {}
        if fdr is not None:
            try:
                listing = fdr.StockListing("KRX-DESC")
                if listing is None or listing.empty:
                    listing = fdr.StockListing("KOSPI")
                if listing is not None and not listing.empty:
                    for _, row in listing.iterrows():
                        code = str(row.get("Code") or row.get("Symbol") or "").zfill(6)
                        sec = str(row.get("Sector") or row.get("Industry") or "").strip()
                        if len(code) == 6 and sec and sec != "nan":
                            mapping[code] = sec
            except Exception as e:
                logger.debug("FDR sector map load failed: %s", e)

        if not mapping:
            mapping = fetch_kind_sector_map()

        with self._lock:
            self._sector_map_cache = mapping
        return dict(mapping)

    def is_trading_day(self, date_str: str | None = None) -> bool:
        eff = date_str or datetime.now(tz=KST).strftime("%Y-%m-%d")
        try:
            dt_obj = datetime.strptime(eff[:10], "%Y-%m-%d").date()
        except ValueError:
            return True
        if dt_obj.weekday() >= 5:
            return False
        now_kst = datetime.now(tz=KST)
        # 과거 날짜이거나 09:10 이후 당일은 실제 지수 데이터 존재 여부로 휴장일 판별
        if eff[:10] < now_kst.strftime("%Y-%m-%d") or (
            eff[:10] == now_kst.strftime("%Y-%m-%d") and (now_kst.hour > 9 or (now_kst.hour == 9 and now_kst.minute >= 10))
        ):
            return self._resolve_latest_trading_day(eff[:10]) == eff[:10]
        return True
