from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config.settings import KST, settings
from .base import MarketDataProvider, is_eligible_common_stock

try:
    import FinanceDataReader as fdr
except ImportError:  # pragma: no cover
    fdr = None

logger = logging.getLogger(__name__)


class KISProvider(MarketDataProvider):
    """[연동 대기] 한국투자증권 KIS Developers REST API 어댑터 (GitHub Actions 호환)."""

    provider_name = "kis"

    def __init__(self, as_of_date: str | None = None):
        self.as_of_date = as_of_date
        self.app_key = settings.kis_app_key
        self.app_secret = settings.kis_app_secret
        self.base_url = settings.kis_api_base_url.rstrip("/")
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._last_req_ts = 0.0
        self._min_interval_sec = 0.056  # 실전 초당 20건 대비 안전 상한(~17.8 req/s)
        self._token: str = ""
        self._token_expire_ts: float = 0.0
        self._token_cache_file = Path(tempfile.gettempdir()) / "kospi_scanner_kis_token.json"
        self._universe_cache: dict[str, list[dict[str, Any]]] = {}

    def _require_credentials(self) -> None:
        if not self.app_key or not self.app_secret:
            raise RuntimeError(
                "KIS API 키(KIS_APP_KEY / KIS_APP_SECRET)가 설정되지 않았습니다. "
                ".env 또는 GitHub Secrets를 확인하세요."
            )

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.time() - self._last_req_ts
            if elapsed < self._min_interval_sec:
                time.sleep(self._min_interval_sec - elapsed)
            self._last_req_ts = time.time()

    def _ensure_token(self) -> str:
        self._require_credentials()
        now = time.time()
        with self._lock:
            if self._token and now < (self._token_expire_ts - 120):
                return self._token

            if self._token_cache_file.exists():
                try:
                    cached = json.loads(self._token_cache_file.read_text(encoding="utf-8"))
                    if (
                        cached.get("base_url") == self.base_url
                        and cached.get("app_key_prefix") == self.app_key[:8]
                        and float(cached.get("expire_ts", 0)) - 120 > now
                    ):
                        self._token = str(cached["access_token"])
                        self._token_expire_ts = float(cached["expire_ts"])
                        return self._token
                except Exception:
                    pass

            resp = self._session.post(
                f"{self.base_url}/oauth2/tokenP",
                json={
                    "grant_type": "client_credentials",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            token = str(data.get("access_token") or "")
            expires_in = float(data.get("expires_in") or 86400)
            if not token:
                raise RuntimeError(f"KIS 토큰 발급 실패: {data}")

            self._token = token
            self._token_expire_ts = now + expires_in
            try:
                self._token_cache_file.write_text(
                    json.dumps(
                        {
                            "base_url": self.base_url,
                            "app_key_prefix": self.app_key[:8],
                            "access_token": self._token,
                            "expire_ts": self._token_expire_ts,
                        }
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return self._token

    def _request_get(
        self,
        path: str,
        tr_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        token = self._ensure_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            self._throttle()
            resp = self._session.get(url, headers=headers, params=params, timeout=10.0)
            if resp.status_code == 200:
                payload = resp.json()
                if str(payload.get("rt_cd")) == "0":
                    return payload
                msg_cd = str(payload.get("msg_cd") or "")
                # 토큰 만료 코드 재시도
                if msg_cd in {"EGW00123", "EGW00121"} and attempt < 2:
                    with self._lock:
                        self._token = ""
                        self._token_expire_ts = 0.0
                    headers["authorization"] = f"Bearer {self._ensure_token()}"
                    continue
                raise RuntimeError(
                    f"KIS API 오류 ({tr_id}): {payload.get('msg_cd')} {payload.get('msg1')}"
                )
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(0.6 * (attempt + 1))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"KIS API 요청 실패 ({path}, {tr_id})")

    def get_universe(
        self,
        min_cap: int | None = None,
        max_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        lo_cap = settings.min_marcap_krw if min_cap is None else int(min_cap)
        hi_cap = settings.max_marcap_krw if max_cap is None else int(max_cap)
        self._require_credentials()

        cache_key = f"{lo_cap}_{hi_cap}"
        with self._lock:
            if cache_key in self._universe_cache:
                return list(self._universe_cache[cache_key])

        if fdr is None:
            raise RuntimeError("KISProvider 유니버스 조회를 위해 finance-datareader 패키지가 필요합니다.")

        listing = fdr.StockListing("KOSPI")
        result: list[dict[str, Any]] = []
        if listing is not None and not listing.empty:
            for _, row in listing.iterrows():
                code = str(row.get("Code") or row.get("Symbol") or "").zfill(6)
                name = str(row.get("Name") or "").strip()
                if not is_eligible_common_stock(code, name):
                    continue
                marcap = float(pd.to_numeric(row.get("Marcap", 0), errors="coerce") or 0)
                if not (lo_cap <= marcap <= hi_cap):
                    continue
                close_p = float(pd.to_numeric(row.get("Close", 0), errors="coerce") or 0)
                tval = float(pd.to_numeric(row.get("Amount", 0), errors="coerce") or 0)
                chg = float(pd.to_numeric(row.get("ChagesRatio", 0), errors="coerce") or 0)
                result.append(
                    {
                        "code": code,
                        "name": name,
                        "marcap": int(marcap),
                        "trading_val": tval,
                        "price": close_p,
                        "chg_pct": chg,
                    }
                )
        result.sort(key=lambda x: x["trading_val"], reverse=True)
        with self._lock:
            self._universe_cache[cache_key] = result
        return result

    def get_safety_status(self, code: str) -> dict[str, Any]:
        norm_code = str(code).zfill(6)
        payload = self._request_get(
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "FHKST01010100",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": norm_code},
        )
        out = payload.get("output") or {}
        price = float(out.get("stck_prpr") or 0)
        chg_pct = float(out.get("prdy_ctrt") or 0)
        trading_val = float(out.get("acml_tr_pbmn") or 0)

        reasons: list[str] = []
        if str(out.get("mang_issu_yn", "N")).upper() == "Y":
            reasons.append("관리종목")
        if str(out.get("trht_yn", "N")).upper() == "Y":
            reasons.append("거래정지")
        if str(out.get("sltr_yn", "N")).upper() == "Y":
            reasons.append("정리매매")
        warn_cd = str(out.get("mrkt_warn_cls_code") or "00")
        warn_map = {"01": "투자주의", "02": "투자경고", "03": "투자위험"}
        if warn_cd in warn_map:
            reasons.append(warn_map[warn_cd])

        return {
            "is_safe": len(reasons) == 0 and price > 0,
            "reject_reason": ", ".join(reasons) if reasons else ("" if price > 0 else "시세 없음"),
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
        eff_end = (end_date or self.as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d")).replace("-", "")
        end_dt = datetime.strptime(eff_end, "%Y%m%d").date()

        all_rows: list[dict[str, Any]] = []
        cursor_end = end_dt
        # KIS inquire-daily-itemchartprice는 1회 최대 100건 반환 -> 분할 역추적
        for _ in range(4):
            cursor_start = cursor_end - timedelta(days=170)
            payload = self._request_get(
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                "FHKST03010100",
                {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": norm_code,
                    "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0",  # 수정주가
                },
            )
            chunk = payload.get("output2") or []
            if not chunk:
                break
            all_rows.extend(chunk)
            oldest = min(str(r.get("stck_bsop_date") or "") for r in chunk if r.get("stck_bsop_date"))
            if not oldest or len(all_rows) >= bars:
                break
            cursor_end = datetime.strptime(oldest, "%Y%m%d").date() - timedelta(days=1)

        cols = ["date", "open", "high", "low", "close", "volume", "amount"]
        if not all_rows:
            return pd.DataFrame(columns=cols)

        records = []
        for r in all_rows:
            dt = str(r.get("stck_bsop_date") or "")
            if len(dt) != 8:
                continue
            close_p = float(r.get("stck_clpr") or 0)
            if close_p <= 0:
                continue
            records.append(
                {
                    "date": f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}",
                    "open": float(r.get("stck_oprc") or close_p),
                    "high": float(r.get("stck_hgpr") or close_p),
                    "low": float(r.get("stck_lwpr") or close_p),
                    "close": close_p,
                    "volume": float(r.get("acml_vol") or 0),
                    "amount": float(r.get("acml_tr_pbmn") or 0),
                }
            )
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df.tail(bars).reset_index(drop=True)

    def get_investor_supply(
        self,
        code: str,
        days: int = 5,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        norm_code = str(code).zfill(6)
        payload = self._request_get(
            "/uapi/domestic-stock/v1/quotations/inquire-investor",
            "FHKST01010900",
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": norm_code},
        )
        rows = payload.get("output") or []
        eff_end = (end_date or self.as_of_date or "").replace("-", "")
        filtered = []
        for r in rows:
            dt = str(r.get("stck_bsop_date") or "")
            if eff_end and dt > eff_end:
                continue
            filtered.append(
                {
                    "date": dt,
                    "frgn": int(float(r.get("frgn_ntby_qty") or 0)),
                    "orgn": int(float(r.get("orgn_ntby_qty") or 0)),
                }
            )
        filtered.sort(key=lambda x: x["date"])
        recent = filtered[-max(days, 5) :]
        if not recent:
            return {
                "frgn_today": 0,
                "orgn_today": 0,
                "frgn_3d": 0,
                "orgn_3d": 0,
                "total_5d": 0,
                "pos_days_5d": 0,
                "intraday_pending": True,
                "orgn_series": [],
            }

        w3 = recent[-3:]
        return {
            "frgn_today": recent[-1]["frgn"],
            "orgn_today": recent[-1]["orgn"],
            "frgn_3d": sum(r["frgn"] for r in w3),
            "orgn_3d": sum(r["orgn"] for r in w3),
            "total_5d": sum(r["frgn"] + r["orgn"] for r in recent),
            "pos_days_5d": sum(1 for r in recent if (r["frgn"] + r["orgn"]) > 0),
            "intraday_pending": False,
            "orgn_series": [r["orgn"] for r in recent],
        }

    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        eff_end = (end_date or self.as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d")).replace("-", "")
        payload = self._request_get(
            "/uapi/domestic-stock/v1/quotations/inquire-index-dailyprice",
            "FHKUP03500100",
            {
                "FID_COND_MRKT_DIV_CODE": "U",
                "FID_INPUT_ISCD": "0001",
                "FID_INPUT_DATE_1": eff_end,
                "FID_PERIOD_DIV_CODE": "D",
            },
        )
        rows = payload.get("output2") or []
        records = []
        for r in rows:
            dt = str(r.get("stck_bsop_date") or "")
            cl = float(r.get("bstp_nmix_prpr") or 0)
            if len(dt) != 8 or cl <= 0:
                continue
            records.append(
                {
                    "date": f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}",
                    "close": cl,
                    "chg_pct": float(r.get("bstp_nmix_prdy_ctrt") or 0),
                }
            )
        if not records:
            return pd.DataFrame(columns=["date", "close", "chg_pct"])
        df = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df.tail(bars).reset_index(drop=True)

    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if end_date or self.as_of_date:
            raise NotImplementedError("KIS 실시간 등락률 랭킹은 과거 날짜 역조회를 지원하지 않습니다. --provider pykrx를 사용하세요.")

        payload = self._request_get(
            "/uapi/domestic-stock/v1/ranking/fluctuation",
            "FHPST01700000",
            {
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20170",
                "fid_input_iscd": "0001",
                "fid_rank_sort_cls_code": "0",
                "fid_input_cnt_1": "0",
                "fid_prc_cls_code": "0",
                "fid_input_price_1": "",
                "fid_input_price_2": "",
                "fid_vol_cnt": "",
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_div_cls_code": "0",
                "fid_rsfl_rate1": "",
                "fid_rsfl_rate2": "",
            },
        )
        rows = payload.get("output") or []
        out: list[dict[str, Any]] = []
        for r in rows:
            code = str(r.get("stck_shrn_iscd") or "").zfill(6)
            name = str(r.get("hts_kor_isnm") or "").strip()
            if not is_eligible_common_stock(code, name):
                continue
            close_p = float(r.get("stck_prpr") or 0)
            chg = float(r.get("prdy_ctrt") or 0)
            tval = float(r.get("acml_tr_pbmn") or 0)
            if close_p <= 0 or tval < settings.top20_min_trading_val_krw:
                continue
            out.append(
                {
                    "code": code,
                    "name": name,
                    "close": close_p,
                    "chg_pct": round(chg, 2),
                    "trading_val": tval,
                    "marcap": 0,
                }
            )
            if len(out) >= top_n:
                break
        return out
