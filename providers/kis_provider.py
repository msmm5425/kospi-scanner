from __future__ import annotations

import base64
import hashlib
import hmac
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
from .naver_fallback import (
    fetch_kind_sector_map,
    fetch_naver_index_ohlcv,
    fetch_naver_kospi_universe,
    fetch_naver_top_gainers,
)

try:
    import FinanceDataReader as fdr
except ImportError:  # pragma: no cover
    fdr = None

logger = logging.getLogger(__name__)


def _default_token_cache_file() -> Path:
    override = os.getenv("KIS_TOKEN_CACHE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        return Path.home() / ".cache" / "kospi_scanner" / "kis_token.json"
    except Exception:
        return Path(tempfile.gettempdir()) / "kospi_scanner_kis_token.json"


def _derive_cache_key(app_key: str, app_secret: str) -> bytes:
    seed = f"kis-tok-v1:{app_key}:{app_secret}".encode("utf-8")
    return hashlib.sha256(seed).digest()


def _keystream(key: bytes, salt: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0
    total = 0
    while total < length:
        blocks.append(hashlib.sha256(key + salt + counter.to_bytes(4, "big")).digest())
        counter += 1
        total += 32
    return b"".join(blocks)[:length]


def _encrypt_token_record(record: dict[str, Any], app_key: str, app_secret: str) -> str:
    raw = json.dumps(record, ensure_ascii=False).encode("utf-8")
    if not app_secret:
        return raw.decode("utf-8")
    key = _derive_cache_key(app_key, app_secret)
    salt = os.urandom(16)
    stream = _keystream(key, salt, len(raw))
    ct = bytes(a ^ b for a, b in zip(raw, stream))
    mac = hmac.new(key, salt + ct, hashlib.sha256).hexdigest()
    return json.dumps(
        {
            "v": 1,
            "enc": True,
            "salt": salt.hex(),
            "ct": base64.b64encode(ct).decode("ascii"),
            "mac": mac,
        }
    )


def _decrypt_token_record(text: str, app_key: str, app_secret: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("enc"):
        return data
    if not app_secret:
        return None
    try:
        key = _derive_cache_key(app_key, app_secret)
        salt = bytes.fromhex(str(data["salt"]))
        ct = base64.b64decode(str(data["ct"]))
        expected_mac = hmac.new(key, salt + ct, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, str(data.get("mac") or "")):
            return None
        stream = _keystream(key, salt, len(ct))
        plain = bytes(a ^ b for a, b in zip(ct, stream))
        parsed = json.loads(plain.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class KISProvider(MarketDataProvider):
    """[연동 완료] 한국투자증권 KIS Developers REST API 어댑터 (GitHub Actions 호환)."""

    provider_name = "kis"

    def __init__(self, as_of_date: str | None = None):
        self.as_of_date = as_of_date
        self.app_key = settings.kis_app_key
        self.app_secret = settings.kis_app_secret
        default_base = (
            "https://openapivts.koreainvestment.com:29443"
            if settings.kis_is_paper
            else "https://openapi.koreainvestment.com:9443"
        )
        self.base_url = (settings.kis_api_base_url or default_base).rstrip("/")
        if settings.kis_is_paper and "openapi.koreainvestment.com" in self.base_url:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        self._lock = threading.RLock()
        self._session = requests.Session()
        self._last_req_ts = 0.0
        self._min_interval_sec = 0.16  # KIS 실전 앱키 버스트(EGW00201) 방지 안전 간격 (~6 req/s)
        self._token: str = ""
        self._token_expire_ts: float = 0.0
        self._token_issued_date_kst: str = ""
        self._token_cache_file = _default_token_cache_file()
        self._legacy_token_cache_file = Path(tempfile.gettempdir()) / "kospi_scanner_kis_token.json"
        self._universe_cache: dict[str, list[dict[str, Any]]] = {}
        self._marcap_map_cache: dict[str, int] = {}
        self._holiday_cache: dict[str, bool] = {}
        self._sector_map_cache: dict[str, str] | None = None

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

    def _load_cached_token_record(self) -> dict[str, Any] | None:
        candidates = [self._token_cache_file]
        if self._legacy_token_cache_file != self._token_cache_file:
            candidates.append(self._legacy_token_cache_file)
        for path in candidates:
            if not path.exists():
                continue
            try:
                cached = _decrypt_token_record(
                    path.read_text(encoding="utf-8"),
                    self.app_key,
                    self.app_secret,
                )
                if not cached:
                    continue
                if (
                    cached.get("base_url") == self.base_url
                    and cached.get("app_key_prefix") == self.app_key[:8]
                    and cached.get("access_token")
                ):
                    return cached
            except Exception:
                continue
        return None

    def _save_cached_token_record(self, record: dict[str, Any]) -> None:
        serialized = _encrypt_token_record(record, self.app_key, self.app_secret)
        targets = [self._token_cache_file]
        if (
            self._legacy_token_cache_file != self._token_cache_file
            and not os.getenv("KIS_TOKEN_CACHE_PATH", "").strip()
        ):
            targets.append(self._legacy_token_cache_file)
        for path in targets:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(serialized, encoding="utf-8")
            except Exception:
                pass

    def ensure_daily_token(
        self,
        min_ttl_sec: float = 120.0,
        force_refresh: bool = False,
        retry_delay_sec: float = 0.0,
    ) -> dict[str, Any]:
        """1일 1회 발급 원칙에 따라 기존 유효 토큰을 최대한 재사용하고 필요 시에만 발급한다."""
        self._require_credentials()
        now = time.time()
        today_kst = datetime.now(tz=KST).strftime("%Y-%m-%d")

        env_tok = os.getenv("KIS_ACCESS_TOKEN", "").strip()
        if env_tok and not force_refresh:
            return {
                "access_token": env_tok,
                "source": "env",
                "remaining_sec": 86400.0,
                "issued_date_kst": today_kst,
            }

        with self._lock:
            if not force_refresh and self._token:
                rem = self._token_expire_ts - now
                same_day_valid = self._token_issued_date_kst == today_kst and rem > 120.0
                if rem > min_ttl_sec or same_day_valid:
                    return {
                        "access_token": self._token,
                        "source": "memory",
                        "remaining_sec": rem,
                        "issued_date_kst": self._token_issued_date_kst,
                    }

            if not force_refresh:
                cached = self._load_cached_token_record()
                if cached:
                    exp_ts = float(cached.get("expire_ts", 0.0))
                    rem = exp_ts - now
                    issued_dt = str(cached.get("issued_date_kst") or "")
                    same_day_valid = issued_dt == today_kst and rem > 120.0
                    if rem > min_ttl_sec or same_day_valid:
                        self._token = str(cached["access_token"])
                        self._token_expire_ts = exp_ts
                        self._token_issued_date_kst = issued_dt or today_kst
                        return {
                            "access_token": self._token,
                            "source": "cache",
                            "remaining_sec": rem,
                            "issued_date_kst": self._token_issued_date_kst,
                        }

            # 토큰 재발급 (1분당 1회 제한 EGW00133 방어 포함)
            data: dict[str, Any] = {}
            for attempt in range(2):
                resp = self._session.post(
                    f"{self.base_url}/oauth2/tokenP",
                    json={
                        "grant_type": "client_credentials",
                        "appkey": self.app_key,
                        "appsecret": self.app_secret,
                    },
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    data = resp.json() or {}
                    break
                err_payload: dict[str, Any] = {}
                try:
                    err_payload = resp.json() or {}
                except Exception:
                    pass
                err_cd = str(err_payload.get("error_code") or "")
                if err_cd == "EGW00133" and retry_delay_sec > 0 and attempt == 0:
                    logger.warning("KIS 토큰 발급 1분 제한(EGW00133) 감지 — %.0f초 대기 후 1회 재시도", retry_delay_sec)
                    time.sleep(retry_delay_sec)
                    continue
                if err_payload:
                    raise RuntimeError(
                        f"KIS 토큰 발급 실패 (HTTP {resp.status_code}): "
                        f"{err_cd} {err_payload.get('error_description', '')}"
                    )
                resp.raise_for_status()

            token = str(data.get("access_token") or "")
            expires_in = float(data.get("expires_in") or 86400)
            if not token:
                raise RuntimeError(f"KIS 토큰 발급 실패: {data}")

            now_done = time.time()
            self._token = token
            self._token_expire_ts = now_done + expires_in
            self._token_issued_date_kst = datetime.now(tz=KST).strftime("%Y-%m-%d")
            record = {
                "base_url": self.base_url,
                "app_key_prefix": self.app_key[:8],
                "access_token": self._token,
                "expire_ts": self._token_expire_ts,
                "issued_at_ts": now_done,
                "issued_date_kst": self._token_issued_date_kst,
            }
            self._save_cached_token_record(record)
            return {
                "access_token": self._token,
                "source": "issued",
                "remaining_sec": expires_in,
                "issued_date_kst": self._token_issued_date_kst,
            }

    def _ensure_token(self, force_refresh: bool = False) -> str:
        return str(self.ensure_daily_token(min_ttl_sec=120.0, force_refresh=force_refresh)["access_token"])

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
        refreshed_once = False
        for attempt in range(5):
            self._throttle()
            resp = self._session.get(url, headers=headers, params=params, timeout=12.0)
            if resp.status_code == 200:
                payload = resp.json()
                if str(payload.get("rt_cd")) == "0":
                    return payload
                msg_cd = str(payload.get("msg_cd") or "")
                # 초당 거래건수 초과 재시도
                if msg_cd == "EGW00201" and attempt < 4:
                    time.sleep(0.55 * (attempt + 1))
                    continue
                # 토큰 만료 코드 재시도 (요청당 최대 1회만 갱신)
                if msg_cd in {"EGW00123", "EGW00121"} and not refreshed_once and attempt < 4:
                    refreshed_once = True
                    with self._lock:
                        self._token = ""
                        self._token_expire_ts = 0.0
                    headers["authorization"] = f"Bearer {self._ensure_token(force_refresh=True)}"
                    continue
                raise RuntimeError(
                    f"KIS API 오류 ({tr_id}): {payload.get('msg_cd')} {payload.get('msg1')}"
                )
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                time.sleep(0.65 * (attempt + 1))
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

        result = fetch_naver_kospi_universe(min_cap=lo_cap, max_cap=hi_cap, session=self._session)

        if not result and fdr is not None:
            try:
                listing = fdr.StockListing("KOSPI")
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
                        vol = float(pd.to_numeric(row.get("Volume", 0), errors="coerce") or 0)
                        tval = float(pd.to_numeric(row.get("Amount", 0), errors="coerce") or 0)
                        if tval <= 0 and close_p > 0 and vol > 0:
                            tval = close_p * vol
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
            except Exception as e:
                logger.warning("FDR StockListing fallback failed: %s", e)

        with self._lock:
            self._universe_cache[cache_key] = result
            for item in result:
                self._marcap_map_cache[item["code"]] = item["marcap"]
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
        if str(out.get("temp_stop_yn", "N")).upper() == "Y" or str(out.get("trht_yn", "N")).upper() == "Y":
            reasons.append("거래정지")
        if str(out.get("sltr_yn", "N")).upper() == "Y":
            reasons.append("정리매매")
        warn_cd = str(out.get("mrkt_warn_cls_code") or "00")
        warn_map = {"01": "투자주의", "02": "투자경고", "03": "투자위험"}
        if warn_cd in warn_map:
            reasons.append(warn_map[warn_cd])
        stat_cd = str(out.get("iscd_stat_cls_code") or "").strip()
        stat_map = {
            "51": "관리종목",
            "52": "투자위험",
            "53": "투자경고",
            "54": "투자주의",
            "58": "거래정지",
            "59": "단기과열",
        }
        if stat_cd in stat_map and stat_map[stat_cd] not in reasons:
            reasons.append(stat_map[stat_cd])

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
        intraday_pending = False
        for r in rows:
            dt = str(r.get("stck_bsop_date") or "")
            if eff_end and dt > eff_end:
                continue
            raw_f = str(r.get("frgn_ntby_qty") or "").strip()
            raw_o = str(r.get("orgn_ntby_qty") or "").strip()
            # 장중 당일 행은 확정 전까지 "" 빈 문자열로 내려오므로 3일/5일 합산에서 제외
            if raw_f == "" and raw_o == "":
                intraday_pending = True
                continue
            filtered.append(
                {
                    "date": dt,
                    "frgn": int(float(raw_f or 0)),
                    "orgn": int(float(raw_o or 0)),
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
            "frgn_today": 0 if intraday_pending else recent[-1]["frgn"],
            "orgn_today": 0 if intraday_pending else recent[-1]["orgn"],
            "frgn_3d": sum(r["frgn"] for r in w3),
            "orgn_3d": sum(r["orgn"] for r in w3),
            "total_5d": sum(r["frgn"] + r["orgn"] for r in recent),
            "pos_days_5d": sum(1 for r in recent if (r["frgn"] + r["orgn"]) > 0),
            "intraday_pending": intraday_pending,
            "orgn_series": [r["orgn"] for r in recent],
        }

    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self._require_credentials()
        eff_end = (end_date or self.as_of_date or datetime.now(tz=KST).strftime("%Y-%m-%d")).replace("-", "")
        end_dt = datetime.strptime(eff_end, "%Y%m%d").date()

        all_rows: list[dict[str, Any]] = []
        cursor_end = end_dt
        try:
            for _ in range(4):
                cursor_start = cursor_end - timedelta(days=140)
                payload = self._request_get(
                    "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
                    "FHKUP03500100",
                    {
                        "FID_COND_MRKT_DIV_CODE": "U",
                        "FID_INPUT_ISCD": "0001",
                        "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
                        "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
                        "FID_PERIOD_DIV_CODE": "D",
                    },
                )
                chunk = payload.get("output2") or []
                if not chunk:
                    break
                all_rows.extend(chunk)
                oldest = min(str(r.get("stck_bsop_date") or "") for r in chunk if r.get("stck_bsop_date"))
                if not oldest or len(all_rows) >= (bars + 5):
                    break
                cursor_end = datetime.strptime(oldest, "%Y%m%d").date() - timedelta(days=1)
        except Exception as e:
            logger.warning("KIS 지수 일봉 조회 실패 — Naver fchart 폴백 시도: %s", e)

        records = []
        for r in all_rows:
            dt = str(r.get("stck_bsop_date") or "")
            cl = float(r.get("bstp_nmix_prpr") or 0)
            if len(dt) != 8 or cl <= 0:
                continue
            records.append(
                {
                    "date": f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}",
                    "close": cl,
                }
            )
        if not records:
            return fetch_naver_index_ohlcv(
                bars=bars,
                end_date=end_date or self.as_of_date,
                session=self._session,
            )
        df = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df["chg_pct"] = (df["close"].pct_change().fillna(0.0) * 100.0).round(2)
        return df.tail(bars).reset_index(drop=True)

    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        if end_date or self.as_of_date:
            raise NotImplementedError("KIS 실시간 등락률 랭킹은 과거 날짜 역조회를 지원하지 않습니다. --provider pykrx를 사용하세요.")

        out: list[dict[str, Any]] = []
        try:
            payload = self._request_get(
                "/uapi/domestic-stock/v1/ranking/fluctuation",
                "FHPST01700000",
                {
                    "fid_cond_mrkt_div_code": "J",
                    "fid_cond_scr_div_code": "20170",
                    "fid_input_iscd": "0001",
                    "fid_rank_sort_cls_code": "0",
                    "fid_input_cnt_1": "0",
                    "fid_prc_cls_code": "1",
                    "fid_input_price_1": "",
                    "fid_input_price_2": "",
                    "fid_vol_cnt": "",
                    "fid_trgt_cls_code": "0",
                    "fid_trgt_exls_cls_code": "0000000000",
                    "fid_div_cls_code": "0",
                    "fid_rsfl_rate1": "",
                    "fid_rsfl_rate2": "",
                },
            )
            rows = payload.get("output") or []
            for r in rows:
                code = str(r.get("stck_shrn_iscd") or "").zfill(6)
                name = str(r.get("hts_kor_isnm") or "").strip()
                if not is_eligible_common_stock(code, name):
                    continue
                close_p = float(r.get("stck_prpr") or 0)
                chg = float(r.get("prdy_ctrt") or 0)
                vol = float(r.get("acml_vol") or 0)
                tval = float(r.get("acml_tr_pbmn") or (close_p * vol))
                if close_p <= 0 or tval < settings.top20_min_trading_val_krw:
                    continue
                out.append(
                    {
                        "code": code,
                        "name": name,
                        "close": close_p,
                        "chg_pct": round(chg, 2),
                        "trading_val": tval,
                        "marcap": self._marcap_map_cache.get(code, 0),
                    }
                )
        except Exception as e:
            logger.warning("KIS 등락률 랭킹(FHPST01700000) 오류 — Naver TOP 20 폴백 사용: %s", e)

        if not out:
            return fetch_naver_top_gainers(
                top_n=top_n,
                min_trading_val=settings.top20_min_trading_val_krw,
                marcap_map=self._marcap_map_cache,
                session=self._session,
            )

        out.sort(key=lambda x: x["chg_pct"], reverse=True)
        return out[: max(1, int(top_n))]

    def get_sector_map(self) -> dict[str, str]:
        with self._lock:
            if self._sector_map_cache is not None:
                return dict(self._sector_map_cache)
        mapping = fetch_kind_sector_map(session=self._session)
        with self._lock:
            self._sector_map_cache = mapping
        return dict(mapping)

    def is_trading_day(self, date_str: str | None = None) -> bool:
        """KIS CTCA0903R 국내휴일여부조회를 사용해 KRX 공휴일/개장일 여부를 판별한다."""
        eff_iso = (date_str or datetime.now(tz=KST).strftime("%Y-%m-%d"))[:10]
        eff_ymd = eff_iso.replace("-", "")
        try:
            if datetime.strptime(eff_ymd, "%Y%m%d").weekday() >= 5:
                return False
        except ValueError:
            return True

        with self._lock:
            if eff_ymd in self._holiday_cache:
                return self._holiday_cache[eff_ymd]

        try:
            payload = self._request_get(
                "/uapi/domestic-stock/v1/quotations/chk-holiday",
                "CTCA0903R",
                {"BASS_DT": eff_ymd, "CTX_AREA_NK": "", "CTX_AREA_FK": ""},
            )
            rows = payload.get("output") or []
            with self._lock:
                for r in rows:
                    dt_val = str(r.get("bass_dt") or "").strip()
                    opnd = str(r.get("opnd_yn") or "").upper() == "Y"
                    if len(dt_val) == 8:
                        self._holiday_cache[dt_val] = opnd
                if eff_ymd in self._holiday_cache:
                    return self._holiday_cache[eff_ymd]
        except Exception as e:
            logger.warning("KIS chk-holiday 조회 실패(%s): %s", eff_ymd, e)

        return True

    def get_session_quote(self, code: str, session: str = "REGULAR") -> dict[str, Any] | None:
        """프리/애프터장(통합시세 UN -> KRX J 폴백) 또는 정규장(J) 참고 시세를 조회한다."""
        norm_code = str(code).zfill(6)
        sess = str(session or "REGULAR").upper()
        codes_to_try = ("UN", "J") if sess in {"PREMARKET", "AFTERMARKET"} else ("J",)

        for mkt_cd in codes_to_try:
            try:
                payload = self._request_get(
                    "/uapi/domestic-stock/v1/quotations/inquire-price",
                    "FHKST01010100",
                    {"FID_COND_MRKT_DIV_CODE": mkt_cd, "FID_INPUT_ISCD": norm_code},
                )
                out = payload.get("output") or {}
                price = float(out.get("stck_prpr") or 0)
                if price > 0:
                    return {
                        "price": price,
                        "chg_pct": float(out.get("prdy_ctrt") or 0),
                        "trading_val": float(out.get("acml_tr_pbmn") or 0),
                        "market_code": mkt_cd,
                    }
            except Exception as e:
                logger.debug("KIS get_session_quote (%s, %s) failed: %s", norm_code, mkt_cd, e)
        return None
