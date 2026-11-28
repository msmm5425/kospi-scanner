from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from config.settings import KST, settings
from engines.auto_learner import run_auto_learning
from engines.error_log import update_error_logs
from engines.scanner import run_scan
from engines.top20_analyzer import analyze_top20
from providers.base import MarketDataProvider, get_market_index_ohlcv
from storage.repo import StorageRepo, repo as default_repo

logger = logging.getLogger(__name__)


class _WalkForwardSliceProvider(MarketDataProvider):
    """선조회한 일봉·지수 캐시를 날짜별 무누수(zero-lookahead) 슬라이스로 제공하는 내부 어댑터."""

    provider_name = "bootstrap_slice"

    def __init__(
        self,
        universe_meta: list[dict[str, Any]],
        ohlcv_cache: dict[str, pd.DataFrame],
        supply_cache: dict[str, dict[str, Any]],
        index_df: pd.DataFrame,
        sector_map: dict[str, str] | None = None,
        kosdaq_index_df: pd.DataFrame | None = None,
    ):
        self._universe_meta = {str(u["code"]).zfill(6): u for u in universe_meta}
        self._ohlcv = ohlcv_cache
        self._supply = supply_cache
        self._index_df = index_df
        self._kosdaq_index_df = kosdaq_index_df if kosdaq_index_df is not None and not kosdaq_index_df.empty else index_df
        self._sector_map = sector_map or {}
        self.current_date: str = ""

    def set_date(self, target_date: str) -> None:
        self.current_date = target_date

    def get_universe(
        self,
        min_cap: int | None = None,
        max_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for code, df in self._ohlcv.items():
            sub = df[df["date"] <= self.current_date]
            if len(sub) < 2:
                continue
            last = sub.iloc[-1]
            prev = sub.iloc[-2]
            prev_c = float(prev["close"])
            close_p = float(last["close"])
            chg = ((close_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
            meta = self._universe_meta.get(code) or {}
            out.append(
                {
                    "code": code,
                    "name": meta.get("name", code),
                    "market": str(meta.get("market") or "KOSPI").upper(),
                    "marcap": int(meta.get("marcap", 500_000_000_000)),
                    "trading_val": float(last.get("amount") or (close_p * float(last["volume"]))),
                    "price": close_p,
                    "chg_pct": round(chg, 2),
                }
            )
        out.sort(key=lambda x: x["trading_val"], reverse=True)
        return out

    def get_safety_status(self, code: str) -> dict[str, Any]:
        norm = str(code).zfill(6)
        df = self._ohlcv.get(norm)
        if df is None:
            return {"is_safe": False, "reject_reason": "데이터 없음", "price": 0.0, "chg_pct": 0.0, "trading_val": 0.0}
        sub = df[df["date"] <= self.current_date]
        if len(sub) < 2:
            return {"is_safe": False, "reject_reason": "데이터 부족", "price": 0.0, "chg_pct": 0.0, "trading_val": 0.0}
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        close_p = float(last["close"])
        open_p = float(last["open"])
        vol = float(last["volume"])
        prev_c = float(prev["close"])
        chg = ((close_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        tval = float(last.get("amount") or (close_p * vol))
        is_safe = close_p > 0 and open_p > 0 and vol > 0
        return {
            "is_safe": is_safe,
            "reject_reason": "" if is_safe else "거래정지",
            "price": close_p,
            "chg_pct": round(chg, 2),
            "trading_val": tval,
        }

    def get_daily_ohlcv(
        self,
        code: str,
        bars: int = 150,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        norm = str(code).zfill(6)
        df = self._ohlcv.get(norm, pd.DataFrame())
        if df.empty:
            return df
        eff = end_date or self.current_date
        sub = df[df["date"] <= eff] if eff else df
        return sub.tail(bars).reset_index(drop=True)

    def get_investor_supply(
        self,
        code: str,
        days: int = 5,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        norm = str(code).zfill(6)
        sup = self._supply.get(norm)
        if sup:
            return dict(sup)
        # 과거 원격 수급 이력 미지원 날짜의 경우 거래량·가격 양봉 기반 보수 중립 대체치 제공
        df = self.get_daily_ohlcv(norm, bars=max(days + 1, 6), end_date=end_date or self.current_date)
        if len(df) >= 2:
            last_bull = float(df.iloc[-1]["close"]) > float(df.iloc[-1]["open"])
            return {
                "frgn_today": 1000 if last_bull else 0,
                "orgn_today": 1000 if last_bull else 0,
                "frgn_3d": 1000 if last_bull else 0,
                "orgn_3d": 1000 if last_bull else 0,
                "total_5d": 1000 if last_bull else 0,
                "pos_days_5d": 3 if last_bull else 1,
                "intraday_pending": False,
                "orgn_series": [],
            }
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

    def get_index_ohlcv(
        self,
        bars: int = 100,
        end_date: str | None = None,
        market: str = "KOSPI",
    ) -> pd.DataFrame:
        eff = end_date or self.current_date
        base_df = self._kosdaq_index_df if str(market or "KOSPI").upper() == "KOSDAQ" else self._index_df
        sub = base_df[base_df["date"] <= eff] if eff else base_df
        return sub.tail(bars).reset_index(drop=True)

    def get_top_gainers(
        self,
        top_n: int = 20,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        eff = end_date or self.current_date
        rows: list[dict[str, Any]] = []
        for code, df in self._ohlcv.items():
            sub = df[df["date"] <= eff] if eff else df
            if len(sub) < 2:
                continue
            last = sub.iloc[-1]
            if eff and str(last["date"]) != eff:
                continue
            prev_c = float(sub.iloc[-2]["close"])
            close_p = float(last["close"])
            chg = ((close_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
            tval = float(last.get("amount") or (close_p * float(last["volume"])))
            if close_p <= 0 or tval < settings.top20_min_trading_val_krw:
                continue
            meta = self._universe_meta.get(code) or {}
            rows.append(
                {
                    "code": code,
                    "name": meta.get("name", code),
                    "close": close_p,
                    "chg_pct": round(chg, 2),
                    "trading_val": tval,
                    "marcap": int(meta.get("marcap", 500_000_000_000)),
                }
            )
        rows.sort(key=lambda x: x["chg_pct"], reverse=True)
        return rows[: max(1, int(top_n))]

    def get_sector_map(self) -> dict[str, str]:
        return dict(self._sector_map)


def run_bootstrap_learning(
    provider: MarketDataProvider,
    storage: StorageRepo | None = None,
    lookback_days: int = 25,
    max_stocks: int = 60,
    dry_run: bool = False,
) -> dict[str, Any]:
    """최근 lookback_days 거래일을 워크포워드 리플레이해 BOOTSTRAP 학습 표본·패턴 큐·자동보정을 시드한다."""
    store = storage or default_repo
    need_bars = max(185, settings.ohlcv_bars + lookback_days + 15)

    index_df = get_market_index_ohlcv(provider, bars=min(200, need_bars), market="KOSPI")
    kosdaq_index_df = (
        get_market_index_ohlcv(provider, bars=min(200, need_bars), market="KOSDAQ")
        if settings.include_kosdaq
        else None
    )
    if index_df.empty or len(index_df) < 30:
        return {"days_replayed": 0, "signals_created": 0, "learning": {}}

    all_dates = index_df["date"].astype(str).tolist()
    replay_dates = all_dates[-max(5, int(lookback_days)) :]

    universe = provider.get_universe()[: max(10, int(max_stocks))]
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    supply_cache: dict[str, dict[str, Any]] = {}

    for item in universe:
        code = str(item["code"]).zfill(6)
        try:
            df = provider.get_daily_ohlcv(code, bars=need_bars)
            if df is not None and len(df) >= 70:
                ohlcv_cache[code] = df
        except Exception as e:
            logger.debug("bootstrap prefetch %s skipped: %s", code, e)

    slicer = _WalkForwardSliceProvider(
        universe_meta=universe,
        ohlcv_cache=ohlcv_cache,
        supply_cache=supply_cache,
        index_df=index_df,
        sector_map=provider.get_sector_map(),
        kosdaq_index_df=kosdaq_index_df,
    )

    total_created = 0
    for dt_str in replay_dates:
        slicer.set_date(dt_str)
        dt_obj = datetime.strptime(dt_str, "%Y-%m-%d").replace(hour=14, minute=30, tzinfo=KST)
        scan_res = run_scan(
            provider=slicer,
            storage=store,
            now_kst=dt_obj,
            dry_run=dry_run,
            min_trading_val_override=2_000_000_000,
            send_notifications=False,
            origin="BOOTSTRAP",
        )
        total_created += int(scan_res["stats"]["new_signals_saved"])

        if not dry_run:
            update_error_logs(provider=slicer, storage=store, as_of_date=dt_str, dry_run=False)
            analyze_top20(provider=slicer, storage=store, as_of_date=dt_str, dry_run=False)

    # 최종 날짜 기준 오답노트 및 AutoLearner 갱신
    final_date = replay_dates[-1]
    slicer.set_date(final_date)
    err_final = update_error_logs(provider=slicer, storage=store, as_of_date=final_date, dry_run=dry_run)
    learn_res = run_auto_learning(storage=store, as_of_date=final_date, dry_run=dry_run)

    return {
        "days_replayed": len(replay_dates),
        "stocks_loaded": len(ohlcv_cache),
        "signals_created": total_created,
        "error_updated": err_final["updated"],
        "error_completed": err_final["completed"],
        "learning": learn_res,
    }
