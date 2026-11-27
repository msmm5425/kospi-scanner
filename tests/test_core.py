from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config.settings import KST
from engines.error_log import update_error_logs
from engines.scanner import run_scan
from engines.top20_analyzer import analyze_top20
from engines.weekly_report import build_weekly_report
from indicators.technicals import compute_indicators
from notify.telegram_bot import format_signal_message
from providers.base import MarketDataProvider, is_eligible_common_stock
from risk.krx_tick import get_tick_size, next_tick_down, round_to_tick
from risk.rr_geometry import evaluate_rr_geometry
from storage.repo import StorageRepo


def _trading_dates(n: int = 160, end: str = "2026-11-20") -> list[str]:
    dates: list[str] = []
    d = datetime.strptime(end, "%Y-%m-%d").date()
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    dates.reverse()
    return dates


def _make_index_df(dates: list[str], weak: bool = False) -> pd.DataFrame:
    n = len(dates)
    closes = np.linspace(2500.0, 2550.0, n)
    if weak:
        closes[-5:] = [2540.0, 2530.0, 2515.0, 2500.0, 2440.0]
    df = pd.DataFrame({"date": dates, "close": closes})
    df["chg_pct"] = (df["close"].pct_change().fillna(0.0) * 100.0).round(2)
    return df


def _make_track_a_vcp_df(dates: list[str]) -> pd.DataFrame:
    """150일 합성 데이터: 정배열 + 직전 20일 대비 5일 변동폭 극수축 + 거래량 고갈 + 피벗 돌파."""
    n = len(dates)
    rows = []
    base = 20_000.0
    for i, dt in enumerate(dates):
        if i < n - 26:
            c = base + (i * 75.0)
            o = c * 0.992
            h = c * 1.014
            l = c * 0.986
            v = 100_000.0
        elif i < n - 6:
            # 직전 20일: 상대적으로 넓은 변동폭(~3.2%) + 피벗 고점 형성
            c = base + (i * 75.0)
            o = c * 0.99
            h = c * 1.018
            l = c * 0.986
            v = 110_000.0
        elif i < n - 1:
            # 직전 5일: 일중 변동폭 극수축(~1.0%) + 거래량 고갈(40% 이하)
            pivot_ref = base + ((n - 7) * 75.0)
            c = pivot_ref * 1.005
            o = c * 0.997
            h = c * 1.005
            l = c * 0.995
            v = 35_000.0
        else:
            # 당일: 15일 피벗 돌파 + 거래량 250% + 양봉 +4.2% (120일 신고가 블루스카이)
            prev_c = rows[-1]["close"]
            o = prev_c * 1.004
            c = round_to_tick(prev_c * 1.042)
            h = round_to_tick(c * 1.006)
            l = round_to_tick(o * 0.996)
            v = 280_000.0
        rows.append(
            {
                "date": dt,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
                "amount": float(c * v),
            }
        )
    return pd.DataFrame(rows)


def _make_track_b_pullback_df(dates: list[str]) -> pd.DataFrame:
    """기준봉(+11%, 3배 거래량) 출현 후 5일간 거래량 건조 눌림목 -> 당일 5MA 재돌파."""
    n = len(dates)
    rows = []
    anchor_idx = n - 6
    anchor_mid = 0.0
    anchor_close = 0.0
    for i, dt in enumerate(dates):
        if i < anchor_idx:
            c = 30_000.0 + i * 20.0
            o, h, l = c * 0.992, c * 1.012, c * 0.988
            v = 80_000.0
        elif i == anchor_idx:
            prev_c = rows[-1]["close"]
            o = round_to_tick(prev_c * 1.005)
            c = round_to_tick(prev_c * 1.11)
            h = round_to_tick(c * 1.01)
            l = round_to_tick(o * 0.995)
            v = 300_000.0
            anchor_mid = (o + c) / 2.0
            anchor_close = c
        elif i < n - 1:
            # 5일간 건전한 조정: 기준봉 중심값 & 20MA 위 밀착 지지, 거래량 30% 수준으로 건조
            c = round_to_tick(anchor_mid * 1.006)
            o = round_to_tick(c * 1.002)
            h = round_to_tick(o * 1.004)
            l = round_to_tick(anchor_mid * 1.001)
            v = 90_000.0
        else:
            prev_c = rows[-1]["close"]
            o = round_to_tick(prev_c * 1.002)
            c = round_to_tick(prev_c * 1.028)
            h = round_to_tick(c * 1.004)
            l = round_to_tick(o * 0.997)
            v = 180_000.0  # 전일 대비 200%
        rows.append(
            {"date": dt, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v), "amount": float(c * v)}
        )
    return pd.DataFrame(rows)


def _make_track_c_turnaround_df(dates: list[str]) -> pd.DataFrame:
    """120일 고점 대비 깊은 조정 후 바닥 박스권 + OBV 40일 신고가 선행 + 당일 60MA 거래량 돌파."""
    n = len(dates)
    rows = []
    for i, dt in enumerate(dates):
        if i < n - 80:
            # 과거 고점대 (42,000원 ~ 45,000원 -> 현재가 대비 충분히 높은 120일 고점)
            c = 42_000.0
            o, h, l = 41_500.0, 44_000.0, 41_000.0
            v = 60_000.0
        elif i < n - 41:
            # 하락 구간 (거래량 약함)
            c = 28_000.0
            o, h, l = 28_300.0, 28_600.0, 27_800.0
            v = 40_000.0
        elif i < n - 1:
            # 바닥 40일 박스권 (소폭 양봉일 거래량 우위로 OBV 선행 누적, 직전 고점 30,000)
            if i == n - 35:
                o, h, l, c, v = 28_000.0, 30_000.0, 27_900.0, 28_300.0, 50_000.0
            elif i % 2 == 1:
                o, h, l, c, v = 28_100.0, 28_500.0, 27_900.0, 28_400.0, 95_000.0
            else:
                o, h, l, c, v = 28_400.0, 28_500.0, 27_900.0, 28_050.0, 25_000.0
        else:
            # 당일: 40일 고점(30,000)은 아래이나 60MA 돌파 + 거래량 200%+
            prev_c = rows[-1]["close"]
            o = round_to_tick(prev_c * 1.003)
            c = round_to_tick(29_400.0)
            h = round_to_tick(29_600.0)
            l = round_to_tick(o * 0.996)
            v = 240_000.0
        rows.append(
            {"date": dt, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v), "amount": float(c * v)}
        )
    return pd.DataFrame(rows)


class MockProvider(MarketDataProvider):
    provider_name = "mock"

    def __init__(
        self,
        universe: list[dict[str, Any]],
        ohlcv_map: dict[str, pd.DataFrame],
        supply_map: dict[str, dict[str, Any]],
        index_df: pd.DataFrame,
        top_gainers: list[dict[str, Any]] | None = None,
        safety_overrides: dict[str, dict[str, Any]] | None = None,
    ):
        self.universe = universe
        self.ohlcv_map = ohlcv_map
        self.supply_map = supply_map
        self.index_df = index_df
        self.top_gainers = top_gainers or []
        self.safety_overrides = safety_overrides or {}

    def get_universe(self, min_cap: int | None = None, max_cap: int | None = None) -> list[dict[str, Any]]:
        return list(self.universe)

    def get_safety_status(self, code: str) -> dict[str, Any]:
        if code in self.safety_overrides:
            return self.safety_overrides[code]
        df = self.ohlcv_map[code]
        last = df.iloc[-1]
        prev = df.iloc[-2]
        chg = ((float(last["close"]) - float(prev["close"])) / float(prev["close"])) * 100.0
        return {
            "is_safe": True,
            "reject_reason": "",
            "price": float(last["close"]),
            "chg_pct": round(chg, 2),
            "trading_val": float(last["amount"]),
        }

    def get_daily_ohlcv(self, code: str, bars: int = 150, end_date: str | None = None) -> pd.DataFrame:
        df = self.ohlcv_map[code].copy()
        if end_date:
            df = df[df["date"] <= end_date]
        return df.tail(bars).reset_index(drop=True)

    def get_investor_supply(self, code: str, days: int = 5, end_date: str | None = None) -> dict[str, Any]:
        sup = self.supply_map.get(code)
        if callable(sup):
            return sup(end_date)
        return dict(sup or {"frgn_today": 0, "orgn_today": 0, "frgn_3d": 0, "orgn_3d": 0, "total_5d": 0, "pos_days_5d": 0, "intraday_pending": False})

    def get_index_ohlcv(self, bars: int = 100, end_date: str | None = None) -> pd.DataFrame:
        df = self.index_df.copy()
        if end_date:
            df = df[df["date"] <= end_date]
        return df.tail(bars).reset_index(drop=True)

    def get_top_gainers(self, top_n: int = 20, end_date: str | None = None) -> list[dict[str, Any]]:
        return list(self.top_gainers[:top_n])


def test_krx_tick_boundaries() -> None:
    assert get_tick_size(1_999) == 1
    assert get_tick_size(2_000) == 5
    assert get_tick_size(4_995) == 5
    assert get_tick_size(5_000) == 10
    assert get_tick_size(19_990) == 10
    assert get_tick_size(20_000) == 50
    assert get_tick_size(49_950) == 50
    assert get_tick_size(50_000) == 100
    assert get_tick_size(199_900) == 100
    assert get_tick_size(200_000) == 500
    assert get_tick_size(499_500) == 500
    assert get_tick_size(500_000) == 1_000
    assert next_tick_down(50_000, 1) == 49_950
    assert round_to_tick(42_513, "down") == 42_500


def test_common_stock_filter() -> None:
    assert is_eligible_common_stock("005930", "삼성전자") is True
    assert is_eligible_common_stock("047040", "대우건설") is True
    assert is_eligible_common_stock("005935", "삼성전자우") is False
    assert is_eligible_common_stock("451700", "삼성스팩9호") is False
    assert is_eligible_common_stock("330590", "롯데리츠") is False
    assert is_eligible_common_stock("122630", "KODEX 레버리지") is False


def test_rr_geometry_blue_sky_and_resistance_rejection() -> None:
    dates = _trading_dates(140)
    vcp_df = compute_indicators(_make_track_a_vcp_df(dates))
    entry = float(vcp_df.iloc[-1]["close"])
    inval_good = entry * 0.955  # -4.5%
    rr_ok = evaluate_rr_geometry(vcp_df, entry, inval_good, "TRACK_A_VCP")
    assert rr_ok["passed"] is True
    assert rr_ok["is_blue_sky"] is True
    assert rr_ok["projected_rr"] >= 3.0

    # 리스크 초과 (-7.0% > 5.5%)
    rr_wide = evaluate_rr_geometry(vcp_df, entry, entry * 0.93, "TRACK_A_VCP")
    assert rr_wide["passed"] is False

    # 근접 저항 존재 시 탈락 (직전 20일 내 2배 거래량 고점이 +6% 위치에 존재)
    blocked_df = _make_track_a_vcp_df(dates)
    blocked_df.loc[len(blocked_df) - 30, "volume"] = 500_000.0
    blocked_df.loc[len(blocked_df) - 30, "high"] = entry * 1.06
    blocked_ind = compute_indicators(blocked_df)
    rr_blocked = evaluate_rr_geometry(blocked_ind, entry, inval_good, "TRACK_A_VCP")
    assert rr_blocked["passed"] is False
    assert rr_blocked["clear_upside_pct"] < 10.0

    # Track B는 최근 기준봉 고점은 무시하되, 기준봉 이전 구간의 2배 거래량 매물대는 차단 확인
    pb_df = _make_track_b_pullback_df(dates)
    anchor_idx = len(pb_df) - 6
    pb_entry = float(pb_df.iloc[-1]["close"])
    pb_inval = pb_entry * 0.965
    pb_rr_ok = evaluate_rr_geometry(compute_indicators(pb_df), pb_entry, pb_inval, "TRACK_B_PULLBACK", impulse_start_idx=anchor_idx)
    assert pb_rr_ok["passed"] is True
    assert pb_rr_ok["is_blue_sky"] is True

    pb_df_res = pb_df.copy()
    pb_df_res.loc[anchor_idx - 25, "volume"] = 500_000.0
    pb_df_res.loc[anchor_idx - 25, "high"] = pb_entry * 1.06
    pb_rr_blocked = evaluate_rr_geometry(compute_indicators(pb_df_res), pb_entry, pb_inval, "TRACK_B_PULLBACK", impulse_start_idx=anchor_idx)
    assert pb_rr_blocked["passed"] is False
    assert pb_rr_blocked["clear_upside_pct"] < 10.0


def test_scanner_tracks_cooldown_and_weak_regime(tmp_path) -> None:
    dates = _trading_dates(160, end="2026-11-20")
    df_a = _make_track_a_vcp_df(dates)
    df_b = _make_track_b_pullback_df(dates)
    df_c = _make_track_c_turnaround_df(dates)

    # Track C는 상단 공간 120일 고점(44,000원 vs ~29,400원 = +49%) 열려 있어 1:3 충분
    universe = [
        {"code": "100010", "name": "알파테스트", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": float(df_a.iloc[-1]["close"]), "chg_pct": 4.5},
        {"code": "100020", "name": "베타테스트", "marcap": 400_000_000_000, "trading_val": float(df_b.iloc[-1]["amount"]), "price": float(df_b.iloc[-1]["close"]), "chg_pct": 4.5},
        {"code": "100030", "name": "감마테스트", "marcap": 300_000_000_000, "trading_val": float(df_c.iloc[-1]["amount"]), "price": float(df_c.iloc[-1]["close"]), "chg_pct": 4.5},
    ]
    sup_good = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}
    provider = MockProvider(
        universe=universe,
        ohlcv_map={"100010": df_a, "100020": df_b, "100030": df_c},
        supply_map={"100010": sup_good, "100020": sup_good, "100030": sup_good},
        index_df=_make_index_df(dates, weak=True),
    )

    store = StorageRepo(data_dir=tmp_path)
    now_intraday = datetime(2026, 11, 20, 14, 0, tzinfo=KST)
    res = run_scan(provider=provider, storage=store, now_kst=now_intraday, send_notifications=False)
    tracks_found = {s["code"]: s["track"] for s in res["signals"]}
    assert tracks_found["100010"] == "TRACK_A_VCP"
    assert tracks_found["100020"] == "TRACK_B_PULLBACK"
    assert tracks_found["100030"] == "TRACK_C_TURNAROUND"

    # 약세장 태깅 및 텔레그램 배너 검증
    sig_a = next(s for s in res["signals"] if s["code"] == "100010")
    assert sig_a["snapshot"]["market_regime"] == "WEAK"
    msg = format_signal_message(sig_a)
    assert "[지수 약세 주의]" in msg

    # 즉시 재스캔 시 5거래일 쿨다운 작동 확인
    res2 = run_scan(provider=provider, storage=store, now_kst=now_intraday, send_notifications=False)
    assert res2["stats"]["cooldown_skipped"] == 3
    assert res2["stats"]["matched_count"] == 0


def test_error_log_top20_and_weekly_report(tmp_path) -> None:
    base_dates = _trading_dates(165, end="2026-11-27")
    sig_date = base_dates[-11]  # 정확히 10거래일 뒤까지 진행
    store = StorageRepo(data_dir=tmp_path)

    # 1. 승자 시그널 (+25%, 5.2R) + 패자 시그널 (D+1 무효화 이탈 + 지수 급락 + 수급 반전)
    store.upsert_signal(
        {
            "signal_id": f"{sig_date.replace('-', '')}_100010_A",
            "signal_date": sig_date,
            "code": "100010",
            "name": "성공주",
            "track": "TRACK_A_VCP",
            "entry_price": 10_000,
            "rr_geometry": {"invalidation_ref_price": 9_500, "risk_pct": 5.0, "nearest_resistance_price": None, "clear_upside_pct": 30.0, "projected_rr": 6.0},
            "snapshot": {"upper_wick_ratio": 0.1, "market_regime": "NORMAL"},
            "tracking": {"status": "ACTIVE", "days_elapsed": 0, "mfe_pct": 0.0, "mae_pct": 0.0, "max_r_multiple": 0.0, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
            "error_tags": [],
        }
    )
    store.upsert_signal(
        {
            "signal_id": f"{sig_date.replace('-', '')}_100020_B",
            "signal_date": sig_date,
            "code": "100020",
            "name": "실패주",
            "track": "TRACK_B_PULLBACK",
            "entry_price": 10_000,
            "rr_geometry": {"invalidation_ref_price": 9_600, "risk_pct": 4.0, "nearest_resistance_price": 11_500, "clear_upside_pct": 15.0, "projected_rr": 3.75},
            "snapshot": {"upper_wick_ratio": 0.35, "market_regime": "WEAK"},
            "tracking": {"status": "ACTIVE", "days_elapsed": 0, "mfe_pct": 0.0, "mae_pct": 0.0, "max_r_multiple": 0.0, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
            "error_tags": [],
        }
    )

    win_rows = []
    lose_rows = []
    for i, dt in enumerate(base_dates):
        if dt <= sig_date:
            win_rows.append({"date": dt, "open": 9_800, "high": 10_100, "low": 9_700, "close": 10_000, "volume": 100_000, "amount": 1e9})
            lose_rows.append({"date": dt, "open": 9_900, "high": 10_100, "low": 9_800, "close": 10_000, "volume": 100_000, "amount": 1e9})
        else:
            step = i - (len(base_dates) - 11)
            win_rows.append({"date": dt, "open": 10_200, "high": 12_500 if step == 4 else 11_800, "low": 9_800, "close": 12_000, "volume": 150_000, "amount": 1.8e9})
            # 실패주: D+1부터 9,500원 이탈 + 마지막 날 갭상승 TOP20 시나리오 아닌 하락 마감
            lose_rows.append({"date": dt, "open": 9_900, "high": 10_000, "low": 9_400, "close": 9_500, "volume": 120_000, "amount": 1.1e9})

    # 놓친 급등주 1개(100040): 당일 갭상승 +22% + 거래량 3배
    miss_df = _make_track_a_vcp_df(base_dates[:-1])
    last_miss_c = float(miss_df.iloc[-1]["close"])
    miss_df = pd.concat(
        [
            miss_df,
            pd.DataFrame(
                [
                    {
                        "date": base_dates[-1],
                        "open": round_to_tick(last_miss_c * 1.05),
                        "high": round_to_tick(last_miss_c * 1.24),
                        "low": round_to_tick(last_miss_c * 1.04),
                        "close": round_to_tick(last_miss_c * 1.22),
                        "volume": 350_000.0,
                        "amount": 6e9,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    idx_df = _make_index_df(base_dates, weak=False)
    # D+1 지수 -2.0% 하락 주입 -> INDEX_DRAG 유발
    d1_idx = len(base_dates) - 10
    idx_df.loc[d1_idx, "chg_pct"] = -2.1

    def supply_fn(end_date=None):
        return {"frgn_today": -20_000, "orgn_today": -10_000, "frgn_3d": -30_000, "orgn_3d": -20_000, "total_5d": 10_000, "pos_days_5d": 3, "intraday_pending": False, "orgn_series": [1, 2, 3, 4, 5]}

    provider = MockProvider(
        universe=[],
        ohlcv_map={"100010": pd.DataFrame(win_rows), "100020": pd.DataFrame(lose_rows), "100040": miss_df},
        supply_map={"100010": supply_fn, "100020": supply_fn, "100040": supply_fn},
        index_df=idx_df,
        top_gainers=[
            {"code": "100010", "name": "성공주", "close": 12_000, "chg_pct": 15.0, "trading_val": 5e9, "marcap": 5e11},
            {"code": "100040", "name": "놓친급등주", "close": float(miss_df.iloc[-1]["close"]), "chg_pct": 22.0, "trading_val": 6e9, "marcap": 4e11},
        ],
    )

    err_res = update_error_logs(provider=provider, storage=store, as_of_date=base_dates[-1])
    assert err_res["completed"] == 2
    sigs = {s["code"]: s for s in store.get_signals()}
    assert sigs["100010"]["tracking"]["hit_20pct"] is True
    assert sigs["100010"]["tracking"]["hit_3r"] is True
    assert "RISK_BREACHED_EARLY" in sigs["100020"]["error_tags"]
    assert "FAKEOUT_WICK" in sigs["100020"]["error_tags"]
    assert "INDEX_DRAG" in sigs["100020"]["error_tags"]
    assert "SUPPLY_REVERSAL" in sigs["100020"]["error_tags"]

    top20_res = analyze_top20(provider=provider, storage=store, as_of_date=base_dates[-1])
    assert top20_res["summary"]["hit_count"] >= 0
    assert top20_res["summary"]["miss_count"] >= 1

    weekly = build_weekly_report(storage=store, as_of_date=base_dates[-1])
    assert weekly["total_signal_count"] == 2
    assert weekly["phase4_distribution"]["recommended_sl_mae_85pct"] is not None
    assert len(weekly["top3_errors"]) >= 1
