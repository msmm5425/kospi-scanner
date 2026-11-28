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
        kosdaq_index_df: pd.DataFrame | None = None,
    ):
        self.universe = universe
        self.ohlcv_map = ohlcv_map
        self.supply_map = supply_map
        self.index_df = index_df
        self.kosdaq_index_df = kosdaq_index_df if kosdaq_index_df is not None else index_df
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

    def get_index_ohlcv(self, bars: int = 100, end_date: str | None = None, market: str = "KOSPI") -> pd.DataFrame:
        src = self.kosdaq_index_df if str(market or "KOSPI").upper() == "KOSDAQ" else self.index_df
        df = src.copy()
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
    assert is_eligible_common_stock("487240", "KODEX AI전력핵심설비") is False
    assert is_eligible_common_stock("466920", "SOL 조선TOP3플러스") is False
    assert is_eligible_common_stock("360750", "TIGER 미국S&P500") is False


def test_premarket_early_window_classification() -> None:
    from config.settings import classify_market_session

    assert classify_market_session(datetime(2026, 11, 27, 7, 50, tzinfo=KST)) == "PREMARKET"
    assert classify_market_session(datetime(2026, 11, 27, 9, 5, tzinfo=KST)) == "REGULAR"
    assert classify_market_session(datetime(2026, 11, 27, 16, 40, tzinfo=KST)) == "AFTERMARKET"


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


def test_kis_provider_payload_parsing() -> None:
    from providers.kis_provider import KISProvider

    kis = KISProvider()
    kis.app_key = "TEST_KEY_00"
    kis.app_secret = "TEST_SEC"

    # 1. inquire-price 실전 응답 파싱 (temp_stop_yn / iscd_stat_cls_code 포함)
    kis._request_get = lambda path, tr_id, params: {
        "rt_cd": "0",
        "output": {
            "stck_prpr": "179000",
            "prdy_ctrt": "4.50",
            "acml_tr_pbmn": "1480984259100",
            "iscd_stat_cls_code": "55",
            "temp_stop_yn": "N",
            "sltr_yn": "N",
            "mrkt_warn_cls_code": "00",
        },
    }
    ok_safety = kis.get_safety_status("005930")
    assert ok_safety["is_safe"] is True
    assert ok_safety["price"] == 179000.0

    kis._request_get = lambda path, tr_id, params: {
        "rt_cd": "0",
        "output": {
            "stck_prpr": "5000",
            "prdy_ctrt": "0",
            "acml_tr_pbmn": "0",
            "iscd_stat_cls_code": "51",
            "temp_stop_yn": "Y",
            "sltr_yn": "N",
            "mrkt_warn_cls_code": "00",
        },
    }
    bad_safety = kis.get_safety_status("000010")
    assert bad_safety["is_safe"] is False
    assert "거래정지" in bad_safety["reject_reason"]
    assert "관리종목" in bad_safety["reject_reason"]

    # 2. inquire-investor 장중 빈 문자열("") 행 제외 후 3일/5일 합산 확인
    kis._request_get = lambda path, tr_id, params: {
        "rt_cd": "0",
        "output": [
            {"stck_bsop_date": "20261127", "frgn_ntby_qty": "", "orgn_ntby_qty": ""},
            {"stck_bsop_date": "20261126", "frgn_ntby_qty": "100", "orgn_ntby_qty": "200"},
            {"stck_bsop_date": "20261125", "frgn_ntby_qty": "50", "orgn_ntby_qty": "50"},
            {"stck_bsop_date": "20261124", "frgn_ntby_qty": "10", "orgn_ntby_qty": "-5"},
        ],
    }
    sup = kis.get_investor_supply("005930", days=5)
    assert sup["intraday_pending"] is True
    assert sup["frgn_3d"] == 160
    assert sup["orgn_3d"] == 245
    assert sup["pos_days_5d"] == 3

    # 3. inquire-daily-indexchartprice (pct_change 계산) & ranking/fluctuation(acml_tr_pbmn 부재 시 close*vol)
    kis._request_get = lambda path, tr_id, params: {
        "rt_cd": "0",
        "output2": [
            {"stck_bsop_date": "20261127", "bstp_nmix_prpr": "4898.73"},
            {"stck_bsop_date": "20261126", "bstp_nmix_prpr": "4711.21"},
        ],
    }
    idx = kis.get_index_ohlcv(bars=10, end_date="2026-11-27")
    assert list(idx["date"]) == ["2026-11-26", "2026-11-27"]
    assert round(float(idx.iloc[-1]["chg_pct"]), 2) == 3.98

    kis._request_get = lambda path, tr_id, params: {
        "rt_cd": "0",
        "output": [
            {"stck_shrn_iscd": "084670", "hts_kor_isnm": "동양고속", "stck_prpr": "36900", "prdy_ctrt": "10.48", "acml_vol": "2258744"},
            {"stck_shrn_iscd": "181710", "hts_kor_isnm": "NHN", "stck_prpr": "61300", "prdy_ctrt": "17.43", "acml_vol": "433182"},
        ],
    }
    gainers = kis.get_top_gainers(top_n=10)
    assert len(gainers) == 2
    assert gainers[0]["code"] == "181710"
    assert gainers[0]["trading_val"] > 0


def test_auto_learner_adaptation_track_d_and_phase4(tmp_path) -> None:
    from engines.auto_learner import run_auto_learning

    store = StorageRepo(data_dir=tmp_path)
    # 18개 승리 + 14개 실패 = 32개 COMPLETED 표본 구축 (오답 편향 포함)
    for i in range(18):
        dt = f"2026-12-{(i % 20) + 1:02d}"
        store.upsert_signal(
            {
                "signal_id": f"20261201_{100000 + i * 10:06d}_A",
                "signal_date": dt,
                "origin": "LIVE",
                "code": f"{100000 + i * 10:06d}",
                "name": f"승자{i}",
                "track": "TRACK_A_VCP",
                "entry_price": 20_000,
                "rr_geometry": {"invalidation_ref_price": 19_000, "risk_pct": 5.0, "nearest_resistance_price": None, "clear_upside_pct": 30.0, "projected_rr": 6.0},
                "snapshot": {"upper_wick_ratio": 0.12, "market_regime": "NORMAL"},
                "tracking": {"status": "COMPLETED", "days_elapsed": 10, "mfe_pct": 18.0 + (i % 10), "mae_pct": -2.5 - (i % 3) * 0.6, "max_r_multiple": 3.6, "hit_10pct": True, "hit_20pct": i % 2 == 0, "hit_30pct": False, "hit_3r": True, "closes": {}},
                "error_tags": [],
            }
        )
    for j in range(14):
        is_weak = j < 7
        tags = ["FAKEOUT_WICK", "RESISTANCE_REJECT", "RISK_BREACHED_EARLY"]
        if is_weak:
            tags.append("INDEX_DRAG")
        store.upsert_signal(
            {
                "signal_id": f"20261202_{200000 + j * 10:06d}_B",
                "signal_date": f"2026-12-{(j % 20) + 1:02d}",
                "origin": "LIVE",
                "code": f"{200000 + j * 10:06d}",
                "name": f"패자{j}",
                "track": "TRACK_B_PULLBACK",
                "entry_price": 30_000,
                "rr_geometry": {"invalidation_ref_price": 28_600, "risk_pct": 4.67, "nearest_resistance_price": 34_000, "clear_upside_pct": 13.3, "projected_rr": 3.1},
                "snapshot": {"upper_wick_ratio": 0.36, "market_regime": "WEAK" if is_weak else "NORMAL"},
                "tracking": {"status": "COMPLETED", "days_elapsed": 10, "mfe_pct": 1.8, "mae_pct": -5.5, "max_r_multiple": 0.39, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
                "error_tags": tags,
            }
        )

    # pattern_queue에 GAP_BREAKAWAY 4회 (1:3 R:R 가능 3회) 적재 + TOP20 Track A 거래량고갈 5회 적재
    store.append_pattern_entries(
        [
            {"date": f"2026-12-0{k}", "code": f"3000{k}0", "name": f"갭{k}", "chg_pct": 14.0, "tags": ["GAP_BREAKAWAY"], "rr_viable_1to3": k <= 3, "pre_entry_rr": 3.8}
            for k in range(1, 5)
        ]
    )
    store.upsert_top20_day(
        {
            "date": "2026-12-20",
            "misses": [
                {"code": f"4000{k}0", "name": f"미스{k}", "diagnosis": "D-1 Track A 근접 탈락 [거래량 고갈 64% (기준 60% 이하)]"}
                for k in range(5)
            ],
        }
    )

    from config.settings import settings

    res = run_auto_learning(storage=store, as_of_date="2026-12-27")
    ap = res["adaptive_params"]
    assert res["cohort_mode"] == "LIVE_ONLY"
    assert ap["max_upper_wick_ratio"] < settings.max_upper_wick_ratio
    assert ap["min_projected_rr"] > settings.min_projected_rr
    assert ap["weak_regime_min_rr"] > settings.weak_regime_min_rr
    assert ap["track_risk_mult"]["TRACK_B_PULLBACK"] < 1.0
    assert ap["track_a_dry_vol_ratio_max"] > settings.track_a_dry_vol_ratio_max
    assert "GAP_BREAKAWAY" in res["promoted_track_d_patterns"]
    assert res["just_activated_phase4"] is True
    assert res["phase4_policy"]["phase4_active"] is True
    assert -6.5 <= res["phase4_policy"]["default_sl_pct"] <= -3.0

    # 승격된 Track D가 실전에서 8건 연속 실패 시 자동 강등 확인
    for d in range(8):
        store.upsert_signal(
            {
                "signal_id": f"20261228_{500000 + d * 10:06d}_D",
                "signal_date": "2026-12-28",
                "origin": "LIVE",
                "code": f"{500000 + d * 10:06d}",
                "name": f"D패자{d}",
                "track": "TRACK_D_GAP_BREAKAWAY",
                "entry_price": 15_000,
                "rr_geometry": {"invalidation_ref_price": 14_300, "risk_pct": 4.67, "nearest_resistance_price": None, "clear_upside_pct": 30.0, "projected_rr": 6.4},
                "snapshot": {"upper_wick_ratio": 0.1, "market_regime": "NORMAL"},
                "tracking": {"status": "COMPLETED", "days_elapsed": 10, "mfe_pct": 2.0, "mae_pct": -5.0, "max_r_multiple": 0.43, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
                "error_tags": ["FAKEOUT_WICK"],
            }
        )
    res2 = run_auto_learning(storage=store, as_of_date="2026-12-28")
    assert "GAP_BREAKAWAY" not in res2["promoted_track_d_patterns"]


def test_track_d_execution_and_walkforward_bootstrap(tmp_path) -> None:
    from engines.bootstrap_learner import run_bootstrap_learning

    dates = _trading_dates(190, end="2026-11-27")
    df_a = _make_track_a_vcp_df(dates)

    # GAP_BREAKAWAY 전용 합성 차트 (A/B/C는 미충족, 갭상승 +4% & 강한 양봉)
    gap_rows = []
    for i, dt in enumerate(dates):
        if i < len(dates) - 1:
            c = 20_000.0 + (i * 20.0)
            gap_rows.append({"date": dt, "open": c * 0.996, "high": c * 1.008, "low": c * 0.992, "close": c, "volume": 80_000.0, "amount": c * 80_000.0})
        else:
            prev_c = gap_rows[-1]["close"]
            o = round_to_tick(prev_c * 1.035)
            c = round_to_tick(prev_c * 1.075)
            h = round_to_tick(c * 1.005)
            l = round_to_tick(prev_c * 1.030)
            gap_rows.append({"date": dt, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": 240_000.0, "amount": float(c * 240_000.0)})
    df_gap = pd.DataFrame(gap_rows)

    sup = {"frgn_today": 10_000, "orgn_today": 12_000, "frgn_3d": 25_000, "orgn_3d": 25_000, "total_5d": 50_000, "pos_days_5d": 4, "intraday_pending": False, "orgn_series": [5000] * 5}
    provider = MockProvider(
        universe=[{"code": "100050", "name": "갭돌파주", "marcap": 500_000_000_000, "trading_val": float(df_gap.iloc[-1]["amount"]), "price": float(df_gap.iloc[-1]["close"]), "chg_pct": 7.5}],
        ohlcv_map={"100050": df_gap},
        supply_map={"100050": sup},
        index_df=_make_index_df(dates, weak=False),
    )

    store = StorageRepo(data_dir=tmp_path)
    lp = store.load_learning_profile()
    lp["promoted_track_d_patterns"] = ["GAP_BREAKAWAY"]
    lp["phase4_policy"] = {
        "phase4_active": True,
        "activated_date": "2026-11-27",
        "default_sl_pct": -4.2,
        "sl_pct_by_track": {},
        "tp_levels_pct": [10.0, 20.0, 30.0],
        "breakeven_trigger_pct": 10.0,
    }
    store.save_learning_profile(lp)

    res = run_scan(provider=provider, storage=store, now_kst=datetime(2026, 11, 27, 14, 0, tzinfo=KST), send_notifications=False)
    assert len(res["signals"]) == 1
    sig = res["signals"][0]
    assert sig["track"] == "TRACK_D_GAP_BREAKAWAY"
    assert sig["phase4_sl_tp"]["sl_pct"] == -4.2
    msg = format_signal_message(sig)
    assert "Phase 4" in msg

    # 워크포워드 부트스트랩 실행 검증
    boot_store = StorageRepo(data_dir=tmp_path / "boot")
    boot_provider = MockProvider(
        universe=[{"code": "100010", "name": "알파테스트", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": float(df_a.iloc[-1]["close"]), "chg_pct": 4.2}],
        ohlcv_map={"100010": df_a},
        supply_map={"100010": sup},
        index_df=_make_index_df(dates, weak=False),
    )
    boot_res = run_bootstrap_learning(provider=boot_provider, storage=boot_store, lookback_days=15, max_stocks=10)
    assert boot_res["days_replayed"] == 15
    assert boot_res["learning"]["date"] == dates[-1]


def test_premarket_aftermarket_regular_anchor_and_loop(tmp_path) -> None:
    from engines.scanner import run_loop_mode

    dates_t_minus_1 = _trading_dates(160, end="2026-11-26")
    df_a = _make_track_a_vcp_df(dates_t_minus_1)
    reg_close = float(df_a.iloc[-1]["close"])
    sup = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}
    uni = [{"code": "100010", "name": "알파테스트", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": reg_close, "chg_pct": 4.2}]

    # 1. 프리장(08:20 KST): 장외 소폭 +1.0% 상승 시 정규장 종가 기준 타점으로 포착 + 별도 표기 확인
    p_pre = MockProvider(uni, {"100010": df_a}, {"100010": sup}, _make_index_df(dates_t_minus_1))
    p_pre.get_session_quote = lambda code, session="REGULAR": {
        "price": reg_close * 1.01,
        "chg_pct": 1.0,
        "trading_val": 5e8,
        "market_code": "UN",
    }
    store_pre = StorageRepo(data_dir=tmp_path / "pre")
    res_pre = run_scan(
        provider=p_pre,
        storage=store_pre,
        now_kst=datetime(2026, 11, 27, 8, 20, tzinfo=KST),
        send_notifications=False,
    )
    assert len(res_pre["signals"]) == 1
    sig_pre = res_pre["signals"][0]
    assert sig_pre["session"] == "PREMARKET"
    assert sig_pre["signal_date"] == "2026-11-27"
    assert sig_pre["entry_price"] == round_to_tick(reg_close)
    assert sig_pre["max_entry_price"] >= sig_pre["entry_price"]
    assert sig_pre["session_info"]["ext_market_code"] == "UN"
    msg_pre = format_signal_message(sig_pre)
    assert "프리장 포착" in msg_pre
    assert "정규장 유효 진입 구간" in msg_pre
    assert "장외 참고가(UN)" in msg_pre

    # 2. 프리장 과도 갭상승(+9.0% > max_entry_price 초과) 시 추격 차단 확인
    p_gap = MockProvider(uni, {"100010": df_a}, {"100010": sup}, _make_index_df(dates_t_minus_1))
    p_gap.get_session_quote = lambda code, session="REGULAR": {
        "price": reg_close * 1.09,
        "chg_pct": 9.0,
        "trading_val": 8e8,
        "market_code": "UN",
    }
    res_gap = run_scan(
        provider=p_gap,
        storage=StorageRepo(data_dir=tmp_path / "pre_gap"),
        now_kst=datetime(2026, 11, 27, 8, 25, tzinfo=KST),
        send_notifications=False,
    )
    assert len(res_gap["signals"]) == 0
    assert res_gap["stats"]["rr_rejected"] == 1

    # 3. 애프터장(18:10 KST): 당일 정규장 종가 기준 타점 + 애프터장 별도 표기 확인
    dates_today = _trading_dates(160, end="2026-11-27")
    df_today = _make_track_a_vcp_df(dates_today)
    reg_today_close = float(df_today.iloc[-1]["close"])
    uni_today = [{"code": "100010", "name": "알파테스트", "marcap": 600_000_000_000, "trading_val": float(df_today.iloc[-1]["amount"]), "price": reg_today_close, "chg_pct": 4.2}]
    p_aft = MockProvider(uni_today, {"100010": df_today}, {"100010": sup}, _make_index_df(dates_today))
    p_aft.get_session_quote = lambda code, session="REGULAR": {
        "price": reg_today_close * 1.005,
        "chg_pct": 0.5,
        "trading_val": 3e8,
        "market_code": "UN",
    }
    res_aft = run_scan(
        provider=p_aft,
        storage=StorageRepo(data_dir=tmp_path / "aft"),
        now_kst=datetime(2026, 11, 27, 18, 10, tzinfo=KST),
        send_notifications=False,
    )
    assert len(res_aft["signals"]) == 1
    assert res_aft["signals"][0]["session"] == "AFTERMARKET"
    assert res_aft["signals"][0]["entry_price"] == round_to_tick(reg_today_close)
    assert "애프터장 포착" in format_signal_message(res_aft["signals"][0])

    # 4. 연속 루프 모드(중복 방지 & 휴장일 조기 종료) 검증
    loop_store = StorageRepo(data_dir=tmp_path / "loop")
    p_aft.is_trading_day = lambda _d=None: True
    loop_res = run_loop_mode(
        provider=p_aft,
        storage=loop_store,
        interval_sec=0,
        max_minutes=1,
        end_kst="23:59",
        send_notifications=False,
        force=True,
        max_iterations=2,
    )
    assert loop_res["cycles"] == 2
    assert loop_res["total_new_signals"] == 1
    assert loop_res["skipped_holiday"] is False

    p_aft.is_trading_day = lambda _d=None: False
    hol_res = run_loop_mode(
        provider=p_aft,
        storage=loop_store,
        interval_sec=0,
        max_minutes=1,
        send_notifications=False,
        force=False,
    )
    assert hol_res["skipped_holiday"] is True
    assert hol_res["cycles"] == 0


def test_closing_briefing_validation_and_ranking(tmp_path) -> None:
    from engines.closing_picker import build_and_send_closing_briefing
    from notify.telegram_bot import format_closing_briefing_message

    dates = _trading_dates(160, end="2026-11-27")
    df_good = _make_track_a_vcp_df(dates)
    good_close = float(df_good.iloc[-1]["close"])

    # 장중 포착됐지만 15:00에 긴 윗꼬리 음봉으로 꺾인 종목 생성
    df_faded = _make_track_a_vcp_df(dates)
    prev_c = float(df_faded.iloc[-2]["close"])
    df_faded.loc[df_faded.index[-1], "open"] = prev_c * 1.02
    df_faded.loc[df_faded.index[-1], "high"] = prev_c * 1.08
    df_faded.loc[df_faded.index[-1], "close"] = prev_c * 0.995
    df_faded.loc[df_faded.index[-1], "low"] = prev_c * 0.99

    sup = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}
    provider = MockProvider(
        universe=[],
        ohlcv_map={"100010": df_good, "100020": df_faded},
        supply_map={"100010": sup, "100020": sup},
        index_df=_make_index_df(dates, weak=False),
    )

    store = StorageRepo(data_dir=tmp_path)
    store.upsert_signal(
        {
            "signal_id": "20261127_100010_A",
            "signal_date": "2026-11-27",
            "origin": "LIVE",
            "code": "100010",
            "name": "종가안착주",
            "track": "TRACK_A_VCP",
            "track_label": "Track A (VCP 타이트 돌파)",
            "entry_price": round_to_tick(good_close),
            "max_entry_price": round_to_tick(good_close * 1.03),
            "rr_geometry": {
                "invalidation_ref_price": round_to_tick(good_close * 0.955, "down"),
                "risk_pct": 4.5,
                "nearest_resistance_price": None,
                "clear_upside_pct": 30.0,
                "projected_rr": 6.67,
                "is_blue_sky": True,
                "max_entry_price": round_to_tick(good_close * 1.03),
            },
            "snapshot": {"marcap_uk": 6000, "upper_wick_ratio": 0.1, "market_regime": "NORMAL"},
            "tracking": {"status": "ACTIVE", "days_elapsed": 0, "mfe_pct": 0.0, "mae_pct": 0.0, "max_r_multiple": 0.0, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
            "error_tags": [],
        }
    )
    store.upsert_signal(
        {
            "signal_id": "20261127_100020_B",
            "signal_date": "2026-11-27",
            "origin": "LIVE",
            "code": "100020",
            "name": "윗꼬리꺾임주",
            "track": "TRACK_B_PULLBACK",
            "track_label": "Track B (기준봉 눌림목 재점화)",
            "entry_price": round_to_tick(prev_c * 1.04),
            "max_entry_price": round_to_tick(prev_c * 1.07),
            "rr_geometry": {
                "invalidation_ref_price": round_to_tick(prev_c * 0.99, "down"),
                "risk_pct": 4.8,
                "nearest_resistance_price": None,
                "clear_upside_pct": 30.0,
                "projected_rr": 6.25,
                "is_blue_sky": True,
                "max_entry_price": round_to_tick(prev_c * 1.07),
            },
            "snapshot": {"marcap_uk": 4000, "upper_wick_ratio": 0.1, "market_regime": "NORMAL"},
            "tracking": {"status": "ACTIVE", "days_elapsed": 0, "mfe_pct": 0.0, "mae_pct": 0.0, "max_r_multiple": 0.0, "hit_10pct": False, "hit_20pct": False, "hit_30pct": False, "hit_3r": False, "closes": {}},
            "error_tags": [],
        }
    )

    brief = build_and_send_closing_briefing(
        provider=provider,
        storage=store,
        now_kst=datetime(2026, 11, 27, 15, 2, tzinfo=KST),
        send_notifications=False,
    )
    assert brief["already_sent"] is False
    assert len(brief["qualified"]) == 1
    assert brief["qualified"][0]["code"] == "100010"
    assert len(brief["invalidated"]) == 1
    assert brief["invalidated"][0]["code"] == "100020"

    msg = format_closing_briefing_message(brief)
    assert "15:00 장 마감 매수 최종 브리핑" in msg
    assert "종가안착주" in msg
    assert "윗꼬리꺾임주" in msg

    # 동일 날짜 재호출 시 중복 발송 방지 확인
    brief2 = build_and_send_closing_briefing(
        provider=provider,
        storage=store,
        now_kst=datetime(2026, 11, 27, 15, 6, tzinfo=KST),
        send_notifications=False,
    )
    assert brief2["already_sent"] is True


def test_kis_daily_token_encryption_and_reuse(tmp_path, monkeypatch) -> None:
    from providers.kis_provider import KISProvider

    cache_file = tmp_path / "kis_token.json"
    monkeypatch.setenv("KIS_TOKEN_CACHE_PATH", str(cache_file))

    issue_calls = 0

    class _FakeResp:
        status_code = 200

        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class _FakeSession:
        def post(self, url: str, json: dict[str, Any] | None = None, timeout: float = 10.0):
            nonlocal issue_calls
            issue_calls += 1
            return _FakeResp({"access_token": f"TOK_{issue_calls}", "expires_in": 86400})

    p1 = KISProvider()
    p1.app_key = "APPKEY123456"
    p1.app_secret = "SUPER_SECRET_KEY"
    p1._session = _FakeSession()

    res1 = p1.ensure_daily_token(min_ttl_sec=1800.0)
    assert res1["source"] == "issued"
    assert res1["access_token"] == "TOK_1"
    assert issue_calls == 1
    raw_disk = cache_file.read_text(encoding="utf-8")
    assert "TOK_1" not in raw_disk  # 암호화 저장 확인

    # 새 KISProvider 인스턴스(다음 워크플로/프로세스)에서 1회 발급 토큰 재사용 확인
    p2 = KISProvider()
    p2.app_key = "APPKEY123456"
    p2.app_secret = "SUPER_SECRET_KEY"
    p2._session = _FakeSession()

    res2 = p2.ensure_daily_token(min_ttl_sec=48_600.0)
    assert res2["source"] == "cache"
    assert res2["access_token"] == "TOK_1"
    assert issue_calls == 1


def test_naver_and_kind_fallbacks(monkeypatch) -> None:
    from providers.kis_provider import KISProvider
    from providers.naver_fallback import (
        fetch_naver_kospi_universe,
        parse_kind_sector_html,
        parse_naver_fchart_xml,
    )

    class _FakeResp:
        def __init__(self, status_code: int, payload: Any):
            self.status_code = status_code
            self._payload = payload

        def json(self) -> Any:
            return self._payload

    class _FakeSession:
        def get(self, url: str, headers=None, timeout: float = 10.0):
            if "marketValue/KOSPI" in url:
                if "page=1&" in url:
                    return _FakeResp(
                        200,
                        {
                            "stocks": [
                                {
                                    "itemCode": "005930",
                                    "stockName": "삼성전자",
                                    "closePriceRaw": "178100",
                                    "fluctuationsRatio": "3.97",
                                    "accumulatedTradingVolumeRaw": "13716654",
                                    "accumulatedTradingValueRaw": "2414846000000",
                                    "marketValueRaw": "1041222220084800",
                                },
                                {
                                    "itemCode": "009470",
                                    "stockName": "삼화전기",
                                    "closePriceRaw": "28350",
                                    "fluctuationsRatio": "23.80",
                                    "accumulatedTradingVolumeRaw": "294361",
                                    "accumulatedTradingValueRaw": "8094000000",
                                    "marketValueRaw": "187501797000",
                                },
                                {
                                    "itemCode": "005935",
                                    "stockName": "삼성전자우",
                                    "closePriceRaw": "75000",
                                    "fluctuationsRatio": "1.2",
                                    "accumulatedTradingVolumeRaw": "100000",
                                    "accumulatedTradingValueRaw": "7500000000",
                                    "marketValueRaw": "60000000000000",
                                },
                            ]
                        },
                    )
                return _FakeResp(200, {"stocks": []})
            if "up/KOSPI" in url:
                return _FakeResp(
                    200,
                    {
                        "stocks": [
                            {
                                "itemCode": "009470",
                                "stockName": "삼화전기",
                                "closePriceRaw": "28350",
                                "fluctuationsRatio": "23.80",
                                "accumulatedTradingVolumeRaw": "294361",
                                "accumulatedTradingValueRaw": "8094000000",
                                "marketValueRaw": "187501797000",
                                "tradeStopType": {"name": "TRADING"},
                            }
                        ]
                    },
                )
            return _FakeResp(404, {})

    uni = fetch_naver_kospi_universe(session=_FakeSession())
    assert len(uni) == 1  # 삼성전자는 >20조 제외, 삼성전자우는 우선주 제외, 삼화전기 통과
    assert uni[0]["code"] == "009470"

    # KIS ranking/fluctuation 실패 시 Naver TOP 20 자동 폴백 확인
    kis = KISProvider()
    kis.app_key = "TEST_KEY"
    kis.app_secret = "TEST_SEC"
    kis._session = _FakeSession()
    kis._request_get = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("HTTP 500"))
    top = kis.get_top_gainers(top_n=10)
    assert len(top) == 1
    assert top[0]["code"] == "009470"
    assert top[0]["chg_pct"] == 23.8

    # fchart XML 파싱 (종목 + KOSPI 지수)
    xml_stock = """<?xml version="1.0" encoding="EUC-KR"?>
    <protocol><chartdata symbol="005930">
      <item data="20261126|172900|176700|170700|171500|13966982" />
      <item data="20261127|171200|179900|169000|178000|13717008" />
    </chartdata></protocol>"""
    df_s = parse_naver_fchart_xml(xml_stock, bars=10, is_index=False)
    assert list(df_s["date"]) == ["2026-11-26", "2026-11-27"]
    assert float(df_s.iloc[-1]["close"]) == 178000.0

    # KIND 업종 HTML 파싱
    kind_html = """<table>
      <tr><th>회사명</th><th>시장구분</th><th>종목코드</th><th>업종</th></tr>
      <tr><td>삼성전자</td><td>유가</td><td>005930</td><td>통신 및 방송 장비 제조업</td></tr>
      <tr><td>이상특수</td><td>유가</td><td>0220W0</td><td>기타</td></tr>
    </table>"""
    sec_map = parse_kind_sector_html(kind_html)
    assert sec_map == {"005930": "통신 및 방송 장비 제조업"}


def test_naver_supply_and_scan_diagnostic_summary(tmp_path) -> None:
    from notify.telegram_bot import format_scan_diagnostic_summary
    from providers.naver_fallback import parse_naver_integration_supply

    sup = parse_naver_integration_supply(
        {
            "dealTrendInfos": [
                {"bizdate": "20261127", "foreignerPureBuyQuant": "+1,200", "organPureBuyQuant": "+300"},
                {"bizdate": "20261126", "foreignerPureBuyQuant": "-100", "organPureBuyQuant": "+400"},
                {"bizdate": "20261125", "foreignerPureBuyQuant": "+50", "organPureBuyQuant": "-10"},
            ]
        },
        days=5,
        end_date="2026-11-27",
    )
    assert sup is not None
    assert sup["frgn_today"] == 1200
    assert sup["frgn_3d"] == 1150
    assert sup["orgn_3d"] == 690
    assert sup["pos_days_5d"] == 3

    # 수급만 부족해 근접 탈락한 종목이 near_misses 및 진단 요약에 포함되는지 확인
    dates = _trading_dates(160, end="2026-11-27")
    df_a = _make_track_a_vcp_df(dates)
    sup_bad = {"frgn_today": -5_000, "orgn_today": -2_000, "frgn_3d": -10_000, "orgn_3d": -5_000, "total_5d": -15_000, "pos_days_5d": 0, "intraday_pending": False}
    provider = MockProvider(
        universe=[{"code": "100010", "name": "근접테스트", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": float(df_a.iloc[-1]["close"]), "chg_pct": 4.2}],
        ohlcv_map={"100010": df_a},
        supply_map={"100010": sup_bad},
        index_df=_make_index_df(dates, weak=False),
    )
    res = run_scan(
        provider=provider,
        storage=StorageRepo(data_dir=tmp_path),
        now_kst=datetime(2026, 11, 27, 15, 35, tzinfo=KST),
        dry_run=True,
        send_notifications=False,
    )
    assert len(res["signals"]) == 0
    assert len(res["near_misses"]) == 1
    assert res["near_misses"][0]["code"] == "100010"

    summary_txt = format_scan_diagnostic_summary(res)
    assert "KOSPI 차트 매수픽 점검" in summary_txt
    assert "근접테스트" in summary_txt


def test_main_scan_latest_trading_day_cli(monkeypatch) -> None:
    import main_scan

    dates = _trading_dates(160, end="2026-11-27")
    df_a = _make_track_a_vcp_df(dates)
    sup_good = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}
    provider = MockProvider(
        universe=[{"code": "100010", "name": "주말점검주", "marcap": 600_000_000_000, "trading_val": float(df_a.iloc[-1]["amount"]), "price": float(df_a.iloc[-1]["close"]), "chg_pct": 4.2}],
        ohlcv_map={"100010": df_a},
        supply_map={"100010": sup_good},
        index_df=_make_index_df(dates, weak=False),
    )
    provider.is_trading_day = lambda _d=None: False
    sent_msgs: list[str] = []

    class _FakeNotifier:
        def __init__(self, dry_run: bool = False):
            self.dry_run = dry_run

        def send_signal_alert(self, _sig) -> bool:
            return True

        def send_text(self, text: str) -> bool:
            sent_msgs.append(text)
            return True

    monkeypatch.setattr(main_scan, "create_provider", lambda provider_name=None, as_of_date=None: provider)
    monkeypatch.setattr(main_scan, "TelegramNotifier", _FakeNotifier)
    rc = main_scan.main(["--dry-run", "--notify-summary", "--latest-trading-day"])
    assert rc == 0
    assert provider.as_of_date == "2026-11-27"
    assert len(sent_msgs) == 1
    assert "주말점검주" in sent_msgs[0]


def test_parallel_scanner_early_safety_filter_and_kis_thread_safety(monkeypatch) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from types import SimpleNamespace

    import providers.kis_provider as kis_mod
    from config.settings import KST
    from engines.scanner import run_scan
    from providers.kis_provider import KISProvider

    kis = KISProvider(as_of_date="2026-11-27")
    kis.app_key = "k"
    kis.app_secret = "s"
    barrier = threading.Barrier(2)

    def _grab_session(_idx: int) -> int:
        barrier.wait(timeout=5.0)
        return id(kis._session)

    with ThreadPoolExecutor(max_workers=2) as pool:
        sess_ids = list(pool.map(_grab_session, range(2)))
    assert len(set(sess_ids)) == 2

    calls = {"n": 0}

    class _FakeHttp:
        def get(self, *_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return SimpleNamespace(status_code=200, json=lambda: {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "rate"})
            return SimpleNamespace(status_code=200, json=lambda: {"rt_cd": "0", "output": {"ok": 1}})

    monkeypatch.setattr(kis_mod.time, "sleep", lambda _s: None)
    kis._ensure_token = lambda force_refresh=False: "dummy"  # type: ignore[method-assign]
    kis._session = _FakeHttp()
    before_iv = kis._min_interval_sec
    res = kis._request_get("/uapi/test", "TESTTR", {})
    assert res["output"]["ok"] == 1
    assert kis._min_interval_sec > before_iv

    dates = _trading_dates(160, end="2026-11-27")
    df_a = _make_track_a_vcp_df(dates)
    close_p = float(df_a.iloc[-1]["close"])
    tval = float(df_a.iloc[-1]["amount"])
    sup_good = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}

    provider = MockProvider(
        universe=[
            {"code": "100001", "name": "통과후보", "marcap": 600_000_000_000, "trading_val": tval, "price": close_p, "chg_pct": 4.2},
            {"code": "100002", "name": "안전수치탈락", "marcap": 600_000_000_000, "trading_val": tval, "price": close_p, "chg_pct": 4.2},
            {"code": "100003", "name": "사전탈락등락", "marcap": 600_000_000_000, "trading_val": tval, "price": close_p, "chg_pct": -2.5},
        ],
        ohlcv_map={"100001": df_a, "100002": df_a},
        supply_map={"100001": sup_good, "100002": sup_good},
        index_df=_make_index_df(dates, weak=False),
    )
    provider.provider_name = "kis"
    rec_lock = threading.Lock()
    safety_calls: list[str] = []
    ohlcv_calls: list[str] = []
    orig_ohlcv = provider.get_daily_ohlcv

    def _safety(code: str) -> dict[str, Any]:
        with rec_lock:
            safety_calls.append(code)
        chg = 0.4 if code == "100002" else 4.2
        return {"is_safe": True, "reject_reason": "", "price": close_p, "chg_pct": chg, "trading_val": tval}

    def _ohlcv(code: str, bars: int = 150, end_date: str | None = None):
        with rec_lock:
            ohlcv_calls.append(code)
        return orig_ohlcv(code, bars=bars, end_date=end_date)

    provider.get_safety_status = _safety  # type: ignore[method-assign]
    provider.get_daily_ohlcv = _ohlcv  # type: ignore[method-assign]

    class _FakeStorage:
        def load_learning_profile(self) -> dict[str, Any]:
            return {"adaptive_params": {}, "promoted_track_d_patterns": []}

        def is_in_cooldown(self, **_kwargs) -> tuple[bool, str]:
            return False, ""

    scan_res = run_scan(
        provider=provider,
        storage=_FakeStorage(),  # type: ignore[arg-type]
        now_kst=datetime(2026, 11, 27, 18, 0, tzinfo=KST),
        dry_run=True,
        send_notifications=False,
        session_override="AFTERMARKET",
    )
    assert scan_res["stats"]["pre_filtered_out"] == 1
    assert scan_res["stats"]["gate_rejected"] == 1
    assert scan_res["stats"]["matched_count"] == 1
    assert scan_res["signals"][0]["code"] == "100001"
    assert "100003" not in safety_calls
    assert "100002" in safety_calls
    assert "100002" not in ohlcv_calls
    assert "100001" in ohlcv_calls


def test_quant_relaxed_thresholds_capture_real_breakouts() -> None:
    from strategies.common_gate import evaluate_common_gate
    from strategies.track_a_vcp import evaluate_track_a
    from strategies.track_b_pullback import evaluate_track_b
    from strategies.track_c_turnaround import evaluate_track_c

    dates = _trading_dates(160, end="2026-11-27")
    idx_df = _make_index_df(dates, weak=False)
    sup_2d = {"frgn_today": 10_000, "orgn_today": -2_000, "frgn_3d": 20_000, "orgn_3d": 0, "total_5d": 25_000, "pos_days_5d": 2, "intraday_pending": False}

    # 1. Track A + Common Gate: +1.3% 돌파, 거래량 138%, 고갈 65%, 피벗 근접 97%, 윗꼬리 0.44
    df_a = _make_track_a_vcp_df(dates)
    for k in range(len(df_a) - 6, len(df_a) - 1):
        df_a.loc[k, "volume"] = 65_000.0
    df_a.loc[len(df_a) - 8, "high"] = round_to_tick(float(df_a.loc[len(df_a) - 8, "high"]) * 1.028)
    prev_c = float(df_a.iloc[-2]["close"])
    o_a = round_to_tick(prev_c * 1.001)
    c_a = round_to_tick(prev_c * 1.013)
    h_a = round_to_tick(c_a + (c_a - o_a) * 0.72)  # 윗꼬리 비율 ~0.42 <= 0.46
    l_a = o_a
    vol_a = 136_000.0
    df_a.loc[df_a.index[-1], ["open", "high", "low", "close", "volume", "amount"]] = [o_a, h_a, l_a, c_a, vol_a, c_a * vol_a]
    df_a_ind = compute_indicators(df_a)
    gate_a = evaluate_common_gate(df_a_ind, idx_df, trading_val=float(df_a_ind.iloc[-1]["amount"]), marcap_krw=5e11)
    assert gate_a["passed"] is True, gate_a["failed_reasons"]
    res_a = evaluate_track_a(df_a_ind, sup_2d)
    assert res_a["passed"] is True, res_a["failed_reasons"]

    # 2. Track B: +8.0% (2.0x) 기준봉, 일시적 중심-2% 흔들기 후 회복, 전일비 122% 재점화
    df_b = _make_track_b_pullback_df(dates)
    anchor_i = len(df_b) - 6
    anchor_prev_c = float(df_b.iloc[anchor_i - 1]["close"])
    b_o = round_to_tick(anchor_prev_c * 1.005)
    b_c = round_to_tick(anchor_prev_c * 1.080)
    b_h = round_to_tick(b_c * 1.006)
    b_l = round_to_tick(b_o * 0.995)
    b_v = 160_000.0
    df_b.loc[anchor_i, ["open", "high", "low", "close", "volume", "amount"]] = [b_o, b_h, b_l, b_c, b_v, b_c * b_v]
    anchor_mid = (b_o + b_c) / 2.0
    for k in range(anchor_i + 1, len(df_b) - 1):
        sub_c = round_to_tick(anchor_mid * (0.982 if k == len(df_b) - 3 else 1.002))
        df_b.loc[k, ["open", "high", "low", "close", "volume", "amount"]] = [
            round_to_tick(sub_c * 1.003),
            round_to_tick(sub_c * 1.006),
            round_to_tick(sub_c * 0.996),
            sub_c,
            78_000.0,
            sub_c * 78_000.0,
        ]
    last_b_prev = float(df_b.iloc[-2]["close"])
    last_b_c = round_to_tick(max(anchor_mid * 1.015, last_b_prev * 1.025))
    df_b.loc[df_b.index[-1], ["open", "high", "low", "close", "volume", "amount"]] = [
        round_to_tick(last_b_prev * 1.002),
        round_to_tick(last_b_c * 1.003),
        round_to_tick(last_b_prev * 0.998),
        last_b_c,
        96_000.0,
        last_b_c * 96_000.0,
    ]
    res_b = evaluate_track_b(compute_indicators(df_b), sup_2d)
    assert res_b["passed"] is True, res_b["failed_reasons"]

    # 3. Track C: 5일 중 2일 수급 + 135% 돌파 거래량
    df_c = _make_track_c_turnaround_df(dates)
    df_c_base_ind = compute_indicators(df_c)
    vma_c = float(df_c_base_ind.iloc[-1]["vol_ma20_prev"])
    df_c.loc[df_c.index[-1], "volume"] = round(vma_c * 1.38)
    df_c.loc[df_c.index[-1], "amount"] = float(df_c.loc[df_c.index[-1], "close"]) * float(df_c.loc[df_c.index[-1], "volume"])
    res_c = evaluate_track_c(compute_indicators(df_c), sup_2d)
    assert res_c["passed"] is True, res_c["failed_reasons"]

    # 4. R:R 기하학: +12.5% 매물대 저항 & -5.0% 손절 -> R:R ~2.50 (2.3+ 통과), -5.6% 손절 -> ~2.23 탈락
    df_rr = _make_track_a_vcp_df(dates)
    entry_rr = float(df_rr.iloc[-1]["close"])
    df_rr.loc[len(df_rr) - 35, "volume"] = 500_000.0
    df_rr.loc[len(df_rr) - 35, "high"] = round_to_tick(entry_rr * 1.125, mode="up")
    df_rr_ind = compute_indicators(df_rr)
    rr_pass = evaluate_rr_geometry(df_rr_ind, entry_rr, entry_rr * 0.95, "TRACK_A_VCP")
    assert rr_pass["passed"] is True, rr_pass["failed_reasons"]
    rr_fail = evaluate_rr_geometry(df_rr_ind, entry_rr, entry_rr * 0.942, "TRACK_A_VCP")
    assert rr_fail["passed"] is False

    # 5. Track A 계단식 상승 돌파(5일 전 저가 ~9.5% 아래) 시 돌파 봉 저가/1*ATR로 타이트 손절 적용 확인
    df_stair = _make_track_a_vcp_df(dates)
    base_st = float(df_stair.loc[len(df_stair) - 6, "close"])
    for step_idx, k_pos in enumerate(range(len(df_stair) - 5, len(df_stair) - 1), start=1):
        sc = round_to_tick(base_st * (1.0 + 0.02 * step_idx))
        so = round_to_tick(sc * 0.996)
        sh = round_to_tick(sc * 1.003)
        sl = round_to_tick(so * 0.997)
        df_stair.loc[k_pos, ["open", "high", "low", "close", "volume", "amount"]] = [so, sh, sl, sc, 35_000.0, sc * 35_000.0]
    st_prev = float(df_stair.iloc[-2]["close"])
    st_o = round_to_tick(st_prev * 1.002)
    st_c = round_to_tick(st_prev * 1.022)
    st_h = round_to_tick(st_c * 1.004)
    st_l = round_to_tick(st_o * 0.997)
    df_stair.loc[df_stair.index[-1], ["open", "high", "low", "close", "volume", "amount"]] = [st_o, st_h, st_l, st_c, 280_000.0, st_c * 280_000.0]
    st_res = evaluate_track_a(compute_indicators(df_stair), sup_2d)
    assert st_res["passed"] is True, st_res["failed_reasons"]
    assert ((st_c - float(st_res["invalidation_raw"])) / st_c * 100.0) <= 6.5


def test_kosdaq_dual_market_universe_index_and_liquidity_cutoffs(tmp_path) -> None:
    from notify.telegram_bot import format_scan_diagnostic_summary
    from providers.kis_provider import KISProvider
    from providers.naver_fallback import fetch_naver_index_ohlcv, fetch_naver_market_universe

    class _FakeResp:
        def __init__(self, status_code: int, payload: Any = None, content: bytes = b""):
            self.status_code = status_code
            self._payload = payload
            self.content = content

        def json(self) -> Any:
            return self._payload

    class _FakeSess:
        def get(self, url: str, headers=None, timeout: float = 10.0):
            if "marketValue/KOSDAQ" in url:
                return _FakeResp(
                    200,
                    {
                        "stocks": [
                            {
                                "itemCode": "247540",
                                "stockName": "에코프로비엠",
                                "stockEndType": "stock",
                                "closePriceRaw": "150000",
                                "fluctuationsRatio": "4.2",
                                "accumulatedTradingVolumeRaw": "300000",
                                "accumulatedTradingValueRaw": "45000000000",
                                "marketValueRaw": "14000000000000",
                            },
                            {
                                "itemCode": "999990",
                                "stockName": "소형코스닥",
                                "stockEndType": "stock",
                                "closePriceRaw": "5000",
                                "fluctuationsRatio": "5.0",
                                "accumulatedTradingVolumeRaw": "400000",
                                "accumulatedTradingValueRaw": "2000000000",
                                "marketValueRaw": "70000000000",  # 700억 < KOSDAQ 하한 1,000억 제외
                            },
                        ]
                    },
                )
            if "sise.nhn?symbol=KOSDAQ" in url:
                xml = """<?xml version="1.0" encoding="EUC-KR"?>
                <protocol><chartdata symbol="KOSDAQ">
                  <item data="20261126|690.0|695.0|685.0|690.0|100000" />
                  <item data="20261127|692.0|710.0|690.0|708.99|120000" />
                </chartdata></protocol>"""
                return _FakeResp(200, content=xml.encode("euc-kr"))
            return _FakeResp(404, {})

    kq_uni = fetch_naver_market_universe(market="KOSDAQ", session=_FakeSess())
    assert len(kq_uni) == 1
    assert kq_uni[0]["code"] == "247540"
    assert kq_uni[0]["market"] == "KOSDAQ"

    kq_idx = fetch_naver_index_ohlcv(bars=5, market="KOSDAQ", session=_FakeSess())
    assert len(kq_idx) == 2
    assert float(kq_idx.iloc[-1]["close"]) == 708.99

    # KISProvider.get_index_ohlcv: KOSPI('0001') vs KOSDAQ('1001') 분기 확인
    seen_iscd: list[str] = []
    kis = KISProvider()
    kis.app_key = "k"
    kis.app_secret = "s"

    def _fake_get(path: str, tr_id: str, params: dict[str, Any]) -> dict[str, Any]:
        iscd = str(params.get("FID_INPUT_ISCD"))
        end_ymd = str(params.get("FID_INPUT_DATE_2") or "99999999")
        seen_iscd.append(iscd)
        rows = [
            {"stck_bsop_date": "20261127", "bstp_nmix_prpr": "708.99" if iscd == "1001" else "4915.00"},
            {"stck_bsop_date": "20261126", "bstp_nmix_prpr": "690.00" if iscd == "1001" else "4711.00"},
        ]
        filtered = [r for r in rows if r["stck_bsop_date"] <= end_ymd]
        return {"rt_cd": "0", "output2": filtered}

    kis._request_get = _fake_get  # type: ignore[method-assign]
    assert float(kis.get_index_ohlcv(bars=5, market="KOSDAQ").iloc[-1]["close"]) == 708.99
    assert float(kis.get_index_ohlcv(bars=5, market="KOSPI").iloc[-1]["close"]) == 4915.0
    assert seen_iscd[0] == "1001" and "0001" in seen_iscd and set(seen_iscd) == {"1001", "0001"}

    # run_scan에서 KOSDAQ 종목은 거래대금 30억 기준(20억 탈락) 및 KOSDAQ 지수 약세 별도 반영 검증
    dates = _trading_dates(160, end="2026-11-27")
    df_pass = _make_track_a_vcp_df(dates)
    df_low_tv = _make_track_a_vcp_df(dates)
    df_low_tv.loc[df_low_tv.index[-1], "amount"] = 2_000_000_000.0  # 20억 (KOSDAQ 30억 미달)
    sup_good = {"frgn_today": 10_000, "orgn_today": 15_000, "frgn_3d": 40_000, "orgn_3d": 30_000, "total_5d": 90_000, "pos_days_5d": 4, "intraday_pending": False}
    p_dual = MockProvider(
        universe=[
            {"code": "247540", "name": "코스닥주도주", "market": "KOSDAQ", "marcap": 2_000_000_000_000, "trading_val": float(df_pass.iloc[-1]["amount"]), "price": float(df_pass.iloc[-1]["close"]), "chg_pct": 4.2},
            {"code": "247550", "name": "코스닥유동성미달", "market": "KOSDAQ", "marcap": 200_000_000_000, "trading_val": 2_000_000_000.0, "price": float(df_low_tv.iloc[-1]["close"]), "chg_pct": 4.2},
        ],
        ohlcv_map={"247540": df_pass, "247550": df_low_tv},
        supply_map={"247540": sup_good, "247550": sup_good},
        index_df=_make_index_df(dates, weak=False),       # KOSPI 정상
        kosdaq_index_df=_make_index_df(dates, weak=True), # KOSDAQ 약세
    )
    res = run_scan(
        provider=p_dual,
        storage=StorageRepo(data_dir=tmp_path),
        now_kst=datetime(2026, 11, 27, 15, 35, tzinfo=KST),
        dry_run=True,
        send_notifications=False,
    )
    assert res["stats"]["pre_filtered_out"] == 1
    assert res["stats"]["matched_count"] == 1
    sig = res["signals"][0]
    assert sig["market"] == "KOSDAQ"
    assert sig["snapshot"]["market_regime"] == "WEAK"  # KOSDAQ 지수 약세 반영 확인
    summary_txt = format_scan_diagnostic_summary(res)
    assert "KOSPI·KOSDAQ 차트 매수픽 점검" in summary_txt
    assert "[KOSDAQ]" in summary_txt
