from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional fallback in stripped envs
    def load_dotenv(*_args, **_kwargs):
        return False


KST = timezone(timedelta(hours=9))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "storage" / "data"
SIGNALS_LOG_FILE = DATA_DIR / "signals_log.json"
TOP20_HISTORY_FILE = DATA_DIR / "top20_history.json"
PATTERN_QUEUE_FILE = DATA_DIR / "pattern_queue.json"
LEARNING_PROFILE_FILE = DATA_DIR / "learning_profile.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_dynamic_min_trading_val_krw(now_kst: datetime | None = None) -> int:
    """장중 시간대별 탄력 최소 거래대금 하한(원) 반환.

    - 프리장(08:00~09:00): 직전 정규장 마감 유동성 기준 30억 원
    - 09:00 ~ 10:30: 10억 원
    - 10:30 ~ 13:00: 20억 원
    - 13:00 이후 (정규장 후반·애프터장·분석 포함): 30억 원
    """
    if now_kst is None:
        return 3_000_000_000
    t = now_kst.timetz().replace(tzinfo=None)
    if dt_time(9, 0) <= t < dt_time(10, 30):
        return 1_000_000_000
    if dt_time(10, 30) <= t < dt_time(13, 0):
        return 2_000_000_000
    return 3_000_000_000


def classify_market_session(now_kst: datetime | None = None) -> str:
    """KST 시각 기준으로 PREMARKET / REGULAR / AFTERMARKET 장 세션을 판별한다."""
    if now_kst is None:
        return "REGULAR"
    t = now_kst.timetz().replace(tzinfo=None)
    if dt_time(8, 0) <= t < dt_time(9, 0):
        return "PREMARKET"
    if dt_time(9, 0) <= t <= dt_time(15, 30):
        return "REGULAR"
    return "AFTERMARKET"


def _default_data_provider() -> str:
    explicit = os.getenv("DATA_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("KIS_APP_KEY", "").strip() and os.getenv("KIS_APP_SECRET", "").strip():
        return "kis"
    if os.getenv("KB_APP_KEY", "").strip() and os.getenv("KB_APP_SECRET", "").strip():
        return "kb"
    return "pykrx"


@dataclass(frozen=True)
class Settings:
    """코스피 1:3 스윙 스캐너 전역 설정."""

    # 프로바이더 및 인증 (DATA_PROVIDER 미지정 시 KIS 키 존재하면 자동 'kis')
    data_provider: str = field(default_factory=_default_data_provider)
    kis_app_key: str = field(default_factory=lambda: os.getenv("KIS_APP_KEY", "").strip())
    kis_app_secret: str = field(default_factory=lambda: os.getenv("KIS_APP_SECRET", "").strip())
    kis_api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "KIS_API_BASE_URL", "https://openapi.koreainvestment.com:9443"
        ).strip()
    )
    kis_is_paper: bool = field(default_factory=lambda: _env_bool("KIS_IS_PAPER", False))

    kb_app_key: str = field(default_factory=lambda: os.getenv("KB_APP_KEY", "").strip())
    kb_app_secret: str = field(default_factory=lambda: os.getenv("KB_APP_SECRET", "").strip())
    kb_api_base_url: str = field(default_factory=lambda: os.getenv("KB_API_BASE_URL", "").strip())

    # 주식 전용 텔레그램 봇
    telegram_token: str = field(
        default_factory=lambda: os.getenv("KOSPI_TELEGRAM_TOKEN", "").strip()
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.getenv("KOSPI_TELEGRAM_CHAT_ID", "").strip()
    )
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip().upper())

    # 유니버스 필터
    min_marcap_krw: int = 50_000_000_000       # 500억 원
    max_marcap_krw: int = 5_000_000_000_000    # 5조 원
    top20_min_trading_val_krw: int = 3_000_000_000  # 15:50 분석 최소 거래대금 30억 원
    excluded_name_keywords: tuple[str, ...] = (
        "스팩", "SPAC", "리츠", "REIT", "ETF", "ETN", "인버스", "레버리지",
        "우B", "우C", "우(", "우선주", "선물", "채권", "합성", "TR",
    )

    # 공통 관문 임계값
    min_chg_pct: float = 1.5
    max_chg_pct: float = 10.0
    max_upper_wick_ratio: float = 0.40
    max_disparity_20_pct: float = 118.0
    weak_regime_index_drop_pct: float = -1.5
    index_bars: int = 100
    ohlcv_bars: int = 150
    supply_days: int = 5

    # Track A (VCP 타이트 돌파)
    track_a_vcp_ratio_max: float = 0.70
    track_a_dry_vol_ratio_max: float = 0.60
    track_a_breakout_vol_ratio_min: float = 1.80
    track_a_pivot_proximity_min: float = 0.985
    track_a_atr_mult: float = 1.5
    track_a_max_risk_pct: float = 5.5

    # Track B (기준봉 눌림목 재점화)
    track_b_anchor_min_chg_pct: float = 9.5
    track_b_anchor_min_vol_ratio: float = 2.50
    track_b_lookback_min_days: int = 4
    track_b_lookback_max_days: int = 15
    track_b_adj_max_vol_vs_anchor: float = 0.45
    track_b_reignite_min_vol_vs_prev: float = 1.50
    track_b_max_risk_pct: float = 5.0

    # Track C (바닥권 수급 매집 전환)
    track_c_max_ratio_to_120h: float = 0.85
    track_c_box_20d_range_max: float = 0.25
    track_c_supply_pos_days_min: int = 3
    track_c_breakout_vol_ratio_min: float = 1.80
    track_c_max_risk_pct: float = 6.5

    # 1:3 R:R 기하학 기준
    min_clear_upside_pct: float = 10.0
    min_projected_rr: float = 3.0
    blue_sky_upside_pct: float = 30.0
    heavy_vol_resistance_mult: float = 2.0
    resistance_lookback_days: int = 120

    # 쿨다운 · 오답노트 · 분석 · 연속 스캔 스케줄 (08:00~21:00)
    cooldown_trading_days: int = 5
    error_tracking_max_days: int = 10
    top20_lookback_trading_days: int = 7
    scan_schedule_kst: tuple[str, ...] = ("08:00-12:25", "12:25-16:50", "16:50-21:00")
    loop_interval_sec: int = 240
    premarket_max_gap_over_ceiling_pct: float = 1.5
    analyze_time_kst: str = "15:50"
    weekly_report_time_kst: str = "금 16:30"


settings = Settings()
