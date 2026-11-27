from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import (
    DATA_DIR,
    KST,
    LEARNING_PROFILE_FILE,
    PATTERN_QUEUE_FILE,
    SIGNALS_LOG_FILE,
    TOP20_HISTORY_FILE,
)


def _default_signals_doc() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": "", "signals": []}


def _default_top20_doc() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": "", "history": []}


def _default_pattern_doc() -> dict[str, Any]:
    return {"schema_version": 1, "updated_at": "", "patterns": []}


def _default_learning_profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": "",
        "learning_started_date": datetime.now(tz=KST).strftime("%Y-%m-%d"),
        "learning_days": 0,
        "sample_counts": {
            "live_total": 0,
            "live_completed": 0,
            "live_winners": 0,
            "bootstrap_completed": 0,
            "effective_completed": 0,
            "effective_winners": 0,
        },
        "adaptive_params": {
            "max_upper_wick_ratio": 0.40,
            "max_disparity_20_pct": 118.0,
            "min_projected_rr": 3.0,
            "weak_regime_min_rr": 3.5,
            "weak_regime_risk_mult": 1.0,
            "track_risk_mult": {
                "TRACK_A_VCP": 1.0,
                "TRACK_B_PULLBACK": 1.0,
                "TRACK_C_TURNAROUND": 1.0,
                "TRACK_D": 1.0,
            },
            "track_a_dry_vol_ratio_max": 0.60,
            "track_weights": {
                "TRACK_A_VCP": 1.0,
                "TRACK_B_PULLBACK": 1.0,
                "TRACK_C_TURNAROUND": 1.0,
                "TRACK_D": 0.9,
            },
            "disabled_tracks": [],
        },
        "promoted_track_d_patterns": [],
        "demoted_track_d_until": {},
        "phase4_policy": {
            "phase4_active": False,
            "activated_date": None,
            "sl_pct_by_track": {},
            "default_sl_pct": None,
            "tp_levels_pct": [10.0, 20.0, 30.0],
            "breakeven_trigger_pct": 10.0,
        },
        "daily_history": [],
        "last_adjustments": [],
    }


def _deep_merge_defaults(default_obj: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    merged = dict(default_obj)
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge_defaults(merged[k], v)
        else:
            merged[k] = v
    return merged


class StorageRepo:
    """스레드 안전한 원자적 JSON 저장소 및 오답노트/쿨다운 조회기."""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self.signals_path = (
            self.data_dir / SIGNALS_LOG_FILE.name if data_dir is not None else SIGNALS_LOG_FILE
        )
        self.top20_path = (
            self.data_dir / TOP20_HISTORY_FILE.name if data_dir is not None else TOP20_HISTORY_FILE
        )
        self.pattern_path = (
            self.data_dir / PATTERN_QUEUE_FILE.name if data_dir is not None else PATTERN_QUEUE_FILE
        )
        self.learning_path = (
            self.data_dir / LEARNING_PROFILE_FILE.name
            if data_dir is not None
            else LEARNING_PROFILE_FILE
        )
        self._lock = threading.RLock()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_kst_str() -> str:
        return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S KST")

    def _read_json(self, path: Path, default_factory) -> dict[str, Any]:
        with self._lock:
            if not path.exists():
                return default_factory()
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                bak_path = path.with_suffix(path.suffix + ".bak")
                if bak_path.exists():
                    try:
                        with bak_path.open("r", encoding="utf-8") as fb:
                            data = json.load(fb)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        pass
            return default_factory()

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(payload)
            payload["updated_at"] = self._now_kst_str()

            if path.exists():
                bak_path = path.with_suffix(path.suffix + ".bak")
                try:
                    shutil.copy2(path, bak_path)
                except Exception:
                    pass

            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                raise

    # ── signals_log.json ─────────────────────────────────────────────────
    def load_signals_doc(self) -> dict[str, Any]:
        doc = self._read_json(self.signals_path, _default_signals_doc)
        if not isinstance(doc.get("signals"), list):
            doc["signals"] = []
        return doc

    def save_signals_doc(self, doc: dict[str, Any]) -> None:
        self._write_json_atomic(self.signals_path, doc)

    def get_signals(self) -> list[dict[str, Any]]:
        return list(self.load_signals_doc().get("signals", []))

    def get_active_signals(self) -> list[dict[str, Any]]:
        return [
            s
            for s in self.get_signals()
            if str((s.get("tracking") or {}).get("status", "ACTIVE")).upper() == "ACTIVE"
        ]

    def upsert_signal(self, signal_record: dict[str, Any]) -> bool:
        """동일 signal_id 존재 시 갱신, 신규면 추가. 신규 추가 시 True 반환."""
        sig_id = str(signal_record.get("signal_id") or "").strip()
        if not sig_id:
            raise ValueError("signal_record requires non-empty signal_id")

        with self._lock:
            doc = self.load_signals_doc()
            signals = doc["signals"]
            for idx, item in enumerate(signals):
                if item.get("signal_id") == sig_id:
                    signals[idx] = signal_record
                    self.save_signals_doc(doc)
                    return False
            signals.append(signal_record)
            self.save_signals_doc(doc)
            return True

    def is_in_cooldown(
        self,
        code: str,
        target_date: str,
        trading_dates: list[str] | None = None,
        cooldown_days: int = 5,
    ) -> tuple[bool, str]:
        """동일 종목이 최근 cooldown_days 거래일 이내 이미 포착되었는지 확인."""
        norm_code = str(code).zfill(6)
        signals = self.get_signals()
        same_code = [s for s in signals if str(s.get("code", "")).zfill(6) == norm_code]
        if not same_code:
            return False, ""

        latest = max(same_code, key=lambda s: str(s.get("signal_date", "")))
        prev_date = str(latest.get("signal_date", ""))
        if not prev_date:
            return False, ""

        if prev_date == target_date:
            return True, f"당일 이미 포착({latest.get('signal_id')})"

        if trading_dates and prev_date in trading_dates and target_date in trading_dates:
            diff_bars = trading_dates.index(target_date) - trading_dates.index(prev_date)
            if 0 <= diff_bars < cooldown_days:
                return (
                    True,
                    f"{cooldown_days}거래일 쿨다운 ({prev_date} 포착, {diff_bars}거래일 경과)",
                )
            return False, ""

        # 거래일 캘린더에 이전 날짜가 없거나 5거래일 이상 이전인 경우 달력 기준 보수 확인
        try:
            d_prev = datetime.strptime(prev_date, "%Y-%m-%d").date()
            d_curr = datetime.strptime(target_date, "%Y-%m-%d").date()
            if d_curr >= d_prev and trading_dates:
                between = [d for d in trading_dates if prev_date < d <= target_date]
                if len(between) < cooldown_days:
                    return (
                        True,
                        f"{cooldown_days}거래일 쿨다운 ({prev_date} 포착, {len(between)}거래일 경과)",
                    )
                return False, ""
            cal_diff = (d_curr - d_prev).days
            if 0 <= cal_diff < cooldown_days:
                return True, f"쿨다운 ({prev_date} 포착, {cal_diff}일 경과)"
        except ValueError:
            pass

        return False, ""

    def get_recent_signals_by_trading_days(
        self,
        target_date: str,
        trading_dates: list[str],
        lookback_days: int = 7,
    ) -> list[dict[str, Any]]:
        """target_date 포함 직전 lookback_days 거래일 내 포착된 시그널 목록 반환."""
        valid_dates = [d for d in trading_dates if d <= target_date]
        window = set(valid_dates[-lookback_days:]) if valid_dates else {target_date}
        return [s for s in self.get_signals() if str(s.get("signal_date", "")) in window]

    # ── top20_history.json ───────────────────────────────────────────────
    def load_top20_doc(self) -> dict[str, Any]:
        doc = self._read_json(self.top20_path, _default_top20_doc)
        if not isinstance(doc.get("history"), list):
            doc["history"] = []
        return doc

    def upsert_top20_day(self, day_record: dict[str, Any]) -> None:
        target_date = str(day_record.get("date") or "").strip()
        if not target_date:
            raise ValueError("top20 day_record requires non-empty date")
        with self._lock:
            doc = self.load_top20_doc()
            hist = doc["history"]
            for idx, item in enumerate(hist):
                if item.get("date") == target_date:
                    hist[idx] = day_record
                    self._write_json_atomic(self.top20_path, doc)
                    return
            hist.append(day_record)
            hist.sort(key=lambda x: str(x.get("date", "")))
            # 최근 180거래일 보관
            if len(hist) > 180:
                doc["history"] = hist[-180:]
            self._write_json_atomic(self.top20_path, doc)

    # ── pattern_queue.json ───────────────────────────────────────────────
    def load_pattern_doc(self) -> dict[str, Any]:
        doc = self._read_json(self.pattern_path, _default_pattern_doc)
        if not isinstance(doc.get("patterns"), list):
            doc["patterns"] = []
        return doc

    def append_pattern_entries(self, entries: list[dict[str, Any]]) -> int:
        """(date, code) 중복 없이 신규 패턴 후보 누적. 추가 건수 반환."""
        if not entries:
            return 0
        added = 0
        with self._lock:
            doc = self.load_pattern_doc()
            patterns = doc["patterns"]
            existing = {(p.get("date"), p.get("code")): idx for idx, p in enumerate(patterns)}
            for entry in entries:
                key = (entry.get("date"), entry.get("code"))
                if key in existing:
                    patterns[existing[key]] = entry
                else:
                    patterns.append(entry)
                    existing[key] = len(patterns) - 1
                    added += 1
            patterns.sort(key=lambda x: (str(x.get("date", "")), str(x.get("code", ""))))
            if len(patterns) > 1000:
                doc["patterns"] = patterns[-1000:]
            self._write_json_atomic(self.pattern_path, doc)
        return added

    # ── learning_profile.json ────────────────────────────────────────────
    def load_learning_profile(self) -> dict[str, Any]:
        loaded = self._read_json(self.learning_path, _default_learning_profile)
        return _deep_merge_defaults(_default_learning_profile(), loaded)

    def save_learning_profile(self, profile: dict[str, Any]) -> None:
        self._write_json_atomic(self.learning_path, profile)


repo = StorageRepo()
