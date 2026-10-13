"""
Fast Kurtogram — 충격 성분이 가장 강한 주파수 대역을 자동으로 찾는다.

0 Hz ~ 나이퀴스트를 단계적으로 분할하고 각 구간의 스펙트럴 커토시스를
구해서, 충격 점수가 가장 높은 구간을 밴드패스 대역으로 선택한다.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from cwru_analysis.config import KURTOGRAM_MIN_BANDWIDTH, KURTOGRAM_NLEVEL


# ─────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────
def compute_kurtogram(
    signal: np.ndarray,
    fs: float,
    nlevel: int = KURTOGRAM_NLEVEL,
    min_bandwidth: float = KURTOGRAM_MIN_BANDWIDTH,
) -> dict:
    """
    주파수 구간별 충격 세기(스펙트럴 커토시스)를 계산한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    nlevel : int
        최대 분할 단계. 5이면 최대 32칸.
    min_bandwidth : float
        최적 대역 선택 시 이보다 좁은 칸은 후보에서 제외 (Hz).

    Returns
    -------
    dict
        {
            "fs": float,
            "rows": list[dict],   -- 단계별 칸 정보
            "best": dict,         -- 선택된 최적 칸
        }
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    nyquist = fs / 2.0

    rows: list[dict] = []
    for level in range(nlevel + 1):
        n_bands = 2 ** level
        bandwidth = nyquist / n_bands
        centers = np.empty(n_bands)
        scores = np.empty(n_bands)

        for idx in range(n_bands):
            low = idx * bandwidth
            high = (idx + 1) * bandwidth
            centers[idx] = 0.5 * (low + high)
            kept = _keep_frequency_band(signal, fs, low, high)
            scores[idx] = _spectral_kurtosis(kept)

        rows.append({
            "level": level,
            "bandwidth": float(bandwidth),
            "centers": centers,
            "kurtosis": scores,
        })

    best = _select_best_band(rows, min_bandwidth)
    return {"fs": float(fs), "rows": rows, "best": best}


def select_best_band(
    signal: np.ndarray,
    fs: float,
    nlevel: int = KURTOGRAM_NLEVEL,
    min_bandwidth: float = KURTOGRAM_MIN_BANDWIDTH,
) -> tuple[float, float]:
    """
    Kurtogram에서 최적 밴드패스 대역(low, high)을 반환한다.

    ``compute_kurtogram``의 간편 래퍼.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    nlevel : int
        최대 분할 단계
    min_bandwidth : float
        최소 대역 폭 (Hz)

    Returns
    -------
    tuple[float, float]
        (low_hz, high_hz)
    """
    best = compute_kurtogram(signal, fs, nlevel, min_bandwidth)["best"]
    return float(best["low"]), float(best["high"])


# ─────────────────────────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────────────────────────
def _spectral_kurtosis(signal: np.ndarray) -> float:
    """힐베르트 진폭의 커토시스로 충격 집중도를 측정한다."""
    amplitude = np.abs(hilbert(signal))
    power = amplitude ** 2
    mean_power = float(np.mean(power))
    if mean_power <= 0.0:
        return 0.0
    return float(np.mean(power ** 2) / (mean_power ** 2) - 2.0)


def _keep_frequency_band(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> np.ndarray:
    """low~high(Hz) 대역만 남긴다."""
    nyquist = fs / 2.0
    if low <= 0.0 and high >= nyquist * 0.999:
        return signal

    low_ratio = max(low / nyquist, 1e-4)
    high_ratio = min(high / nyquist, 0.999)
    if high_ratio - low_ratio < 0.01:
        return np.zeros_like(signal)

    if low <= 0.0:
        sos = butter(3, high_ratio, btype="low", output="sos")
    else:
        sos = butter(3, [low_ratio, high_ratio], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def _select_best_band(rows: list[dict], min_bandwidth: float) -> dict:
    """폭이 충분한 칸 중 점수가 가장 높은 칸을 선택한다."""
    candidates: list[dict] = []
    for row in rows:
        if row["bandwidth"] < min_bandwidth:
            continue
        idx = int(np.argmax(row["kurtosis"]))
        center = float(row["centers"][idx])
        bw = float(row["bandwidth"])
        candidates.append({
            "level": int(row["level"]),
            "center": center,
            "bandwidth": bw,
            "low": center - bw / 2.0,
            "high": center + bw / 2.0,
            "kurtosis": float(row["kurtosis"][idx]),
        })

    if not candidates:
        raise ValueError(
            "Kurtogram에서 선택할 수 있는 구간이 없습니다. "
            "min_bandwidth를 낮추거나 nlevel을 확인하세요."
        )
    return max(candidates, key=lambda c: c["kurtosis"])
