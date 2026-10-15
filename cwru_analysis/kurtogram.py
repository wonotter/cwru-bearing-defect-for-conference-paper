"""
Fast Kurtogram — 충격 성분이 가장 강한 주파수 대역을 자동으로 찾는다.

Antoni(2007)의 복소 FIR 필터 뱅크로 이진·삼진 분할과 다운샘플링을 수행한다.
각 대역의 스펙트럴 커토시스를 구하고, 최소 폭을 만족하는 최적 대역을 선택한다.

원본: J. Antoni, Fast_Kurtogram.m (12-2014, File Exchange 48912).
저작권 및 사용 조건: THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

from cwru_analysis.config import (
    KURTOGRAM_FILTER_CUTOFF,
    KURTOGRAM_FILTER_ORDER,
    KURTOGRAM_MIN_BANDWIDTH,
    KURTOGRAM_NLEVEL,
)


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
    Antoni의 이진·삼진 필터 뱅크로 Fast Kurtogram을 계산한다.

    Parameters
    ----------
    signal : np.ndarray
        시간 영역 신호 (1D)
    fs : float
        샘플링 레이트 (Hz)
    nlevel : int
        이진 최대 레벨. 5이면 0, 1, 1.6, ..., 4.6, 5의 10개 행.
    min_bandwidth : float
        최적 대역 선택 시 이보다 좁은 칸은 후보에서 제외 (Hz).

    Returns
    -------
    dict
        {
            "fs": float,
            "rows": list[dict],   -- 단계별 칸 정보 (level은 실수)
            "best": dict,         -- 선택된 최적 칸
        }
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("signal은 비어 있지 않은 1차원 배열이어야 합니다.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal에 NaN 또는 무한대가 포함되어 있습니다.")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs는 양의 유한한 값이어야 합니다.")
    if not isinstance(nlevel, (int, np.integer)) or nlevel < 0:
        raise ValueError("nlevel은 0 이상의 정수여야 합니다.")
    if nlevel > np.log2(len(signal)) - 7:
        raise ValueError(
            "Antoni 원본의 길이 조건을 만족하지 않습니다. "
            "샘플 수를 늘리거나 nlevel을 낮추세요."
        )
    if not np.isfinite(min_bandwidth) or min_bandwidth < 0.0:
        raise ValueError("min_bandwidth는 0 이상의 유한한 값이어야 합니다.")

    signal = signal - np.mean(signal)
    h, g, h1, h2, h3 = _design_filter_bank()
    rows = [_make_kurtogram_row(0.0, [_spectral_kurtosis(signal)], fs)]
    nodes = [signal]

    for level in range(1, nlevel + 1):
        next_nodes: list[np.ndarray] = []
        scores: list[float] = []
        for node in nodes:
            a, d = _binary_filter_bank(node, h, g)
            # MATLAB의 a(length(h):end), d(length(g):end)와 같은 범위.
            scores.append(_spectral_kurtosis(a[len(h) - 1:]))
            scores.append(_spectral_kurtosis(d[len(g) - 1:]))
            next_nodes.extend((a, d))
        rows.append(_make_kurtogram_row(float(level), scores, fs))

        # 원본은 마지막 이진 행 직전까지 삼진 행을 끼워 넣는다.
        if level < nlevel:
            scores = []
            for node in nodes:
                bands = _ternary_filter_bank(node, h1, h2, h3)
                for band in bands:
                    # K_wpQ_local은 삼진 결과에도 length(h)를 적용한다.
                    scores.append(_spectral_kurtosis(band[len(h) - 1:]))
            ternary_level = level + np.log2(3.0) - 1.0
            rows.append(_make_kurtogram_row(ternary_level, scores, fs))

        nodes = next_nodes

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
# 해석 FIR 필터 뱅크
# ─────────────────────────────────────────────────────────────────
def _design_filter_bank() -> tuple[np.ndarray, ...]:
    """Antoni 원본의 h, g, h1, h2, h3 필터 계수를 생성한다."""
    order = KURTOGRAM_FILTER_ORDER
    cutoff = KURTOGRAM_FILTER_CUTOFF
    n = np.arange(order + 1)
    h = firwin(order + 1, cutoff, window="hamming")
    h = h * np.exp(2j * np.pi * n * 0.125)

    # 원본의 mod(..., order)를 보존한다. g는 h보다 한 탭 짧다.
    n = np.arange(1, order + 1)
    g = h[np.mod(-n, order)] * (-1.0) ** n

    ternary_order = 3 * order // 2
    n = np.arange(ternary_order + 1)
    h1 = firwin(ternary_order + 1, 2.0 * cutoff / 3.0, window="hamming")
    h1 = h1 * np.exp(2j * np.pi * n / 12.0)
    h2 = h1 * np.exp(2j * np.pi * n / 6.0)
    h3 = h1 * np.exp(2j * np.pi * n / 3.0)
    return h, g, h1, h2, h3


def _binary_filter_bank(
    signal: np.ndarray,
    h: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """이진 필터링 후 2:2:N 샘플을 남기고 고역 가지의 주파수를 정렬한다."""
    a = lfilter(h, 1.0, signal)[1::2]
    d = lfilter(g, 1.0, signal)[1::2]
    d[::2] *= -1.0  # 원본의 d .* (-1).^(1:N)
    return a, d


def _ternary_filter_bank(
    signal: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """삼진 필터링 후 원본의 3:3:N 샘플만 남긴다."""
    a1 = lfilter(h1, 1.0, signal)[2::3]
    a2 = lfilter(h2, 1.0, signal)[2::3]
    a3 = lfilter(h3, 1.0, signal)[2::3]
    return a1, a2, a3


# ─────────────────────────────────────────────────────────────────
# SK 계산 및 행 구성
# ─────────────────────────────────────────────────────────────────
def _spectral_kurtosis(signal: np.ndarray) -> float:
    """Antoni의 kurt2: 실수 신호는 3, 복소 신호는 2를 차감한다."""
    signal = signal - np.mean(signal)
    power = np.abs(signal) ** 2
    mean_power = float(np.mean(power))
    if mean_power < np.finfo(np.float64).eps:
        return 0.0
    correction = 3.0 if np.isrealobj(signal) else 2.0
    return float(np.mean(power ** 2) / (mean_power ** 2) - correction)


def _make_kurtogram_row(level: float, scores: list[float], fs: float) -> dict:
    """행의 실제 칸 폭·중심 주파수를 구하고 원본처럼 음의 SK를 0으로 만든다."""
    bandwidth = fs / (2.0 * len(scores))
    centers = (np.arange(len(scores)) + 0.5) * bandwidth
    return {
        "level": float(level),
        "bandwidth": float(bandwidth),
        "centers": centers,
        "kurtosis": np.maximum(np.asarray(scores, dtype=np.float64), 0.0),
    }


# ─────────────────────────────────────────────────────────────────
# 최적 대역 선택
# ─────────────────────────────────────────────────────────────────
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
            "level": float(row["level"]),
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
    # MATLAB max_IJ와 같이 동률이면 낮은 주파수, 그다음 위쪽 행을 택한다.
    return max(candidates, key=lambda c: (c["kurtosis"], -c["low"]))
