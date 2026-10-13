"""
논문용 그래프 생성 모듈 (Phase 3).

포락선 스펙트럼, Kurtogram 맵, 결함 유형별·크기별·HP별 비교 그래프를
논문 삽입 품질로 생성한다.

생성 대상
──────────
  1. 40개 파일 각각의 포락선 스펙트럼 (결함 주파수 마커 포함)
  2. 40개 파일 각각의 Kurtogram 히트맵
  3. 조건별 결함 유형 비교   (12장: 3 sizes × 4 HPs)
  4. 결함 크기별 비교        (12장: 3 types × 4 HPs)
  5. HP(RPM)별 비교          (10장: 3 types × 3 sizes + 1 Normal)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwru_analysis.config import (
    DEFECT_FREQ_COLORS,
    DEFECT_FREQ_MULTIPLIERS,
    FAULT_BALL,
    FAULT_INNER,
    FAULT_NORMAL,
    FAULT_OUTER,
    FAULT_TYPE_SHORT,
    FAULT_TYPE_TO_FREQ,
    FIGURE_DPI,
    N_HARMONICS_PLOT,
    PLOT_FREQ_MAX,
    RESULTS_DIR,
)
from cwru_analysis.envelope import compute_envelope_spectrum
from cwru_analysis.frequencies import calculate_defect_frequencies


# ─────────────────────────────────────────────────────────────────
# 결과 디렉토리
# ─────────────────────────────────────────────────────────────────
ENVELOPE_DIR = RESULTS_DIR / "envelope_spectra"
KURTOGRAM_DIR = RESULTS_DIR / "kurtograms"
COMPARISON_DIR = RESULTS_DIR / "comparisons"


# 비교 그래프에서 서브플롯 정렬 순서
_TYPE_ORDER = {FAULT_OUTER: 0, FAULT_INNER: 1, FAULT_BALL: 2, FAULT_NORMAL: 3}


def _ensure_dir(path: Path) -> None:
    """디렉토리가 없으면 생성한다."""
    path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 결함 주파수 마커 (공통 헬퍼)
# ─────────────────────────────────────────────────────────────────
def _extended_harmonics(defect_freqs: dict, fname: str,
                        n: int, freq_max: float) -> list[float]:
    """defect_freqs의 기본 주파수에서 n차까지 고조파를 생성한다."""
    if fname not in defect_freqs or not defect_freqs[fname]:
        return []
    base = defect_freqs[fname][0]
    return [base * (i + 1) for i in range(n) if base * (i + 1) <= freq_max]


def _add_defect_markers(
    ax: plt.Axes,
    defect_freqs: dict,
    fault_type: str,
    spectrum_freqs: np.ndarray,
    spectrum_magnitude: np.ndarray,
    freq_max: float = PLOT_FREQ_MAX,
    n_harmonics: int = N_HARMONICS_PLOT,
) -> None:
    """
    포락선 스펙트럼의 피크 위에서 시작하는 결함 주파수 점선을 추가한다.

    - 결함 데이터: 해당 결함 유형의 고조파를 빨간/파란/초록 점선으로 강조
    - 정상 데이터: BPFO/BPFI/BSF 고조파를 진한 점선으로 표시
    - 레전드에 이론 기본 주파수와 1차 점선의 관측 피크 주파수(Hz)를 함께 표기
    - 각 고조파 주변(±2%, 최소 ±2 FFT 빈)의 최대 진폭 FFT 빈에 점선을 정렬
    - 점선은 해당 피크 위부터 표시하며 스펙트럼 데이터는 그대로 유지
    """
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    freq_resolution = (
        float(spectrum_freqs[1] - spectrum_freqs[0])
        if len(spectrum_freqs) > 1 else 0.0
    )

    def marker_position(freq: float) -> tuple[float, float]:
        half_width = max(freq * 0.02, 2 * freq_resolution)
        peak_mask = np.abs(spectrum_freqs - freq) <= half_width
        peak_indices = np.flatnonzero(peak_mask)
        if peak_indices.size:
            peak_index = peak_indices[np.argmax(spectrum_magnitude[peak_indices])]
            marker_freq = float(spectrum_freqs[peak_index])
            peak = float(spectrum_magnitude[peak_index])
        else:
            marker_freq, peak = freq, 0.0
        # axvline의 ymin은 축 높이 비율이므로 피크 진폭을 0~1로 변환한다.
        bottom = float(np.clip((peak - y_min) / y_span + 0.005, 0.0, 1.0))
        return marker_freq, bottom

    if fault_type == FAULT_NORMAL:
        for fname in ("BPFO", "BPFI", "BSF"):
            style = DEFECT_FREQ_COLORS[fname]
            base = defect_freqs[fname][0] if defect_freqs[fname] else 0
            harmonics = _extended_harmonics(defect_freqs, fname, n_harmonics,
                                            freq_max)
            labeled = False
            for freq in harmonics:
                marker_freq, bottom = marker_position(freq)
                label = (
                    f"{fname} (theory: {base:.1f} Hz)\n"
                    f"Observed peak (1×): {marker_freq:.1f} Hz"
                ) if not labeled else None
                ax.axvline(
                    marker_freq, ymin=bottom, ymax=1.0,
                    color=style["color"], linestyle="--",
                    alpha=0.85, linewidth=1.2, zorder=1, label=label,
                )
                labeled = True
    else:
        freq_name = FAULT_TYPE_TO_FREQ.get(fault_type)
        if freq_name is None:
            return

        style = DEFECT_FREQ_COLORS[freq_name]
        base = defect_freqs[freq_name][0] if defect_freqs[freq_name] else 0
        harmonics = _extended_harmonics(defect_freqs, freq_name, n_harmonics,
                                        freq_max)
        for i, freq in enumerate(harmonics):
            marker_freq, bottom = marker_position(freq)
            label = (
                f"{freq_name} (theory: {base:.1f} Hz)\n"
                f"Observed peak (1×): {marker_freq:.1f} Hz"
            ) if i == 0 else None
            ax.axvline(
                marker_freq, ymin=bottom, ymax=1.0,
                color=style["color"], linestyle="--",
                alpha=0.7, linewidth=1.0, zorder=1, label=label,
            )


# ─────────────────────────────────────────────────────────────────
# 개별 포락선 스펙트럼 그래프
# ─────────────────────────────────────────────────────────────────
def plot_envelope_spectrum(
    data: dict,
    band: tuple[float, float],
    defect_freqs: dict,
    save: bool = True,
) -> Path | tuple:
    """
    단일 파일의 포락선 스펙트럼을 그래프로 생성한다.

    Parameters
    ----------
    data : dict
        load_dataset()으로 로드된 데이터
    band : tuple[float, float]
        Kurtogram에서 선택한 (low_hz, high_hz)
    defect_freqs : dict
        calculate_defect_frequencies() 반환값
    save : bool
        True → 파일 저장 후 경로 반환, False → (fig, ax) 반환
    """
    info = data["info"]
    signal, fs, rpm = data["signal"], data["fs"], data["rpm"]

    freqs, magnitude, _ = compute_envelope_spectrum(signal, fs, band)

    fig, ax = plt.subplots(figsize=(10, 4))
    mask = freqs <= PLOT_FREQ_MAX
    ax.plot(freqs[mask], magnitude[mask],
            color="#1f77b4", linewidth=0.6, alpha=0.9, zorder=3,
            label="Spectrum")
    ax.set_ylim(bottom=0)
    _add_defect_markers(
        ax, defect_freqs, info.fault_type, freqs[mask], magnitude[mask],
    )

    ax.set_xlim(0, PLOT_FREQ_MAX)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Amplitude (g)", fontsize=10)

    size_str = f' {info.fault_diameter}"' if info.fault_diameter else ""
    title = (
        f"Envelope Spectrum: {data['key']} - {info.fault_type}"
        f" Fault{size_str} "
        f"({info.hp}HP, {rpm:.0f} RPM)\n"
        f"Kurtogram {band[0]:.0f}~{band[1]:.0f} Hz"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if save:
        _ensure_dir(ENVELOPE_DIR)
        filepath = ENVELOPE_DIR / f"envelope_{data['key']}.png"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return filepath

    return fig, ax


# ─────────────────────────────────────────────────────────────────
# Kurtogram 맵
# ─────────────────────────────────────────────────────────────────
def plot_kurtogram_map(
    data: dict,
    kurtogram_result: dict,
    save: bool = True,
) -> Path | tuple:
    """
    Kurtogram 히트맵을 생성한다.

    선택된 최적 대역은 검은 테두리로 강조 표시한다.

    Parameters
    ----------
    data : dict
        load_dataset()으로 로드된 데이터
    kurtogram_result : dict
        compute_kurtogram() 반환값
    save : bool
        True → 파일 저장 후 경로 반환, False → (fig, ax) 반환
    """
    info = data["info"]
    rpm = data["rpm"]
    rows = kurtogram_result["rows"]
    best = kurtogram_result["best"]
    fs = kurtogram_result["fs"]

    all_scores = np.concatenate([row["kurtosis"] for row in rows])
    vmax = float(np.max(all_scores))
    if vmax <= 0:
        vmax = 1.0
    norm = plt.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.hot

    fig, ax = plt.subplots(figsize=(10, 5))

    for row in rows:
        level = row["level"]
        bw = row["bandwidth"]
        centers = row["centers"]
        scores = row["kurtosis"]
        for idx in range(len(centers)):
            low = centers[idx] - bw / 2
            color_val = norm(max(0, float(scores[idx])))
            rect = plt.Rectangle(
                (low, level - 0.4), bw, 0.8,
                facecolor=cmap(color_val),
                edgecolor="gray", linewidth=0.3,
            )
            ax.add_patch(rect)

    best_rect = plt.Rectangle(
        (best["low"], best["level"] - 0.45), best["bandwidth"], 0.9,
        facecolor="none", edgecolor="black", linewidth=2.5,
    )
    ax.add_patch(best_rect)

    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Decomposition Level (k)", fontsize=10)
    ax.set_yticks(range(len(rows)))

    size_str = f' {info.fault_diameter}"' if info.fault_diameter else ""
    title = (
        f"Kurtogram \u2014 {data['key']} ({info.fault_type}{size_str}, "
        f"{info.hp}HP, {rpm:.0f} RPM)\n"
        f"Best: {best['low']:.0f} \u2013 {best['high']:.0f} Hz "
        f"(\u03ba = {best['kurtosis']:.2f})"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Spectral Kurtosis")
    fig.tight_layout()

    if save:
        _ensure_dir(KURTOGRAM_DIR)
        filepath = KURTOGRAM_DIR / f"kurtogram_{data['key']}.png"
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return filepath

    return fig, ax


# ─────────────────────────────────────────────────────────────────
# 비교 그래프 — 공통 로직
# ─────────────────────────────────────────────────────────────────
def _plot_comparison(
    datasets: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    defect_freqs: dict[str, dict],
    suptitle: str,
    subtitle_fn,
    save_path: Path | None = None,
) -> Path | tuple:
    """비교 그래프의 공통 렌더링 로직."""
    keys = list(datasets.keys())
    n = len(keys)
    if n == 0:
        return save_path

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        data = datasets[key]
        signal, fs = data["signal"], data["fs"]
        band = bands[key]
        freqs, magnitude, _ = compute_envelope_spectrum(signal, fs, band)
        defect = defect_freqs[key]

        mask = freqs <= PLOT_FREQ_MAX
        ax.plot(freqs[mask], magnitude[mask],
                color="#1f77b4", linewidth=0.6, alpha=0.9, zorder=3,
                label="Spectrum")
        ax.set_ylim(bottom=0)
        _add_defect_markers(
            ax, defect, data["info"].fault_type, freqs[mask], magnitude[mask],
        )

        ax.set_xlim(0, PLOT_FREQ_MAX)
        ax.set_ylabel("Amplitude (g)", fontsize=9)
        ax.set_title(subtitle_fn(data), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Frequency (Hz)", fontsize=10)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig, axes


# ─────────────────────────────────────────────────────────────────
# 결함 유형별 비교
# ─────────────────────────────────────────────────────────────────
def plot_type_comparison(
    datasets: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    defect_freqs: dict[str, dict],
    condition_label: str,
    save: bool = True,
) -> Path | tuple:
    """
    같은 조건(HP + 결함 크기) 내 결함 유형별 포락선 비교.

    서브플롯 순서: Outer Race → Inner Race → Ball → Normal
    """
    sorted_keys = sorted(
        datasets.keys(),
        key=lambda k: _TYPE_ORDER.get(datasets[k]["info"].fault_type, 99),
    )
    ordered = {k: datasets[k] for k in sorted_keys}
    save_path = COMPARISON_DIR / f"compare_types_{condition_label}.png" if save else None

    def subtitle(data):
        s = f' {data["info"].fault_diameter}"' if data["info"].fault_diameter else ""
        return f'{data["key"]} \u2014 {data["info"].fault_type}{s}'

    return _plot_comparison(
        ordered, bands, defect_freqs,
        suptitle=f"Fault Type Comparison \u2014 {condition_label}",
        subtitle_fn=subtitle,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────
# 결함 크기별 비교
# ─────────────────────────────────────────────────────────────────
def plot_size_comparison(
    datasets: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    defect_freqs: dict[str, dict],
    type_label: str,
    hp: int,
    save: bool = True,
) -> Path | tuple:
    """
    같은 유형·HP에서 결함 크기별(0.007 / 0.014 / 0.021) 포락선 비교.

    서브플롯 순서: 결함 크기 오름차순
    """
    sorted_keys = sorted(
        datasets.keys(),
        key=lambda k: datasets[k]["info"].fault_diameter or "",
    )
    ordered = {k: datasets[k] for k in sorted_keys}
    save_path = (
        COMPARISON_DIR / f"compare_sizes_{type_label}_{hp}HP.png" if save else None
    )

    def subtitle(data):
        return f'{data["key"]} \u2014 {data["info"].fault_diameter}"'

    return _plot_comparison(
        ordered, bands, defect_freqs,
        suptitle=f"Size Comparison \u2014 {type_label} @ {hp}HP",
        subtitle_fn=subtitle,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────
# HP(RPM)별 비교
# ─────────────────────────────────────────────────────────────────
def plot_hp_comparison(
    datasets: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    defect_freqs: dict[str, dict],
    type_label: str,
    size: str,
    save: bool = True,
) -> Path | tuple:
    """
    같은 유형·크기에서 HP별(0 / 1 / 2 / 3) 포락선 비교.

    서브플롯 순서: HP 오름차순
    """
    sorted_keys = sorted(
        datasets.keys(),
        key=lambda k: datasets[k]["info"].hp,
    )
    ordered = {k: datasets[k] for k in sorted_keys}
    save_path = (
        COMPARISON_DIR / f"compare_hp_{type_label}_{size}.png" if save else None
    )

    def subtitle(data):
        return f'{data["key"]} \u2014 {data["info"].hp}HP ({data["rpm"]:.0f} RPM)'

    return _plot_comparison(
        ordered, bands, defect_freqs,
        suptitle=f"HP Comparison \u2014 {type_label} {size}\"",
        subtitle_fn=subtitle,
        save_path=save_path,
    )


# ─────────────────────────────────────────────────────────────────
# Phase 3 전체 그래프 일괄 생성
# ─────────────────────────────────────────────────────────────────
def generate_all_plots(
    catalog: dict,
    all_data: dict[str, dict],
    bands: dict[str, tuple[float, float]],
    kurtogram_results: dict[str, dict],
    defect_freqs: dict[str, dict],
) -> dict:
    """
    Phase 3의 모든 그래프를 일괄 생성한다.

    Parameters
    ----------
    catalog : dict[str, DatasetEntry]
    all_data : dict[str, dict]
    bands : dict[str, tuple[float, float]]
    kurtogram_results : dict[str, dict]
    defect_freqs : dict[str, dict]

    Returns
    -------
    dict
        {"envelope": n, "kurtogram": n, "type_comp": n,
         "size_comp": n, "hp_comp": n}  — 생성된 그래프 수
    """
    from cwru_analysis.catalog import (
        filter_catalog,
        get_all_conditions,
    )

    counts = {
        "envelope": 0, "kurtogram": 0,
        "type_comp": 0, "size_comp": 0, "hp_comp": 0,
    }

    # ── 1. 개별 포락선 스펙트럼 (40장) ──
    print("\n  [3-1] 개별 포락선 스펙트럼 생성 중...")
    for key, data in all_data.items():
        plot_envelope_spectrum(data, bands[key], defect_freqs[key])
        counts["envelope"] += 1
    print(f"        → {counts['envelope']}장 완료")

    # ── 2. 개별 Kurtogram 맵 (40장) ──
    print("  [3-2] 개별 Kurtogram 맵 생성 중...")
    for key, data in all_data.items():
        plot_kurtogram_map(data, kurtogram_results[key])
        counts["kurtogram"] += 1
    print(f"        → {counts['kurtogram']}장 완료")

    # ── 3. 결함 유형별 비교 (12장) ──
    print("  [3-3] 결함 유형별 비교 그래프 생성 중...")
    for diameter, hp in get_all_conditions():
        from cwru_analysis.catalog import get_experiment_datasets
        ds = get_experiment_datasets(catalog, diameter, hp, include_ball=True)
        sub_data = {k: all_data[k] for k in ds if k in all_data}
        sub_bands = {k: bands[k] for k in sub_data}
        sub_defect = {k: defect_freqs[k] for k in sub_data}
        if sub_data:
            plot_type_comparison(
                sub_data, sub_bands, sub_defect, f"{diameter}_{hp}HP",
            )
            counts["type_comp"] += 1
    print(f"        → {counts['type_comp']}장 완료")

    # ── 4. 결함 크기별 비교 (3타입 × 4HP = 12장) ──
    print("  [3-4] 결함 크기별 비교 그래프 생성 중...")
    for ft in (FAULT_OUTER, FAULT_INNER, FAULT_BALL):
        ts = FAULT_TYPE_SHORT[ft]
        for hp in range(4):
            entries = filter_catalog(catalog, fault_types=[ft], hp=hp)
            sub_data = {k: all_data[k] for k in entries if k in all_data}
            sub_bands = {k: bands[k] for k in sub_data}
            sub_defect = {k: defect_freqs[k] for k in sub_data}
            if sub_data:
                plot_size_comparison(sub_data, sub_bands, sub_defect, ts, hp)
                counts["size_comp"] += 1
    print(f"        → {counts['size_comp']}장 완료")

    # ── 5. HP(RPM)별 비교 (3타입 × 3크기 + Normal = 10장) ──
    print("  [3-5] HP(RPM)별 비교 그래프 생성 중...")
    for ft in (FAULT_OUTER, FAULT_INNER, FAULT_BALL):
        ts = FAULT_TYPE_SHORT[ft]
        for diameter in ("0.007", "0.014", "0.021"):
            entries = filter_catalog(
                catalog, fault_types=[ft], fault_diameter=diameter,
            )
            sub_data = {k: all_data[k] for k in entries if k in all_data}
            sub_bands = {k: bands[k] for k in sub_data}
            sub_defect = {k: defect_freqs[k] for k in sub_data}
            if sub_data:
                plot_hp_comparison(sub_data, sub_bands, sub_defect, ts, diameter)
                counts["hp_comp"] += 1

    normal_entries = filter_catalog(catalog, fault_types=[FAULT_NORMAL])
    sub_data = {k: all_data[k] for k in normal_entries if k in all_data}
    sub_bands = {k: bands[k] for k in sub_data}
    sub_defect = {k: defect_freqs[k] for k in sub_data}
    if sub_data:
        plot_hp_comparison(sub_data, sub_bands, sub_defect, "Normal", "all")
        counts["hp_comp"] += 1
    print(f"        → {counts['hp_comp']}장 완료")

    total = sum(counts.values())
    print(f"\n  Phase 3 그래프 총 {total}장 생성 완료")
    return counts
