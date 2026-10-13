"""
CWRU 베어링 결함 주파수 분석 및 규칙 분류기 — 메인 실행 스크립트.

Phase 1~5를 순차적으로 실행한다.
  Phase 1: 카탈로그 구축 + 데이터 로드 검증
  Phase 2: 신호 처리 파이프라인 검증
  Phase 3: 논문용 포락선 스펙트럼 그래프 생성
  Phase 4: 규칙 분류기 실행 (실험 A: 볼 포함 / 실험 B: 볼 제외)
  Phase 5: 성능 평가 + 결과 저장
"""

import time

from cwru_analysis.catalog import (
    build_catalog,
    filter_catalog,
    get_all_conditions,
    get_experiment_datasets,
    print_catalog_summary,
)
from cwru_analysis.classifier import run_experiment
from cwru_analysis.config import (
    FAULT_BALL,
    FAULT_INNER,
    FAULT_NORMAL,
    FAULT_OUTER,
    FAULT_TYPE_SHORT,
    FAULT_TYPE_TO_FREQ,
)
from cwru_analysis.envelope import compute_envelope_spectrum
from cwru_analysis.evaluation import run_evaluation
from cwru_analysis.frequencies import calculate_defect_frequencies, print_defect_frequencies
from cwru_analysis.kurtogram import compute_kurtogram, select_best_band
from cwru_analysis.loading import load_dataset, load_datasets, print_dataset_summary
from cwru_analysis.plotting import generate_all_plots


def phase1():
    """Phase 1: 카탈로그 구축 + 전체 데이터 로드 검증."""
    print("\n" + "=" * 70)
    print("  Phase 1: 카탈로그 구축 및 데이터 로드 검증")
    print("=" * 70)

    catalog = build_catalog()
    print_catalog_summary(catalog)

    print("[로드 테스트] 전체 데이터셋 로드 중...")
    all_data = load_datasets(catalog)
    for data in all_data.values():
        print_dataset_summary(data)

    print(f"\n[필터링 테스트] 실험 조건 {len(get_all_conditions())}개")
    for diameter, hp in get_all_conditions():
        ds_a = get_experiment_datasets(catalog, diameter, hp, include_ball=True)
        ds_b = get_experiment_datasets(catalog, diameter, hp, include_ball=False)
        print(f"  {diameter}\" / {hp}HP: "
              f"A={len(ds_a)}개({', '.join(ds_a.keys())}), "
              f"B={len(ds_b)}개({', '.join(ds_b.keys())})")

    print(f"\n  Phase 1 완료: {len(all_data)}개 파일 로드 성공\n")
    return catalog, all_data


def phase2(catalog, all_data):
    """Phase 2: 신호 처리 파이프라인 검증."""
    print("\n" + "=" * 70)
    print("  Phase 2: 신호 처리 파이프라인 검증")
    print("=" * 70)

    # 2-1. 결함 주파수 계산 (대표 RPM 4개)
    print("\n[2-1] 결함 주파수 계산")
    for hp, rpm in ((0, 1797), (1, 1772), (2, 1750), (3, 1730)):
        print_defect_frequencies(rpm)

    # 2-2. Kurtogram + 포락선 스펙트럼 (대표 4종: 0.007" / 0HP)
    print("\n[2-2] Kurtogram + 포락선 스펙트럼 (0.007\" / 0HP 대표)")
    test_keys = ["OR007@6_0", "IR007_0", "B007_0", "Normal_0"]
    for key in test_keys:
        data = all_data[key]
        signal, fs, rpm = data["signal"], data["fs"], data["rpm"]

        band = select_best_band(signal, fs)
        freqs, magnitude, envelope = compute_envelope_spectrum(signal, fs, band)

        defect = calculate_defect_frequencies(rpm)
        shaft = defect["shaft_freq"]

        print(f"\n  {key} (RPM={rpm:.0f}, shaft={shaft:.2f} Hz)")
        print(f"    Kurtogram 선택 대역: {band[0]:.0f} ~ {band[1]:.0f} Hz")
        print(f"    포락선 스펙트럼: {len(freqs)} bins, "
              f"freq_resolution={freqs[1]-freqs[0]:.3f} Hz, "
              f"max_freq={freqs[-1]:.0f} Hz")
        print(f"    엔벨로프 길이: {len(envelope)} samples")

        fault_type = data["info"].fault_type
        freq_name = FAULT_TYPE_TO_FREQ.get(fault_type)
        if freq_name:
            harmonics = defect[freq_name]
            print(f"    {freq_name} 고조파 위치:")
            for i, h in enumerate(harmonics):
                idx = int(round(h / (freqs[1] - freqs[0])))
                amp = magnitude[idx] if 0 <= idx < len(magnitude) else 0.0
                tag = " *even" if freq_name == "BSF" and (i+1) % 2 == 0 else ""
                print(f"      {i+1}x = {h:7.2f} Hz -> amp = {amp:.6f}{tag}")

    # 2-3. 전체 40개 파일 Kurtogram 일괄 검증
    print("\n[2-3] 전체 40개 파일 Kurtogram 대역 요약")
    print(f"  {'Key':<16s} {'RPM':>6s} {'Band Low':>9s} {'Band High':>10s} "
          f"{'BW':>6s} {'Level':>5s} {'Kurtosis':>9s}")
    print(f"  {'-'*16} {'-'*6} {'-'*9} {'-'*10} {'-'*6} {'-'*5} {'-'*9}")

    for key, data in all_data.items():
        result = compute_kurtogram(data["signal"], data["fs"])
        best = result["best"]
        print(f"  {key:<16s} {data['rpm']:>6.0f} "
              f"{best['low']:>9.0f} {best['high']:>10.0f} "
              f"{best['bandwidth']:>6.0f} {best['level']:>5d} "
              f"{best['kurtosis']:>9.2f}")

    print(f"\n  Phase 2 완료\n")


def phase3(catalog, all_data):
    """Phase 3: 논문용 포락선 스펙트럼 그래프 생성."""
    print("\n" + "=" * 70)
    print("  Phase 3: 논문용 포락선 스펙트럼 그래프 생성")
    print("=" * 70)

    t0 = time.time()

    # 3-1. 전체 파일에 대해 Kurtogram 대역 + 결함 주파수 계산
    print("\n  [3-0] Kurtogram 대역 및 결함 주파수 일괄 계산 중...")
    bands: dict[str, tuple[float, float]] = {}
    kurtogram_results: dict[str, dict] = {}
    defect_freqs: dict[str, dict] = {}

    for key, data in all_data.items():
        result = compute_kurtogram(data["signal"], data["fs"])
        bands[key] = (result["best"]["low"], result["best"]["high"])
        kurtogram_results[key] = result
        defect_freqs[key] = calculate_defect_frequencies(data["rpm"])
    print(f"        → {len(bands)}개 파일 계산 완료")

    # 3-2. 모든 그래프 생성
    counts = generate_all_plots(
        catalog, all_data, bands, kurtogram_results, defect_freqs,
    )

    elapsed = time.time() - t0
    print(f"\n  Phase 3 완료 ({elapsed:.1f}초 소요)\n")
    return bands, kurtogram_results, defect_freqs


def phase4(all_data, bands):
    """Phase 4: 규칙 분류기 실행."""
    print("\n" + "=" * 70)
    print("  Phase 4: 규칙 분류기 실행")
    print("=" * 70)

    t0 = time.time()

    # 실험 A: 볼 포함 4종 분류
    print("\n  [4-A] 실험 A 실행 (Outer Race / Inner Race / Ball / Normal)")
    results_a = run_experiment(all_data, bands, include_ball=True)

    # 실험 B: 볼 제외 3종 분류
    print("\n  [4-B] 실험 B 실행 (Outer Race / Inner Race / Normal)")
    results_b = run_experiment(all_data, bands, include_ball=False)

    elapsed = time.time() - t0
    print(f"\n  Phase 4 완료 ({elapsed:.1f}초 소요)\n")
    return results_a, results_b


def phase5(results_a, results_b):
    """Phase 5: 성능 평가 + 결과 저장."""
    print("\n" + "=" * 70)
    print("  Phase 5: 성능 평가 및 결과 저장")
    print("=" * 70)

    t0 = time.time()

    # 실험 A 평가
    eval_a = run_evaluation(results_a, include_ball=True,
                            experiment_label="with_ball")

    # 실험 B 평가
    eval_b = run_evaluation(results_b, include_ball=False,
                            experiment_label="without_ball")

    # 결과 비교 요약
    print("\n" + "-" * 50)
    print("  실험 A vs B 비교")
    print("-" * 50)
    oa = eval_a["overall"]
    ob = eval_b["overall"]
    print(f"  실험 A (볼 포함 4종):  Accuracy={oa['accuracy']:.4f}, "
          f"Macro F1={oa['macro_f1']:.4f}")
    print(f"  실험 B (볼 제외 3종):  Accuracy={ob['accuracy']:.4f}, "
          f"Macro F1={ob['macro_f1']:.4f}")

    print(f"\n  클래스별 F1 비교:")
    print(f"    {'Class':<15s} {'Exp A':>8s} {'Exp B':>8s}")
    print(f"    {'-'*15} {'-'*8} {'-'*8}")
    all_classes = sorted(set(list(oa["f1"].keys()) + list(ob["f1"].keys())))
    for cls in all_classes:
        fa = oa["f1"].get(cls, "-")
        fb = ob["f1"].get(cls, "-")
        fa_str = f"{fa:.4f}" if isinstance(fa, float) else fa
        fb_str = f"{fb:.4f}" if isinstance(fb, float) else fb
        print(f"    {cls:<15s} {fa_str:>8s} {fb_str:>8s}")

    elapsed = time.time() - t0
    print(f"\n  Phase 5 완료 ({elapsed:.1f}초 소요)\n")
    return eval_a, eval_b


def main():
    catalog, all_data = phase1()
    phase2(catalog, all_data)
    bands, kurtogram_results, defect_freqs = phase3(catalog, all_data)
    results_a, results_b = phase4(all_data, bands)
    eval_a, eval_b = phase5(results_a, results_b)

    print("\n" + "=" * 70)
    print("  모든 Phase 완료")
    print("=" * 70)
    print("  결과 위치: results/")
    print("    - results/envelope_spectra/   (포락선 스펙트럼 40장)")
    print("    - results/kurtograms/         (Kurtogram 맵 40장)")
    print("    - results/comparisons/        (비교 그래프)")
    print("    - results/classification/     (분류 결과 + 평가)")
    print("      ├── with_ball/              (실험 A)")
    print("      └── without_ball/           (실험 B)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
