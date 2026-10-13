"""
CWRU 베어링 결함 주파수 분석 및 규칙 분류기 -- 메인 실행 스크립트.

Phase별 구현이 완성될 때마다 여기에 실행 코드가 추가된다.
"""

from cwru_analysis.catalog import (
    build_catalog,
    get_all_conditions,
    get_experiment_datasets,
    print_catalog_summary,
)
from cwru_analysis.frequencies import calculate_defect_frequencies, print_defect_frequencies
from cwru_analysis.kurtogram import compute_kurtogram, select_best_band
from cwru_analysis.envelope import compute_envelope_spectrum
from cwru_analysis.loading import load_dataset, load_datasets, print_dataset_summary


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
        from cwru_analysis.config import FAULT_TYPE_TO_FREQ
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


def main():
    catalog, all_data = phase1()
    phase2(catalog, all_data)


if __name__ == "__main__":
    main()
