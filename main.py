"""
CWRU 베어링 결함 주파수 분석 및 규칙 분류기 — 메인 실행 스크립트.

Phase별 구현이 완성될 때마다 여기에 실행 코드가 추가된다.
현재: Phase 1 — 카탈로그 구축 및 데이터 로드 검증.
"""

from cwru_analysis.catalog import (
    build_catalog,
    get_all_conditions,
    get_experiment_datasets,
    print_catalog_summary,
)
from cwru_analysis.loading import load_datasets, print_dataset_summary


def main():
    # ──────────────────────────────────────────────────────────
    # Phase 1: 카탈로그 구축 + 전체 데이터 로드 검증
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Phase 1: 카탈로그 구축 및 데이터 로드 검증")
    print("=" * 70)

    catalog = build_catalog()
    print_catalog_summary(catalog)

    # 전체 40개 파일 로드 테스트
    print("\n[로드 테스트] 전체 데이터셋 로드 중...")
    all_data = load_datasets(catalog)
    for data in all_data.values():
        print_dataset_summary(data)

    # 실험 조건별 필터링 테스트
    print(f"\n[필터링 테스트] 실험 조건 {len(get_all_conditions())}개")
    for diameter, hp in get_all_conditions():
        # 실험 A (볼 포함)
        ds_a = get_experiment_datasets(catalog, diameter, hp, include_ball=True)
        # 실험 B (볼 제외)
        ds_b = get_experiment_datasets(catalog, diameter, hp, include_ball=False)
        print(f"  {diameter}\" / {hp}HP: "
              f"실험A={len(ds_a)}개({', '.join(ds_a.keys())}), "
              f"실험B={len(ds_b)}개({', '.join(ds_b.keys())})")

    print("\n" + "=" * 70)
    print(f"  Phase 1 완료: {len(all_data)}개 파일 로드 성공")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
