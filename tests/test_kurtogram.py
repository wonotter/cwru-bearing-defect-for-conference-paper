"""Antoni MATLAB 기준값과 프로젝트의 최소 대역폭 정책을 검증한다."""

from pathlib import Path
import unittest

import numpy as np

from cwru_analysis.kurtogram import (
    _design_filter_bank,
    _spectral_kurtosis,
    compute_kurtogram,
    select_best_band,
)


REFERENCE_PATH = Path(__file__).parent / "data" / "antoni_reference.npz"


class KurtogramTests(unittest.TestCase):
    """이진·삼진 행 전체를 독립적으로 실행한 MATLAB 결과와 비교한다."""

    @classmethod
    def setUpClass(cls):
        with np.load(REFERENCE_PATH) as reference:
            cls.reference = {key: reference[key] for key in reference.files}

    def test_analysis_filters_match_reference(self):
        for name, actual in zip(("h", "g", "h1", "h2", "h3"), _design_filter_bank()):
            with self.subTest(filter=name):
                np.testing.assert_allclose(
                    actual, self.reference[name], rtol=1e-13, atol=1e-14,
                )

    def test_all_rows_match_reference(self):
        for name in ("white_noise", "impacts", "odd_length", "level_one", "level_two"):
            with self.subTest(case=name):
                expected = self.reference[f"{name}_scores"]
                result = compute_kurtogram(
                    self.reference[f"{name}_signal"], 12000.0,
                    nlevel=int(self.reference[f"{name}_nlevel"]), min_bandwidth=0.0,
                )
                # MATLAB은 각 칸의 SK를 같은 폭의 표시 열에 반복해 저장한다.
                actual = np.vstack([
                    np.repeat(row["kurtosis"], expected.shape[1] // len(row["kurtosis"]))
                    for row in result["rows"]
                ])
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_minimum_bandwidth_matches_masked_reference(self):
        signal = self.reference["impacts_signal"]
        result = compute_kurtogram(signal, 12000.0)
        expected = self.reference["impacts_scores"].copy()
        band_counts = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
        widths = 6000.0 / band_counts
        expected[widths < 600.0] = -np.inf

        # MATLAB max_IJ는 열을 먼저 순회하며 동률이면 위쪽 행을 택한다.
        row_idx, column_idx = np.unravel_index(
            np.argmax(expected.ravel(order="F")), expected.shape, order="F",
        )
        band_idx = column_idx // (expected.shape[1] // band_counts[row_idx])
        low = band_idx * widths[row_idx]
        best = result["best"]
        self.assertEqual(best["bandwidth"], widths[row_idx])
        self.assertEqual(best["low"], low)
        self.assertEqual(best["high"], low + widths[row_idx])
        self.assertEqual(select_best_band(signal, 12000.0), (best["low"], best["high"]))

    def test_real_and_complex_gaussian_corrections(self):
        real = np.tile([-1.0, 1.0], 128)
        complex_signal = np.tile([1.0, -1.0, 1j, -1j], 64)
        self.assertAlmostEqual(_spectral_kurtosis(real), -2.0)
        self.assertAlmostEqual(_spectral_kurtosis(complex_signal), -1.0)
        self.assertAlmostEqual(_spectral_kurtosis(complex_signal + 10.0 + 2j), -1.0)

    def test_level_zero_uses_real_signal_kurtosis(self):
        signal = self.reference["impacts_signal"]
        result = compute_kurtogram(signal, 12000.0, nlevel=0)
        self.assertEqual(len(result["rows"]), 1)
        self.assertAlmostEqual(
            result["best"]["kurtosis"], self.reference["impacts_scores"][0, 0],
        )
        self.assertEqual(select_best_band(signal, 12000.0, nlevel=0), (0.0, 6000.0))

    def test_zero_power_and_negative_scores_are_clipped(self):
        result = compute_kurtogram(np.ones(4096), 12000.0)
        for row in result["rows"]:
            np.testing.assert_array_equal(row["kurtosis"], np.zeros(len(row["centers"])))
        self.assertEqual(result["best"]["level"], 0.0)

    def test_short_signal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "길이 조건"):
            compute_kurtogram(np.ones(4095), 12000.0, nlevel=5)

    def test_invalid_minimum_bandwidth_is_rejected(self):
        signal = np.ones(4096)
        with self.assertRaisesRegex(ValueError, "선택할 수 있는 구간"):
            compute_kurtogram(signal, 12000.0, min_bandwidth=6001.0)
        with self.assertRaisesRegex(ValueError, "min_bandwidth"):
            compute_kurtogram(signal, 12000.0, min_bandwidth=float("nan"))


if __name__ == "__main__":
    unittest.main()
