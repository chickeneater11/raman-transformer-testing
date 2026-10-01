"""Checks for leakage, determinism, and the spectral diagnostic calculations."""
import unittest
import numpy as np

from spectral_diagnostics import (
    CLASSES, DiagnosticConfig, correlation_matrix, fit_shared_response,
    interpolate_checked, leave_one_molecule_out, stable_indices,
)


class SpectralDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DiagnosticConfig()
        w = np.linspace(0, 1, 1001)
        self.x = np.stack([(j + 1) * (1 + w) + .05 * np.sin((j + 1) * w) for j in range(7)])
        self.y = 2 * self.x + 3
        self.sd = np.full_like(self.x, .1)

    def test_known_shared_affine_response(self):
        result = leave_one_molecule_out({"Renishaw": self.x, "Horiba": self.y},
                                       {"Renishaw": self.sd, "Horiba": self.sd}, self.cfg)
        np.testing.assert_allclose(result["prediction"], self.y, rtol=1e-5, atol=1e-4)

    def test_held_out_target_cannot_change_its_fit(self):
        means = {"Renishaw": self.x, "Horiba": self.y.copy()}
        sd = {"Renishaw": self.sd, "Horiba": self.sd}
        before = leave_one_molecule_out(means, sd, self.cfg)
        means["Horiba"][2] += 1e6
        after = leave_one_molecule_out(means, sd, self.cfg)
        for key in ("gain", "offset", "prediction", "low_information"):
            np.testing.assert_array_equal(before[key][2], after[key][2])

    def test_degenerate_reference_contrast_is_flagged(self):
        x = np.ones_like(self.x)
        a, b, low = fit_shared_response(x, self.y, self.sd, self.cfg)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        self.assertTrue(low.all())

    def test_seed_is_independent_of_global_rng_and_call_order(self):
        expected = stable_indices(1000, 750, 42, "Renishaw/adenine")
        np.random.seed(9834)
        np.random.normal(size=10000)
        stable_indices(1000, 750, 42, "Horiba/RNA")
        actual = stable_indices(1000, 750, 42, "Renishaw/adenine")
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(np.unique(actual)), 750)
        self.assertFalse(np.array_equal(actual, stable_indices(1000, 750, 43, "Renishaw/adenine")))

    def test_grid_sorting_and_no_extrapolation(self):
        grid = np.arange(450, 455)
        out = interpolate_checked(grid[::-1], np.array([grid[::-1], 2 * grid[::-1]]), grid, "synthetic")
        np.testing.assert_array_equal(out, np.array([grid, 2 * grid]))
        with self.assertRaisesRegex(ValueError, "refusing extrapolation"):
            interpolate_checked(grid, out, np.arange(449, 455), "synthetic")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            interpolate_checked([450, 451, 451, 454], np.ones((2, 4)), grid, "synthetic")

    def test_shape_correlation_separates_positive_gain_and_offset(self):
        np.testing.assert_allclose(np.diag(correlation_matrix(self.x, self.y)), np.ones(len(CLASSES)), atol=1e-14)
        self.assertTrue(np.isnan(correlation_matrix(np.ones((1, 10)), np.ones((1, 10)))).all())


if __name__ == "__main__":
    unittest.main()
