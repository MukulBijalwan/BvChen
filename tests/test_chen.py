"""Tests for the univariate Chen distribution."""

import numpy as np
import pytest
from scipy import integrate

from bvchen.chen import (chen_moment, dchen, fit_chen, hchen, pchen,
                         qchen, rchen, schen)


class TestChenDensity:
    def test_integrates_to_one(self):
        val, _ = integrate.quad(lambda x: dchen(x, 1.7, 1.3), 0, np.inf)
        assert abs(val - 1.0) < 1e-8

    def test_integrates_to_one_beta_below_one(self):
        # bathtub-shaped hazard regime
        val, _ = integrate.quad(lambda x: dchen(x, 0.9, 0.5), 0, np.inf)
        assert abs(val - 1.0) < 1e-8

    def test_log_density_matches(self):
        x = np.linspace(0.05, 3.0, 50)
        np.testing.assert_allclose(np.log(dchen(x, 1.2, 0.9)),
                                   dchen(x, 1.2, 0.9, log=True),
                                   rtol=1e-10)

    def test_cdf_survival_complement(self):
        x = np.linspace(0.01, 5.0, 100)
        np.testing.assert_allclose(pchen(x, 1.7, 1.3)
                                   + schen(x, 1.7, 1.3), 1.0,
                                   atol=1e-12)

    def test_pdf_is_derivative_of_cdf(self):
        eps = 1e-6
        x = 1.23
        numeric = (pchen(x + eps, 1.7, 1.3) - pchen(x - eps, 1.7, 1.3)) \
            / (2 * eps)
        assert abs(numeric - float(dchen(x, 1.7, 1.3))) < 1e-5

    def test_hazard_ratio(self):
        x = np.linspace(0.05, 2.0, 20)
        np.testing.assert_allclose(hchen(x, 1.7, 1.3),
                                   dchen(x, 1.7, 1.3) / schen(x, 1.7, 1.3),
                                   rtol=1e-10)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            dchen(1.0, -1.0, 1.0)
        with pytest.raises(ValueError):
            pchen(1.0, 1.0, 0.0)


class TestChenQuantile:
    def test_quantile_inverts_cdf(self):
        p = np.array([0.05, 0.2, 0.5, 0.9, 0.99])
        np.testing.assert_allclose(pchen(qchen(p, 1.7, 1.3), 1.7, 1.3),
                                   p, atol=1e-10)

    def test_zero_quantile_is_zero(self):
        assert qchen(0.0, 1.7, 1.3) == 0.0


class TestChenRandom:
    def test_moments_match_simulation(self):
        rng = np.random.default_rng(42)
        x = rchen(200_000, 2.0, 1.5, random_state=rng)
        m1 = chen_moment(1, 2.0, 1.5)
        assert abs(x.mean() - m1) / m1 < 0.02
        m2 = chen_moment(2, 2.0, 1.5)
        assert abs(x.var() - (m2 - m1 ** 2)) / (m2 - m1 ** 2) < 0.05

    def test_reproducible_with_seed(self):
        a = rchen(50, 2.0, 1.5, random_state=7)
        b = rchen(50, 2.0, 1.5, random_state=7)
        np.testing.assert_array_equal(a, b)


class TestChenFit:
    def test_recovers_parameters(self):
        x = rchen(1000, 2.0, 1.5, random_state=1)
        fit = fit_chen(x)
        assert fit["converged"]
        assert abs(fit["alpha"] - 2.0) / 2.0 < 0.25
        assert abs(fit["beta"] - 1.5) / 1.5 < 0.2

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            fit_chen(np.array([-1.0, 2.0]))
        with pytest.raises(ValueError):
            fit_chen(np.array([]))
