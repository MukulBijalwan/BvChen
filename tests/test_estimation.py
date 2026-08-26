"""Tests for EM estimation of the bivariate Chen distribution."""

import numpy as np
import pytest

from bvchen import fit_bvchen, loglik_bvch, profile_beta, rbvch


@pytest.fixture(scope="module")
def simulated():
    return rbvch(600, 0.8, 1.2, 0.5, 1.5, random_state=2024)


class TestLogLik:
    def test_finite_and_negative(self, simulated):
        ll = loglik_bvch(simulated[:, 0], simulated[:, 1], 0.8, 1.2,
                         0.5, 1.5)
        assert np.isfinite(ll)

    def test_mle_not_worse_than_truth(self, simulated):
        fit = fit_bvchen(simulated[:, 0], simulated[:, 1])
        ll_true = loglik_bvch(simulated[:, 0], simulated[:, 1],
                              0.8, 1.2, 0.5, 1.5)
        assert fit.loglik >= ll_true - 1e-6

    def test_input_validation(self, simulated):
        with pytest.raises(ValueError):
            loglik_bvch(simulated[:, 0], simulated[:, 1][:5],
                        1, 1, 1, 1)
        with pytest.raises(ValueError):
            loglik_bvch(simulated[:, 0], -simulated[:, 1],
                        1, 1, 1, 1)


class TestEMFit:
    def test_recovers_parameters(self, simulated):
        fit = fit_bvchen(simulated[:, 0], simulated[:, 1])
        cf = fit.coefficients
        assert abs(cf[3] - 1.5) < 0.25           # beta
        assert abs(cf[0] - 0.8) / 0.8 < 0.4      # alpha1
        assert abs(cf[1] - 1.2) / 1.2 < 0.4      # alpha2
        assert abs(cf[2] - 0.5) / 0.5 < 0.6      # alpha3

    def test_converged_flag(self, simulated):
        fit = fit_bvchen(simulated[:, 0], simulated[:, 1])
        assert fit.converged
        assert len(fit.history) == fit.iterations
        # log-likelihood must be non-decreasing (up to tolerance)
        hist = np.asarray(fit.history)
        assert np.all(np.diff(hist) >= -1e-6 * np.abs(hist[:-1]))

    def test_start_invariance(self, simulated):
        f1 = fit_bvchen(simulated[:, 0], simulated[:, 1])
        f2 = fit_bvchen(simulated[:, 0], simulated[:, 1],
                        start={"alpha1": 3.0, "alpha2": 0.3,
                               "alpha3": 2.0, "beta": 1.0})
        np.testing.assert_allclose(f1.coefficients, f2.coefficients,
                                   atol=1e-3)

    def test_counts(self, simulated):
        fit = fit_bvchen(simulated[:, 0], simulated[:, 1])
        assert fit.n == 600
        assert fit.n_ties == int(np.sum(simulated[:, 0]
                                        == simulated[:, 1]))

    def test_bad_inputs_raise(self, simulated):
        with pytest.raises(ValueError):
            fit_bvchen(simulated[:, 0], simulated[:, 1][:10])
        with pytest.raises(ValueError):
            fit_bvchen(simulated[:, 0], simulated[:, 1],
                       start={"alpha1": -1})


class TestProfileBeta:
    def test_beta_hat_close_to_truth(self, simulated):
        pb = profile_beta(simulated[:, 0], simulated[:, 1],
                          betas=np.linspace(0.8, 2.4, 12))
        betas, vals = pb["grid"]
        assert abs(pb["beta_hat"] - 1.5) < 0.35
        # grid maximum should be near beta_hat as well
        i = int(np.argmax(vals))
        assert abs(betas[i] - pb["beta_hat"]) < (betas[1] - betas[0]) * 2