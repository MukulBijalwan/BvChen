"""Tests for the bivariate Chen distribution functions."""

import numpy as np
import pytest
from scipy import integrate

from bvchen import (BvChen, bvch_moment, dbvch, dcond1, dmarg1, hcond1,
                    kendall_tau, pbvch, pmarg1, pmarg2, rbvch, sbvch,
                    survcop)
from bvchen.chen import dchen, pchen, schen

A1, A2, A3, B = 0.8, 1.1, 0.6, 1.4
TOT = A1 + A2 + A3


class TestDensity:
    def test_diagonal_mass_equals_alpha3_over_total(self):
        m_diag, _ = integrate.quad(
            lambda z: dbvch(z, z, A1, A2, A3, B, component="sing"),
            0, np.inf)
        assert abs(m_diag - A3 / TOT) < 1e-4

    def test_region_masses(self):
        ys = np.linspace(0.02, 8, 200)
        iv = [integrate.quad(
            lambda x: float(dbvch(x, y, A1, A2, A3, B)), 0, y)[0]
            for y in ys]
        m_lt = np.trapezoid(iv, ys)
        assert abs(m_lt - A1 / TOT) < 5e-3

    def test_components_partition(self):
        # full = ac off diagonal; sing zero off diagonal and vice versa
        xs = np.array([0.5, 1.0, 2.0])
        full = dbvch(xs, xs * 1.5, A1, A2, A3, B)
        ac = dbvch(xs, xs * 1.5, A1, A2, A3, B, component="ac")
        np.testing.assert_allclose(full, ac, rtol=1e-12)
        sing_off = dbvch(xs, xs * 1.5, A1, A2, A3, B, component="sing")
        np.testing.assert_array_equal(sing_off, 0.0)

    def test_ac_factorises_as_product_of_chens(self):
        z1, z2 = 0.5, 1.5          # z1 < z2 region
        expected = dchen(z1, A1, B) * dchen(z2, A2 + A3, B)
        assert abs(float(dbvch(z1, z2, A1, A2, A3, B)) - expected) < 1e-12
        z1, z2 = 1.5, 0.5          # z1 > z2 region
        expected = dchen(z2, A2, B) * dchen(z1, A1 + A3, B)
        assert abs(float(dbvch(z1, z2, A1, A2, A3, B)) - expected) < 1e-12

    def test_invalid_component_raises(self):
        with pytest.raises(ValueError):
            dbvch(1.0, 1.0, A1, A2, A3, B, component="bogus")


    def test_2d_meshgrid_shape_preserved(self):
        # regression: dbvch/pbvch/sbvch must keep the broadcasted meshgrid
        # shape rather than collapsing to a flat 1-D buffer
        g = np.linspace(0.05, 4.0, 32)
        X, Y = np.meshgrid(g, g, indexing="ij")
        for fn, comp in [(dbvch, "full"), (dbvch, "ac"), (dbvch, "sing"),
                         (pbvch, None), (sbvch, None)]:
            if comp is None:
                out = fn(X, Y, A1, A2, A3, B)
            else:
                out = fn(X, Y, A1, A2, A3, B, component=comp)
            assert out.shape == X.shape
            assert np.isfinite(out).all()


class TestCDFSurvival:
    def test_inclusion_exclusion_identity(self):
        for z1, z2 in [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0)]:
            lhs = pbvch(z1, z2, A1, A2, A3, B)
            rhs = (1 - schen(z1, A1 + A3, B) - schen(z2, A2 + A3, B)
                   + sbvch(z1, z2, A1, A2, A3, B))
            assert abs(lhs - rhs) < 1e-10

    def test_boundaries(self):
        assert sbvch(0, 0, A1, A2, A3, B) == 1.0
        assert sbvch(np.inf, np.inf, A1, A2, A3, B) < 1e-12
        assert pbvch(np.inf, np.inf, A1, A2, A3, B) == 1.0
        assert pbvch(0, 0, A1, A2, A3, B) >= 0.0

    def test_cdf_monotone(self):
        grid = np.linspace(0.1, 4.0, 40)
        vals = pbvch(grid, grid * 1.2, A1, A2, A3, B)
        assert np.all(np.diff(vals) >= -1e-12)


class TestMarginals:
    def test_marginals_are_chen(self):
        x = np.linspace(0.05, 3.0, 50)
        np.testing.assert_allclose(dmarg1(x, A1, A2, A3, B),
                                   dchen(x, A1 + A3, B), rtol=1e-12)
        np.testing.assert_allclose(pmarg2(x, A1, A2, A3, B),
                                   pchen(x, A2 + A3, B), rtol=1e-12)


class TestConditionals:
    def test_conditional_normalises_to_one(self):
        y = 1.0
        below, _ = integrate.quad(
            lambda x: np.ravel(dcond1(x, y, A1, A2, A3, B))[0], 0, y)
        above, _ = integrate.quad(
            lambda x: np.ravel(dcond1(x, y, A1, A2, A3, B))[0], y, np.inf)
        atom = dcond1(y, y, A1, A2, A3, B).atom
        assert abs(below + above + atom - 1.0) < 1e-4

    def test_continuous_part_below_is_free_of_y(self):
        y1, y2 = 0.8, 1.5
        z = 0.4
        np.testing.assert_allclose(np.ravel(dcond1(z, y1, A1, A2, A3, B)),
                                   np.ravel(dcond1(z, y2, A1, A2, A3, B)),
                                   rtol=1e-12)

    def test_hazard_nan_on_diagonal(self):
        h = hcond1(np.array([0.5, 1.0, 1.5]), 1.0, A1, A2, A3, B)
        assert np.isnan(h[1])
        assert np.isfinite(h[0]) and np.isfinite(h[2])

class TestRandom:
    def test_tie_rate(self):
        sim = rbvch(6000, A1, A2, A3, B, random_state=42)
        rate = np.mean(sim[:, 0] == sim[:, 1])
        assert abs(rate - A3 / TOT) < 0.03

    def test_shape_and_positivity(self):
        sim = rbvch(100, 1.0, 1.0, 1.0, 1.0, random_state=0)
        assert sim.shape == (100, 2)
        assert np.all(sim > 0)

    def test_marginal_ks(self):
        from scipy import stats
        sim = rbvch(3000, 1.0, 1.5, 0.7, 1.4, random_state=42)
        ks = stats.ks_1samp(sim[:, 0],
                            lambda q: pmarg1(q, 1.0, 1.5, 0.7, 1.4))
        assert ks.pvalue > 0.01
        ks_min = stats.ks_1samp(
            np.minimum(sim[:, 0], sim[:, 1]),
            lambda q: pchen(q, 3.2, 1.4))
        assert ks_min.pvalue > 0.01


class TestCopulaAndTau:
    def test_copula_corners(self):
        assert survcop(1, 1, 1, 1, 1) == 1.0
        assert survcop(0, 0.5, 1, 1, 1) == 0.0
        assert survcop(1, 0, 1, 1, 1) == 0.0

    def test_tau_increases_with_common_shock(self):
        tau_small = kendall_tau(1, 1, 0.01, 1.4, n=40000, random_state=9)
        tau_big = kendall_tau(1, 1, 10.0, 1.4, n=40000, random_state=9)
        assert tau_big > tau_small

    def test_tau_range(self):
        tau = kendall_tau(1, 1, 1, 1.5, n=30000, random_state=3)
        assert -0.1 <= tau <= 0.6


class TestMoments:
    def test_total_probability_moments_sum(self):
        m00 = bvch_moment(0, 0, A1, A2, A3, B)
        assert abs(m00 - 1.0) < 1e-12

    def test_moments_match_simulation(self):
        sim = rbvch(120000, A1, A2, A3, B, random_state=11)
        e1 = bvch_moment(1, 0, A1, A2, A3, B)
        e2 = bvch_moment(0, 1, A1, A2, A3, B)
        e11 = bvch_moment(1, 1, A1, A2, A3, B)
        assert abs(e1 - sim[:, 0].mean()) / e1 < 0.03
        assert abs(e2 - sim[:, 1].mean()) / e2 < 0.03
        assert abs(e11 - np.mean(sim[:, 0] * sim[:, 1])) / e11 < 0.05

    def test_marginal_moments_consistent_with_univariate(self):
        from bvchen.chen import chen_moment
        e1 = bvch_moment(1, 0, A1, A2, A3, B)
        ref = chen_moment(1, A1 + A3, B)
        assert abs(e1 - ref) < 1e-6


class TestClassInterface:
    def test_roundtrip(self):
        dist = BvChen(A1, A2, A3, B)
        assert dist.params == (A1, A2, A3, B)
        assert dist.alpha == A3
        x, y = 0.7, 1.3
        assert dist.pdf(x, y) == dbvch(x, y, A1, A2, A3, B)
        assert dist.cdf(x, y) == pbvch(x, y, A1, A2, A3, B)
        assert dist.sf(x, y) == sbvch(x, y, A1, A2, A3, B)

    def test_min_distribution(self):
        dist = BvChen(A1, A2, A3, B)
        mn = dist.min_dist()
        q = 1.0
        assert mn['p'](q) == pchen(q, TOT, B)

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            BvChen(-1, 1, 1, 1)
        with pytest.raises(ValueError):
            BvChen(1, 1, 1, 0)
