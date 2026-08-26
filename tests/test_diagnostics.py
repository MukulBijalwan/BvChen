"""Tests for goodness-of-fit diagnostics."""

import numpy as np

from bvchen import gof, rbvch
from bvchen.diagnostics import qqplot


class TestGOF:
    def test_does_not_reject_well_specified_data(self):
        sim = rbvch(800, 1.0, 1.0, 1.0, 1.5, random_state=5)
        g = gof(sim[:, 0], sim[:, 1], 1.0, 1.0, 1.0, 1.5)
        assert all(res.pvalue > 0.01 for res in g.values())

    def test_rejects_misspecified_data(self):
        # exponential(1) data are far from Chen(1, 1, 1, 1.5) marginals
        rng = np.random.default_rng(0)
        z = rng.exponential(scale=1.0, size=400)
        g = gof(z, z * 1.1, 1.0, 1.0, 1.0, 1.5)
        pvals = [res.pvalue for res in g.values()]
        assert max(pvals) < 0.05 or sorted(pvals)[1] < 0.05

    def test_result_keys(self):
        sim = rbvch(200, 1.0, 1.0, 1.0, 1.5, random_state=1)
        g = gof(sim[:, 0], sim[:, 1], 1.0, 1.0, 1.0, 1.5)
        assert set(g.keys()) == {"ks_z1", "ks_z2", "ks_min"}


class TestQQPlot:
    def test_runs_headless(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sim = rbvch(300, 1.0, 1.0, 1.0, 1.5, random_state=7)
        ax = qqplot(sim[:, 0], sim[:, 1], 1.0, 1.0, 1.0, 1.5)
        assert len(ax) == 3
        plt.close("all")

    def test_too_few_observations_raises(self):
        import pytest
        with pytest.raises(ValueError):
            qqplot([1.0], [2.0], 1, 1, 1, 1.5)