import unittest

from scipy.stats import norm, multivariate_normal
from numpy.testing import assert_allclose

from biolucia.model import *
from biolucia.experiment import InitialValueExperiment
from biolucia.observation import LinearWeightedSumOfSquaresObservation, AffineMeasurementUncertainty

from tests.test_models import equilibrium_model


def assert_finite_difference_almost_equal(fun, jac, k, rtol=1e-6, atol=0.0, rdiff=1e-8, adiff=0.0):
    f0 = fun(k)
    D_analytic = jac(k)
    D_finite = np.empty_like(D_analytic)

    for i in range(len(k)):
        ki = k.copy()
        diff = ki[i] * rdiff + adiff
        ki[i] = ki[i] + diff

        f = fun(ki)
        D_finite[..., i] = (f - f0) / diff

    assert_allclose(D_analytic, D_finite, rtol, atol)


def assert_log_probability_dk(m, con, obs, measurements, **kwargs):
    def fun(k):
        ks_updated = OrderedDict(zip(m.default_parameters().keys(), k))
        m_updated = m.update_parameters(ks_updated)
        sim_updated = m_updated.simulate(con, parameters=m.default_parameters())
        return obs.log_probability(sim_updated, measurements)

    def jac(k):
        ks_updated = OrderedDict(zip(m.default_parameters().keys(), k))
        m_updated = m.update_parameters(ks_updated)
        sim_updated = m_updated.simulate(con, parameters=m.default_parameters())
        return obs.log_probability_dk(sim_updated, measurements)

    assert_finite_difference_almost_equal(fun, jac, [*m.default_parameters().values()], **kwargs)


class ObservationTestCase(unittest.TestCase):
    def test_log_probability(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = m.simulate(con)

        unc = AffineMeasurementUncertainty(2, 0)

        obs = LinearWeightedSumOfSquaresObservation([0], ['A'], unc)
        self.assertAlmostEqual(obs.probability(sim, m['A0'].value), norm.pdf(m['A0'].value, m['A0'].value, 2))
        self.assertAlmostEqual(obs.log_probability(sim, m['A0'].value), norm.logpdf(m['A0'].value, m['A0'].value, 2))

        obs = LinearWeightedSumOfSquaresObservation([0], ['A'], unc)
        self.assertAlmostEqual(obs.probability(sim, m['A0'].value + 2), norm.pdf(m['A0'].value + 2, m['A0'].value, 2))
        self.assertAlmostEqual(obs.log_probability(sim, m['A0'].value + 2), norm.logpdf(m['A0'].value + 2, m['A0'].value, 2))

        self.assertEqual(obs.sample(sim).size, 1)

        obs = LinearWeightedSumOfSquaresObservation([0, 0, 0], ['A', 'B', 'C'], unc)
        ics = [m['A0'].value, m['B0'].value, 0]
        self.assertAlmostEqual(obs.probability(sim, ics), multivariate_normal.pdf([0, 0, 0], [0, 0, 0], 2 ** 2))
        self.assertAlmostEqual(obs.log_probability(sim, ics), multivariate_normal.logpdf([0, 0, 0], [0, 0, 0], 2 ** 2))
        self.assertAlmostEqual(obs.log_probability(sim, ics), multivariate_normal.logpdf([0, 0, 0], [0, 0, 0], np.asarray([2, 2, 2])**2))

        self.assertEqual(obs.sample(sim).size, 3)

    def test_log_probability_dk(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        unc = AffineMeasurementUncertainty(2, 0)

        obs = LinearWeightedSumOfSquaresObservation([0], ['A'], unc)
        assert_log_probability_dk(m, con, obs, m['A0'].value, atol=1e-6)
        assert_log_probability_dk(m, con, obs, m['A0'].value + 2)

        obs = LinearWeightedSumOfSquaresObservation([0, 0, 0], ['A', 'B', 'C'], unc)
        assert_log_probability_dk(m, con, obs, [m['A0'].value + 0.1, m['B0'].value + 0.2, 0.3])
