import unittest

import numpy as np
from scipy.stats import norm, multivariate_normal

from biolucia.model import *
from biolucia.experiment import InitialValueExperiment
from biolucia.observation import LinearWeightedSumOfSquaresObservation

from tests.test_models import equilibrium_model


class ObservationTestCase(unittest.TestCase):
    def test_log_probability(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = m.simulate(con)

        obs = LinearWeightedSumOfSquaresObservation([0], ['A'], lambda t, y_name, ybar: np.ones(ybar.shape)*2)
        self.assertAlmostEqual(obs.probability(sim, m['A0'].value), norm.pdf(m['A0'].value, m['A0'].value, 2))
        self.assertAlmostEqual(obs.log_probability(sim, m['A0'].value), norm.logpdf(m['A0'].value, m['A0'].value, 2))

        obs = LinearWeightedSumOfSquaresObservation([0], ['A'], lambda t, y_name, ybar: np.ones(ybar.shape)*2)
        self.assertAlmostEqual(obs.probability(sim, m['A0'].value + 2), norm.pdf(m['A0'].value + 2, m['A0'].value, 2))
        self.assertAlmostEqual(obs.log_probability(sim, m['A0'].value + 2), norm.logpdf(m['A0'].value + 2, m['A0'].value, 2))

        self.assertEqual(obs.sample(sim).size, 1)

        obs = LinearWeightedSumOfSquaresObservation([0, 0, 0], ['A', 'B', 'C'],
                                                    lambda t, y_name, ybar: np.ones(ybar.shape) * 2)
        ics = [m['A0'].value, m['B0'].value, 0]
        self.assertAlmostEqual(obs.probability(sim, ics), multivariate_normal.pdf([0, 0, 0], [0, 0, 0], 2 ** 2))
        self.assertAlmostEqual(obs.log_probability(sim, ics), multivariate_normal.logpdf([0, 0, 0], [0, 0, 0], 2 ** 2))
        self.assertAlmostEqual(obs.log_probability(sim, ics), multivariate_normal.logpdf([0, 0, 0], [0, 0, 0], np.asarray([2, 2, 2])**2))

        self.assertEqual(obs.sample(sim).size, 3)
