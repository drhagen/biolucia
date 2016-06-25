import unittest

from numpy import array, array_equal
from biolucia.model import *
from biolucia.experiment import InitialValueExperiment

from tests.test_models import equilibrium_model, equilibrium_dose_variant, dose_step


class Simulate(unittest.TestCase):
    def test_simulate_system(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = m.simulate(con)

        self.assertEqual(sim.system_values(0, 'kf'), 0.5)
        self.assertEqual(sim.system_values(0, 'A'), 10)
        self.assertLess(sim.system_values(1, 'A'), 10)

        self.assertTrue(array_equal(sim.system_values(0, ['A', 'B']), array([10, 5])))
        self.assertTrue(array_equal(sim.system_values([0, 1], 'kf'), array([0.5, 0.5])))
        self.assertTrue(array_equal(sim.system_values([0, 1, 2], ['kr', 'kf']),
                                    array([[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]])))

    def test_simulate_system2(self):
        m = equilibrium_model()
        con = InitialValueExperiment(equilibrium_dose_variant())

        sim = m.simulate(con)

        self.assertEqual(sim.system_values(0, 'kf'), 0.5)
        self.assertEqual(sim.system_values(0, 'A'), 10)
        self.assertLess(sim.system_values(1, 'A'), 10)

    def test_simulate_dose(self):
        m = dose_step()
        con = InitialValueExperiment()

        sim = m.simulate()

        self.assertEqual(sim.system_values(0, 'A'), 0.0)
        self.assertEqual(sim.system_values(1, 'A'), 2.0)
        self.assertEqual(sim.system_values(1.5, 'A'), 2.0)
        self.assertEqual(sim.system_values(2, 'A'), 2.0)