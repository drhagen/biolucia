import unittest
from numpy.testing import assert_allclose

from biolucia.model import *
from biolucia.experiment import InitialValueExperiment

from tests.test_models import equilibrium_model, equilibrium_dose_variant, dose_step


class SimulateTestCase(unittest.TestCase):
    def test_simulate_system(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = m.simulate(con)

        self.assertEqual(sim.vector_values(0, 'A'), m['A0'].value)
        assert_allclose(sim.vector_values(0, ['A', 'A0']), [m['A0'].value, m['A0'].value])

        self.assertEqual(sim.matrix_values(0, 'kf'), 0.5)
        self.assertEqual(sim.matrix_values(0, 'A'), 10)
        self.assertLess(sim.matrix_values(1, 'A'), 10)

        assert_allclose(sim.matrix_values(0, ['A', 'B']), array([10, 5]))
        assert_allclose(sim.matrix_values([0, 1], 'kf'), array([0.5, 0.5]))
        assert_allclose(sim.matrix_values([0, 1, 2], ['kr', 'kf']), array([[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]]))

    def test_simulate_system2(self):
        m = equilibrium_model()
        con = InitialValueExperiment(equilibrium_dose_variant())

        sim = m.simulate(con)

        self.assertEqual(sim.matrix_values(0, 'kf'), 0.5)
        self.assertEqual(sim.matrix_values(0, 'A'), 10)
        self.assertLess(sim.matrix_values(1, 'A'), 10)

    def test_simulate_dose(self):
        m = dose_step()

        sim = m.simulate()

        self.assertEqual(sim.matrix_values(0, 'A'), 0.0)
        self.assertEqual(sim.matrix_values(1, 'A'), 2.0)
        self.assertEqual(sim.matrix_values(1.5, 'A'), 2.0)
        self.assertEqual(sim.matrix_values(2, 'A'), 3.0)

    def test_events(self):
        m = Model()
        m = m.add("A* = 0", "A' = 1", "@(A > 1) A = 0")

        sim = m.simulate()

        self.assertTrue(0.9 < sim.matrix_values(0.95, 'A') < 1)
        self.assertTrue(0 < sim.matrix_values(1.05, 'A') < 0.1)
        self.assertTrue(0.9 < sim.matrix_values(1.95, 'A') < 1)
        self.assertTrue(0 < sim.matrix_values(2.05, 'A') < 0.1)

    def test_empty(self):
        m = Model()

        sim = m.simulate()

        self.assertEqual(sim.matrix_values(10).shape, (0,))
        self.assertEqual(sim.matrix_values([10, 20]).shape, (2, 0))
