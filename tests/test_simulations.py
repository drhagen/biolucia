import unittest
from numpy.testing import assert_allclose

from biolucia.model import *
from biolucia.experiment import InitialValueExperiment
from biolucia.simulation import simulate

from tests.test_models import equilibrium_model, equilibrium_dose_variant, dose_step


class SimulateSystemTestCase(unittest.TestCase):
    def test_simulate_system(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = simulate(m, con)

        self.assertEqual(sim.vector_values(0, 'A'), m['A0'].value)
        assert_allclose(sim.vector_values(0, ['A', 'A0']), [m['A0'].value, m['A0'].value])

        self.assertEqual(sim.matrix_values(0, 'kf'), 0.5)
        self.assertEqual(sim.matrix_values(0, 'A'), 10)
        self.assertLess(sim.matrix_values(1, 'A'), 10)

        assert_allclose(sim.matrix_values(0, ['A', 'B']), np.asarray([10, 5]))
        assert_allclose(sim.matrix_values([0, 1], 'kf'), np.asarray([0.5, 0.5]))
        assert_allclose(sim.matrix_values([0, 1, 2], ['kr', 'kf']), np.asarray([[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]]))

    def test_simulate_single(self):
        m = equilibrium_model()

        sim = simulate(m)

        self.assertEqual(sim.vector_values(0, 'A'), m['A0'].value)
        assert_allclose(sim.vector_values(0, ['A', 'A0']), [m['A0'].value, m['A0'].value])

        self.assertEqual(sim.matrix_values(0, 'kf'), 0.5)
        self.assertEqual(sim.matrix_values(0, 'A'), 10)
        self.assertLess(sim.matrix_values(1, 'A'), 10)

        assert_allclose(sim.matrix_values(0, ['A', 'B']), np.asarray([10, 5]))
        assert_allclose(sim.matrix_values([0, 1], 'kf'), np.asarray([0.5, 0.5]))
        assert_allclose(sim.matrix_values([0, 1, 2], ['kr', 'kf']), np.asarray([[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]]))

    def test_simulate_system2(self):
        m = equilibrium_model()
        con = InitialValueExperiment(equilibrium_dose_variant())

        sim = simulate(m, con)

        self.assertEqual(sim.matrix_values(0, 'kf'), 0.5)
        self.assertEqual(sim.matrix_values(0, 'A'), 10)
        self.assertLess(sim.matrix_values(1, 'A'), 10)

    def test_simulate_dose(self):
        m = dose_step()
        con = InitialValueExperiment()

        sim = simulate(m, con)

        self.assertEqual(sim.matrix_values(0, 'A'), 0.0)
        self.assertEqual(sim.matrix_values(1, 'A'), 2.0)
        self.assertEqual(sim.matrix_values(1.5, 'A'), 2.0)
        self.assertEqual(sim.matrix_values(2, 'A'), 3.0)

        self.assertEqual(sim.matrix_values(0, 'B'), 0.0)
        self.assertAlmostEqual(sim.matrix_values(0.5, 'B'), 2.0)
        self.assertAlmostEqual(sim.matrix_values(1, 'B'), 3.0)
        self.assertAlmostEqual(sim.matrix_values(2, 'B'), 7.0)
        self.assertAlmostEqual(sim.matrix_values(3, 'B'), 16.0)

    def test_bouncing(self):
        m = Model()
        m = m.add('x* = 50', 'v* = 0', "x' = v", "v' = -1", '@(x < 0) v = -v')
        sim = simulate(m)

        self.assertGreater(sim.matrix_values(100, 'x'), 0)

    def test_events(self):
        m = Model()
        m = m.add("A* = 0", "A' = 1", "@(A > 1) A = 0")
        con = InitialValueExperiment()

        sim = simulate(m, con)

        self.assertTrue(0.9 < sim.matrix_values(0.95, 'A') < 1)
        self.assertTrue(0 < sim.matrix_values(1.05, 'A') < 0.1)
        self.assertTrue(0.9 < sim.matrix_values(1.95, 'A') < 1)
        self.assertTrue(0 < sim.matrix_values(2.05, 'A') < 0.1)

    def test_empty(self):
        m = Model()
        con = InitialValueExperiment()

        sim = simulate(m, con)

        self.assertEqual(sim.matrix_values(10).shape, (0,))
        self.assertEqual(sim.matrix_values([10, 20]).shape, (2, 0))

    def test_update_events(self):
        m = equilibrium_model()
        con = InitialValueExperiment(Model().add('@(B < 2) B = 0'))

        sim = simulate(m, con)

        self.assertTrue(sim.matrix_values(0.7, 'B') < 1)

    def test_dose_right_after_discontinuity(self):
        m = Model()
        m = m.add('A* = 10', "A'(t < 5) = -1 * A", "A'(t > 5) = 0", 'A(5.1) += 10')

        sim = simulate(m)

        self.assertGreater(sim.matrix_values(5.2, 'A'), 5)

    def test_dose_right_after_stopping(self):
        m = Model()
        m = m.add('A* = 10', "A'(t < 5) = -1 * A", "A'(t > 5) = 0", 'A(5.1) += 10')

        sim = simulate(m)

        self.assertLess(sim.matrix_values(5.05, 'A'), 5)
        self.assertGreater(sim.matrix_values(5.2, 'A'), 5)

    def test_discontinuity_right_after_stopping(self):
        m = Model()
        m = m.add('A* = 10', "A'(t < 5) = -1 * A", "A'(t > 5) = 1")

        sim = simulate(m)

        self.assertLess(sim.matrix_values(4.95, 'A'), sim.matrix_values(5.1, 'A'))

    def test_discontinuity_at_stopping(self):
        m = Model()
        m = m.add('A* = 10', "A'(t < 5) = -1 * A", "A'(t > 5) = 1")

        sim = simulate(m)

        self.assertLess(sim.matrix_values(4.95, 'A'), sim.matrix_values(5.1, 'A'))


class SimulateSensitivitiesTestCase(unittest.TestCase):
    def test_simulate_sensitivity(self):
        m = equilibrium_model()
        con = InitialValueExperiment()

        sim = simulate(m, con, parameters=['A0', 'kr'])

        assert_allclose(sim.matrix_sensitivities(0.0, 'A'), [1.0, 0.0])
        self.assertEqual(sim.matrix_sensitivities(0.0, 'A').shape, (2,))
        assert_allclose(sim.matrix_sensitivities([0.0, 10.0], 'A0'), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(sim.matrix_sensitivities([0.0, 10.0], 'A0').shape, (2, 2))
        assert_allclose(sim.matrix_sensitivities(0.0, ['A', 'A0']), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(sim.matrix_sensitivities(0.0, ['A', 'A0']).shape, (2, 2))
        assert_allclose(sim.matrix_sensitivities([0.0, 10.0], ['A0', 'B0']), [[[1.0, 0.0], [0.0, 0.0]],
                                                                              [[1.0, 0.0], [0.0, 0.0]]])
        self.assertEqual(sim.matrix_sensitivities([0.0, 10.0], ['A', 'A0']).shape, (2, 2, 2))

    def test_simulate_sensitivity_dose(self):
        m = dose_step()

        sim = simulate(m, parameters=['k0', 'k1', 'k2'])

        assert_allclose(sim.matrix_sensitivities(0.0, 'A'), [0.0, 0.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(2.0, 'A'), [0.0, 0.0, 0.0])

        assert_allclose(sim.matrix_sensitivities(0.0, 'B'), [0.0, 0.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(0.5, 'B'), [0.5, 0.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(1.0, 'B'), [0.0, 1.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(2.0, 'B'), [1.0, 1.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(3.0, 'B'), [2.0, 1.0, 1.0])

    def test_dose_sensitivity(self):
        m = Model().add('k1 = 4', 'k2 = 2', 'a* = 2 * k1', 'a(2) = k2', 'a(4) += k1')

        sim = simulate(m, parameters=['k1', 'k2'])

        assert_allclose(sim.matrix_sensitivities(0.0, 'a'), [2.0, 0.0])
        assert_allclose(sim.matrix_sensitivities(2.0, 'a'), [0.0, 1.0])
        assert_allclose(sim.matrix_sensitivities(4.0, 'a'), [1.0, 1.0])

    def test_bounce_event_sensitivity(self):
        m = Model().add('f = 1.0', 'p* = 50', 'v* = 0', "p' = v", "v' = -1", '@(p < 0) v = -f*v')

        sim = simulate(m, parameters=['f'])

        # Return velocity is proportional to f: dv/df = v. Kinetic energy is
        assert_allclose(sim.matrix_values(10, 'p'), [0], atol=1e-8)
        assert_allclose(sim.matrix_values(9.99999999, 'v'), [-10])
        assert_allclose(sim.matrix_values(10.0000001, 'v'), [10])
        assert_allclose(sim.matrix_sensitivities(10, 'p'), [0], atol=1e-8)
        assert_allclose(sim.matrix_sensitivities(9.99999999, 'v'), [0])
        assert_allclose(sim.matrix_sensitivities(10.0000001, 'v'), [10])

        assert_allclose(sim.matrix_values(20, 'p'), [50])
        assert_allclose(sim.matrix_values(20, 'v'), [0], atol=1e-8)
        assert_allclose(sim.matrix_sensitivities(20, 'v'), [10])

        assert_allclose(sim.matrix_values(30, 'p'), [0], atol=1e-8)
        assert_allclose(sim.matrix_values(29.9999999, 'v'), -10)
        assert_allclose(sim.matrix_values(30.0000001, 'v'), 10)
        assert_allclose(sim.matrix_sensitivities(10, 'p'), [0], atol=1e-8)
        assert_allclose(sim.matrix_sensitivities(29.9999999, 'v'), [10])

    def test_event_time_sensitivity(self):
        m = Model().add('k = 1', 'x* = 0', "x' = 0", 'y* = 0', "y' = x", '@(t > k) x = 1')

        sim = simulate(m, parameters=['k'])

        assert_allclose(sim.matrix_values([0.0, 1.0, 2.0], ['x', 'y']), [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        assert_allclose(sim.matrix_sensitivities(0.0, ['x', 'y']), [[0.0], [0.0]])
        assert_allclose(sim.matrix_sensitivities(1.0, ['x', 'y']), [[0.0], [-1.0]])
        assert_allclose(sim.matrix_sensitivities(2.0, ['x', 'y']), [[0.0], [-1.0]])
