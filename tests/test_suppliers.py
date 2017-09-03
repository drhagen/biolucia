import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from biolucia.ode import ListDoseSupplier, ListDiscontinuitySupplier, LazyIntegrableSolutionDoseSupplier
from biolucia.simulation import simulate

from tests.test_models import bouncing, dose_step


class DoseSupplierTestCase(unittest.TestCase):
    def test_list_dose_supplier(self):
        a = ListDoseSupplier([0.0, 1.0, 2.0])
        assert_array_equal(a.supply(0.0), [0.0])
        assert_array_equal(a.supply(0.0, 0.0), [])
        assert_array_equal(a.supply(0.0, 2.0), [1.0, 2.0])
        assert_array_equal(a.supply(1.0, 5.0), [2.0])
        assert_array_equal(a.supply(0.0, 1.5), [1.0])

        a = ListDoseSupplier([])
        assert_array_equal(a.supply(0.0, 0.0), [])
        assert_array_equal(a.supply(0.0, 2.0), [])


class DiscontinuitySupplierTestCase(unittest.TestCase):
    def test_list_discontinuity_supplier(self):
        a = ListDiscontinuitySupplier([0.0, 1.0, 2.0])
        assert_array_equal(a.supply(0.0), [1.0])
        assert_array_equal(a.supply(0.0, 0.0), [1.0])
        assert_array_equal(a.supply(0.0, 2.0), [1.0, 2.0])
        assert_array_equal(a.supply(1.0, 5.0), [2.0, np.inf])
        assert_array_equal(a.supply(0.0, 1.5), [1.0, 2.0])
        assert_array_equal(a.supply(1.0, 1.0), [2.0])
        assert_array_equal(a.supply(1.5, 1.5), [2.0])
        assert_array_equal(a.supply(2.0, 2.0), [np.inf])

        a = ListDiscontinuitySupplier([])
        assert_array_equal(a.supply(0.0, 0.0), [np.inf])
        assert_array_equal(a.supply(0.0, 2.0), [np.inf])


class LazyIntegrableSolutionDoseSupplierTestCase(unittest.TestCase):
    def test_lazy_integrable_solution_dose_supplier_dose(self):
        m = dose_step()
        sim = simulate(m)

        a = LazyIntegrableSolutionDoseSupplier(sim.solution)
        assert_array_almost_equal(a.supply(0.0, 10), [1.0, 2.0, 3.0])

    def test_lazy_integrable_solution_dose_supplier_bounce(self):
        m = bouncing()
        sim = simulate(m)

        a = LazyIntegrableSolutionDoseSupplier(sim.solution)
        assert_array_almost_equal(a.supply(15, 75), [30, 50, 70])

    def test_lazy_integrable_solution_dose_supplier_combined(self):
        m = bouncing().add('x(40) = 50')
        sim = simulate(m)

        a = LazyIntegrableSolutionDoseSupplier(sim.solution)
        assert_array_almost_equal(a.supply(15, 75), [30, 40, 50, 70])
