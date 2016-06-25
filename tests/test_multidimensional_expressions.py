import unittest

from biolucia.helpers import *
from biolucia.analytic import multidimensional_derivative, multidimensional_lambdify


class MultidimensionalDerivative(unittest.TestCase):
    def test_one_dimensional_derivative(self):
        exprs = [ex('x**2 + y'), ex('x*y + 5')]

        der1 = multidimensional_derivative(exprs, ['x', 'y'])
        der2 = multidimensional_derivative(der1, ['x', 'y'])

        self.assertTrue(np.array_equal(der1, np.array([[ex('2*x'), 1], [ex('y'), ex('x')]])))
        self.assertTrue(np.array_equal(der2,
                                       np.array([[[2, 0], [0, 0]], [[0, 1], [1, 0]]])))

    def test_multidimensional_derivatives(self):
        exprs = [ex('x**2 + y'), ex('x*y + 5')]

        der1 = multidimensional_derivative(exprs, ['x', 'y'])
        der2 = multidimensional_derivative(der1, ['x', 'y'])

        func1 = multidimensional_lambdify(['x', 'y'], der1)
        func2 = multidimensional_lambdify(['x', 'y'], der2)

        val1 = func1(3, 4)
        val2 = func2(3, 4)

        self.assertTrue(np.array_equal(val1, np.array([[6, 1], [4, 3]])))
        self.assertTrue(np.array_equal(val2,
                                       np.array([[[2, 0], [0, 0]], [[0, 1], [1, 0]]])))
