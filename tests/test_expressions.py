import unittest

from biolucia.helpers import *
from biolucia.model import *


class PiecewiseTestCase(unittest.TestCase):
    def test_substitution(self):
        expr = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('x + y')), AnalyticSegment(20, 30, ex('x + 5'))])

        whole = PiecewiseAnalytic([AnalyticSegment(-inf, inf, ex('z'))])
        expr_whole = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('z + y')), AnalyticSegment(20, 30, ex('z + 5'))])
        self.assertEqual(expr.subs([('x', whole)]), expr_whole)

        front = PiecewiseAnalytic([AnalyticSegment(0, 15, ex('z'))])
        expr_front = PiecewiseAnalytic([AnalyticSegment(10, 15, ex('z + y'))])
        self.assertEqual(expr.subs([('x', front)]), expr_front)

        back = PiecewiseAnalytic([AnalyticSegment(15, 25, ex('z'))])
        expr_back = PiecewiseAnalytic([AnalyticSegment(15, 20, ex('z + y')), AnalyticSegment(20, 25, ex('z + 5'))])
        self.assertEqual(expr.subs([('x', back)]), expr_back)

        middle = PiecewiseAnalytic([AnalyticSegment(12, 18, ex('z'))])
        expr_middle = PiecewiseAnalytic([AnalyticSegment(12, 18, ex('z + y'))])
        self.assertEqual(expr.subs([('x', middle)]), expr_middle)

        disjoint = PiecewiseAnalytic([AnalyticSegment(30, 50, ex('z'))])
        expr_disjoint = PiecewiseAnalytic([])
        self.assertEqual(expr.subs([('x', disjoint)]), expr_disjoint)

    def test_addition(self):
        expr = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('x + y')), AnalyticSegment(20, 30, ex('x + 5'))])

        whole = PiecewiseAnalytic([AnalyticSegment(-inf, inf, ex('z'))])
        expr_whole = PiecewiseAnalytic([AnalyticSegment(-inf, 10, ex('z')), AnalyticSegment(10, 20, ex('x + z + y')),
                                        AnalyticSegment(20, 30, ex('z + x + 5')), AnalyticSegment(30, inf, ex('z'))])
        self.assertEqual(expr + whole, expr_whole)
        self.assertEqual(whole + expr, expr_whole)

        front = PiecewiseAnalytic([AnalyticSegment(0, 15, ex('z'))])
        expr_front = PiecewiseAnalytic([AnalyticSegment(0, 10, ex('z')), AnalyticSegment(10, 15, ex('x + y + z')),
                                        AnalyticSegment(15, 20, ex('x + y')), AnalyticSegment(20, 30, ex('x + 5'))])
        self.assertEqual(expr + front, expr_front)
        self.assertEqual(front + expr, expr_front)

    def test_constant_substitution(self):
        expr = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('x + y')), AnalyticSegment(20, 30, ex('x + 5'))])
        expr_constant = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('5')), AnalyticSegment(20, 30, ex('8'))])
        self.assertEqual(expr.subs([('x', 3), ('y', 2)]), expr_constant)

    def test_to_function(self):
        expr = PiecewiseAnalytic([AnalyticSegment(10, 20, ex('x + y')), AnalyticSegment(20, 30, ex('x + 5'))])
        func = expr.to_function(['t', 'x', 'y'])

        self.assertEqual(func(10, 5, 6), 11)
        self.assertEqual(func(20, 4, 3), 9)
        self.assertTrue(isnan(func(40, 2, 2)))
