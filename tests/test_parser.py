import unittest

from biolucia.model import *
from biolucia.parser import *


class ParserTestCase(unittest.TestCase):
    def test_parse_components(self):
        a = parse_all(constant, 'a = 10')
        self.assertEqual(a, (Symbol('a'), Constant('a', 10)))

        b = parse_all(rule, 'b = x + y')
        self.assertEqual(b, (Symbol('b'), Rule('b', 'x + y')))

        c = parse_all(initial, 'c* = sin(x)')
        self.assertEqual(c, (Symbol('c'), Initial('sin(x)')))

        d = parse_all(dose, 'd(1.2) = dose ^ 2')
        self.assertEqual(d, (Symbol('d'), Dose(1.2, 'dose ** 2')))

        e = parse_all(ode, "e' = (s) + x * r")
        self.assertEqual(e, (Symbol('e'), Ode('s + r*x')))
        ee = parse_all(ode, "e' = ((s) + (x * r))")
        self.assertEqual(e, ee)

        f = parse_all(event, '@(A < 2) A += 10')
        self.assertEqual(f, Event('A - 2', EventDirection.down, True, [Effect('A', 'A + 10')]))

        f = parse_all(event, '@(A > B) A = B - 2')
        self.assertEqual(f, Event('A - B', EventDirection.up, True, [Effect('A', 'B - 2')]))

        g = parse_all(component, '@(A > B) A = B - 2')
        self.assertEqual(g, f)

        h = parse_all(ode, "e' = 0")
        self.assertEqual(h, (Symbol('e'), Ode('0')))

        i = parse_all(initial, "e* = 1.2")
        self.assertEqual(i, (Symbol('e'), Initial(1.2)))

        j = parse_all(rule, "j(t<10) = z*2")
        self.assertEqual(j, (Symbol('j'), Rule('j', AnalyticSegment(-inf, 10, 'z*2'))))

        k = parse_all(rule, "k(1 < t < 10) = z*2")
        self.assertEqual(k, (Symbol('k'), Rule('k', AnalyticSegment(1, 10, 'z*2'))))

        l = parse_all(ode, "l(1 < t < 10)' = cos(t)")
        self.assertEqual(l, (Symbol('l'), Ode(AnalyticSegment(1, 10, 'cos(t)'))))
