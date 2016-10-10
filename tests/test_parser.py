import unittest
import os

from biolucia.model import *
from biolucia.parser import *


class ParserTestCase(unittest.TestCase):
    def test_parse_components(self):
        a = ModelParsers.constant.parse('a = 10').or_die()
        self.assertEqual(a, (Symbol('a'), Constant('a', 10)))

        b = ModelParsers.rule.parse('b = x + y').or_die()
        self.assertEqual(b, (Symbol('b'), Rule('b', 'x + y')))

        c = ModelParsers.initial.parse('c* = sin(x)').or_die()
        self.assertEqual(c, (Symbol('c'), Initial('sin(x)')))

        c = ModelParsers.initial.parse('c* = atan2(x, y)').or_die()
        self.assertEqual(c, (Symbol('c'), Initial('atan2(x, y)')))

        d = ModelParsers.dose.parse('d(1.2) = dose ^ 2').or_die()
        self.assertEqual(d, (Symbol('d'), Dose(1.2, 'dose ** 2')))

        e = ModelParsers.ode.parse("e' = (s) + x * r").or_die()
        self.assertEqual(e, (Symbol('e'), Ode('s + r*x')))

        ee = ModelParsers.ode.parse("e' = ((s) + (x * r))").or_die()
        self.assertEqual(e, ee)

        f = ModelParsers.event.parse('@(A < 2) A += 10').or_die()
        self.assertEqual(f, Event('A - 2', EventDirection.down, True, [Effect('A', 'A + 10')]))

        f = ModelParsers.event.parse('@(A < 2) A += 10, B += 2').or_die()
        self.assertEqual(f, Event('A - 2', EventDirection.down, True, [Effect('A', 'A + 10'), Effect('B', 'B + 2')]))

        f = ModelParsers.event.parse('@(A > B) A = B - 2').or_die()
        self.assertEqual(f, Event('A - B', EventDirection.up, True, [Effect('A', 'B - 2')]))

        g = ModelParsers.component.parse('@(A > B) A = B - 2').or_die()
        self.assertEqual(g, f)

        h = ModelParsers.ode.parse("e' = 0").or_die()
        self.assertEqual(h, (Symbol('e'), Ode('0')))

        i = ModelParsers.initial.parse("e* = 1.2").or_die()
        self.assertEqual(i, (Symbol('e'), Initial(1.2)))

        j = ModelParsers.rule.parse("j(t<10) = z*2").or_die()
        self.assertEqual(j, (Symbol('j'), Rule('j', AnalyticSegment(-inf, 10, 'z*2'))))

        k = ModelParsers.rule.parse("k(1 < t < 10) = z*2").or_die()
        self.assertEqual(k, (Symbol('k'), Rule('k', AnalyticSegment(1, 10, 'z*2'))))

        l = ModelParsers.ode.parse("l(1 < t < 10)' = cos(t)").or_die()
        self.assertEqual(l, (Symbol('l'), Ode(AnalyticSegment(1, 10, 'cos(t)'))))

    def test_combined_components(self):
        a = ModelParsers.model.parse('%components').or_die()
        self.assertEqual(len(a.parts), 0)
        self.assertEqual(len(a.events), 0)

        a = ModelParsers.model.parse('').or_die()
        self.assertEqual(len(a.parts), 0)
        self.assertEqual(len(a.events), 0)

        a = ModelParsers.model.parse("%components\nA' = -k\nA* = 10\nk = 0.2").or_die()
        self.assertEqual(len(a.parts), 2)
        self.assertEqual(len(a.events), 0)

        a = ModelParsers.model.parse("%components\nA' = -k\nA* = 10\nk = 0.2\n@(A < 2) A += 1").or_die()
        self.assertEqual(len(a.parts), 2)
        self.assertEqual(len(a.events), 1)

    def test_file_read(self):
        models_dir = '../models/'
        for file in os.listdir(models_dir):
            if file.endswith('.txt'):
                m = read_model(models_dir + file)
                self.assertIsInstance(m, Model)
