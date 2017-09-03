from biolucia.helpers import *
from biolucia.model import *


def equilibrium_model():
    m = Model()
    m = m.add(Constant('A0', 10))
    m = m.add(Constant('B0', 5))
    m = m.add(State('A', Initial(ex('A0')), [], Ode(PiecewiseAnalytic([AnalyticSegment(0, inf, ex('kr*C-kf*A*B'))]))))
    m = m.add(State('B', Initial(ex('B0')), [], Ode(PiecewiseAnalytic([AnalyticSegment(0, inf, ex('kr*C-kf*A*B'))]))))
    m = m.add(State('C', Initial(ex('0')), [], Ode(PiecewiseAnalytic([AnalyticSegment(0, inf, ex('kf*A*B-kr*C'))]))))
    m = m.add(Constant('kf', 0.5))
    m = m.add(Constant('kr', 0.2))

    return m


def equilibrium_dose_variant():
    m = Model()
    m = m.add(State('A', None, [Dose(2, ex('1')), Dose(4, ex('1'))], None))

    return m


def dose_step():
    m = Model()
    m = m.add('A* = 0', "A' = 0", 'A(1) = 2', 'A(2) += 1')
    m = m.add('k0 = 4', 'k1 = 3', 'k2 = 5', 'B* = 0', "B' = k0", 'B(1) = k1', 'B(3) += k2')

    return m


def bouncing():
    m = Model()
    m = m.add('x* = 50', 'v* = 0', "x' = v", "v' = -1", '@(x < 0) v = -v')

    return m
