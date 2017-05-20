from typing import Sequence, Union, Tuple, List, Callable
from numbers import Real
from collections import OrderedDict
from sympy import Expr, lambdify
import copy

import numpy as np
from numpy import inf, array
from scipy.integrate import Radau
from scipy.integrate._ivp.ivp import find_active_events, solve_event_equation

from .analytic import multidimensional_derivative, multidimensional_lambdify


# TODO: Split this into an abstract base class IntegrableSystem with only the methods. Then create
# SymbolicIntegrableSystem that converts these arguments into the _func fields. This will make it possible to merge
# LazyIntegrableSolution in IntegrableSystem and not lose the non-symbolic capabilities. This will also make it possible
# to merge the general sensitivities code from BioluciaSystemSimulation. Maybe merge this all into the simulation...
class IntegrableSystem:
    def __init__(self, parameters: 'OrderedDict[str, float]', states: 'OrderedDict[str, Tuple[Expr, Expr]]',
                 discontinuities: List[float], doses: 'OrderedDict[float, List[Expr]]',
                 events: List[Tuple[Expr, int, List[Expr]]], outputs: 'OrderedDict[str, Expr]'):
        self.parameters = parameters
        self.states = states
        self.discontinuities = discontinuities
        self.doses = doses
        self.events = events
        self.outputs = outputs

        self.k_names = [*parameters.keys()]
        self.x_names = [*states.keys()]
        self.y_names = [*outputs.keys()]

        all_parameters = ['t'] + self.x_names + self.k_names

        self.k = np.asarray([*parameters.values()], dtype=float)

        self.x0_syms = [value[0] for value in states.values()]
        self.x0_func = lambdify(self.k_names, self.x0_syms)
        self.x0 = self.x0_func(*self.k)

        self.x0_dk_syms = multidimensional_derivative(self.x0_syms, self.k_names)
        self.x0_dk_func = multidimensional_lambdify(self.k_names, self.x0_dk_syms)
        self.x0_dk = self.x0_dk_func(*self.k)

        self.f_syms = [value[1] for value in states.values()]
        self.f_func = lambdify(all_parameters, self.f_syms)

        # TODO: make all derivatives lazy properties (they can still be copied safely)
        # TODO: make all these appropriately sparse
        self.f_dx_syms = multidimensional_derivative(self.f_syms, self.x_names)
        self.f_dx_func = multidimensional_lambdify(all_parameters, self.f_dx_syms)

        self.f_dk_syms = multidimensional_derivative(self.f_syms, self.k_names)
        self.f_dk_func = multidimensional_lambdify(all_parameters, self.f_dk_syms)

        self.f_dx_dx_syms = multidimensional_derivative(self.f_dx_syms, self.x_names)
        self.f_dx_dx_func = multidimensional_lambdify(all_parameters, self.f_dx_dx_syms)

        self.f_dx_dk_syms = multidimensional_derivative(self.f_dx_syms, self.k_names)
        self.f_dx_dk_func = multidimensional_lambdify(all_parameters, self.f_dx_dk_syms)

        self.f_dk_dx_syms = multidimensional_derivative(self.f_dk_syms, self.x_names)
        self.f_dk_dx_func = multidimensional_lambdify(all_parameters, self.f_dk_dx_syms)

        self.f_dk_dk_syms = multidimensional_derivative(self.f_dk_syms, self.k_names)
        self.f_dk_dk_func = multidimensional_lambdify(all_parameters, self.f_dk_dk_syms)

        self.t_dose = list(doses.keys())
        self.d_func = OrderedDict((item[0], lambdify(all_parameters, item[1])) for item in self.doses.items())

        self.d_dx_syms = OrderedDict((item[0], multidimensional_derivative(item[1], self.x_names))
                                     for item in self.doses.items())
        self.d_dx_func = OrderedDict((item[0], multidimensional_lambdify(all_parameters, item[1]))
                                     for item in self.d_dx_syms.items())

        self.d_dk_syms = OrderedDict((item[0], multidimensional_derivative(item[1], self.k_names))
                                     for item in self.doses.items())
        self.d_dk_func = OrderedDict((item[0], multidimensional_lambdify(all_parameters, item[1]))
                                     for item in self.d_dk_syms.items())

        self.e_syms = [value[0] for value in events]
        self.e_func = [lambdify(all_parameters, value) for value in self.e_syms]
        self.e_dir = np.asarray([value[1] for value in events])
        self.e_eff_func = [lambdify(all_parameters, value[2]) for value in events]

        self.y_func = OrderedDict((item[0], lambdify(all_parameters, item[1])) for item in outputs.items())

        self.y_dx_syms = OrderedDict((item[0], multidimensional_derivative(item[1], self.x_names))
                                     for item in self.outputs.items())
        self.y_dx_func = OrderedDict((item[0], multidimensional_lambdify(all_parameters, item[1]))
                                     for item in self.y_dx_syms.items())

        self.y_dk_syms = OrderedDict((item[0], multidimensional_derivative(item[1], self.k_names))
                                     for item in self.outputs.items())
        self.y_dk_func = OrderedDict((item[0], multidimensional_lambdify(all_parameters, item[1]))
                                     for item in self.y_dk_syms.items())

    def f(self, t, x):
        return self.f_func(t, *x, *self.k)

    def f_dx(self, t, x):
        return self.f_dx_func(t, *x, *self.k)

    def f_dk(self, t, x):
        return self.f_dk_func(t, *x, *self.k)

    def f_dx_dx(self, t, x):
        return self.f_dx_dx_func(t, *x, *self.k)

    def f_dx_dk(self, t, x):
        return self.f_dx_dk_func(t, *x, *self.k)

    def f_dk_dx(self, t, x):
        return self.f_dk_dx_func(t, *x, *self.k)

    def f_dk_dk(self, t, x):
        return self.f_dk_dk_func(t, *x, *self.k)

    def d(self, t, x):
        return self.d_func[t](t, *x, *self.k)

    def d_dx(self, t, x):
        return self.d_dx_func[t](t, *x, *self.k)

    def d_dk(self, t, x):
        return self.d_dk_func[t](t, *x, *self.k)

    @property
    def e(self):
        return EventGetter(self.e_func, self.k)

    @property
    def e_eff(self):
        return EventGetter(self.e_eff_func, self.k)

    def y(self, name, t, x):
        return self.y_func[name](t, *x, *self.k)

    def y_dx(self, name, t, x):
        return self.y_dx_func[name](t, *x, *self.k)

    def y_dk(self, name, t, x):
        return self.y_dk_func[name](t, *x, *self.k)

    def update(self, parameters):
        new_self = copy.copy(self)
        new_self.parameters = OrderedDict(zip(self.parameters.keys(), parameters))
        new_self.k = np.asarray(parameters)
        # TODO: consider making x0 a function or property because it usually only called once and it simplifies update
        new_self.x0 = new_self.x0_func(*new_self.k)
        new_self.x0_dk = new_self.x0_dk_func(*new_self.k)
        return new_self


class EventGetter:
    def __init__(self, functions, k):
        self.functions = functions
        self.k = k

    def __getitem__(self, i):
        event = self.functions[i]

        def wrapped_function(t, x):
            return event(t, *x, *self.k)

        return wrapped_function

    def __iter__(self):
        for function in self.functions:
            def wrapped_function(t, x):
                return function(t, *x, *self.k)

            yield wrapped_function


class LazyIntegrableSolution:
    def __init__(self, initials: array, odes: Callable[[float, array], array],
                 jacobian: Callable[[float, array], array], discontinuities: array,
                 dose_times: array, dose: Callable[[float, array], array],
                 triggers: List[Callable[[float, array], int]], directions: array,
                 effects: List[Callable[[float, array], array]]):
        self.initials = initials
        self.odes = odes
        self.jacobian = jacobian
        self.discontinuities = discontinuities
        self.dose_times = dose_times
        self.dose = dose
        self.triggers = triggers
        self.directions = directions
        self.effects = effects

        sorted_disc = sorted(set(self.discontinuities + self.dose_times + [inf]))
        self.all_discontinuities = [time for time in sorted_disc if time > 0]

        self.i_next_discontinuity = 0
        self.i_next_dose = 0

        if self.next_dose() == 0.0:
            y0 = self.dose(0.0, self.initials)
            self.i_next_dose += 1
        else:
            y0 = self.initials

        self.solver = Radau(self.odes, 0.0, y0, self.next_discontinuity(short=True), jac=self.jacobian)

        self.solutions = []
        self.last_event_value = np.asarray([event(0.0, y0) for event in self.triggers])

    def integrate_to(self, requested_time):
        while requested_time > self.solver.t:
            # Advance the solver
            if self.solver.status == 'running':  # or self.solver.status == 'started':
                # Take step with existing solver
                self.solver.step()

                # Die immediately if solver failed
                if self.solver.status == 'failed':
                    raise IntegrationFailureException(self.solver.t)

                # Save state fom successful step
                last_time = self.solver.t_old
                this_time = self.solver.t
                this_state = self.solver.y
                self.solutions.append(self.solver.dense_output())

                # Detect events
                last_event_value = self.last_event_value
                this_event_value = np.asarray([event(this_time, this_state)
                                               for event in self.triggers])
                self.last_event_value = this_event_value

                i_active_events = find_active_events(last_event_value, this_event_value, self.directions)
                if i_active_events.size > 0:
                    sol = self.solutions[-1]
                    # Assume that there is a maximum of one root per event per step
                    roots = [solve_event_equation(self.triggers[i_event], sol, last_time, this_time)
                             for i_event in i_active_events]
                    i_first_active = np.argmin(roots)
                    t_first_active = roots[i_first_active]
                    i_event = i_active_events[i_first_active]

                    y0 = self.effects[i_event](t_first_active, sol(t_first_active))

                    # Initialize a new solver
                    self.solver = Radau(self.odes, t_first_active, y0,
                                        self.next_discontinuity(short=True), jac=self.jacobian)
            elif self.solver.status == 'finished':
                # Reached a discontinuity, apply doses and initialize a new solver

                current_discontinuity = self.next_discontinuity()
                self.i_next_discontinuity += 1

                # Handle doses
                if current_discontinuity == self.next_dose():
                    y0 = self.dose(current_discontinuity, self.solver.y)
                    self.i_next_dose += 1
                else:
                    y0 = self.solver.y

                # Initialize a new solver
                self.solver = Radau(self.odes, current_discontinuity, y0,
                                    self.next_discontinuity(short=True), jac=self.jacobian)
            else:
                raise IntegrationFailureException(self.solver.t)

    def next_discontinuity(self, short=False):
        next_discontinuity = self.all_discontinuities[self.i_next_discontinuity]

        if short:
            if next_discontinuity == inf:
                return next_discontinuity
            else:
                return np.nextafter(next_discontinuity, -np.inf)
        else:
            return next_discontinuity

    def next_dose(self):
        if self.i_next_dose < len(self.dose_times):
            return self.dose_times[self.i_next_dose]
        else:
            return inf

    def __call__(self, when: Union[Real, Sequence[Real]]):
        max_when = when if isinstance(when, Real) else max(when)
        self.integrate_to(max_when)

        t_firsts = [solution.t_old for solution in self.solutions]

        inds = np.searchsorted(t_firsts, when, side='right') - 1
        if isinstance(when, Real):
            if when == self.solver.t:
                # No integration yet
                return self.solver.y
            else:
                return self.solutions[inds](when)
        else:
            output = np.zeros((len(when), len(self.initials)))
            for i_when in range(len(when)):
                if when[i_when] == self.solver.t:
                    # No integration yet
                    output[i_when] = self.solver.y
                else:
                    output[i_when] = self.solutions[inds[i_when]](when[i_when])
            return output


class IntegrationFailureException(Exception):
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return f'Integration failure at t = {self.t}'
