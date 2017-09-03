from typing import Sequence, Union, Tuple, List, Callable
from numbers import Real
from collections import OrderedDict
from sympy import Expr, lambdify
import copy

import numpy as np
from numpy import inf, nan, ndarray
from scipy.integrate import Radau, DenseOutput
from scipy.integrate._ivp.ivp import find_active_events, solve_event_equation

from .analytic import multidimensional_derivative, multidimensional_lambdify


class DoseSupplier:
    def supply(self, start: float, end: float = None):
        """Provides all dose times between start (exclusive) and end (inclusive). If only start is supplied, then any
        doses equal to it are returned."""
        raise NotImplementedError()

    def contains(self, t: float):
        raise NotImplementedError()


class ListDoseSupplier(DoseSupplier):
    def __init__(self, items: Union[ndarray, List[float]]):
        """items must be sorted, positive, finite, and unique"""
        self.items = np.asarray(items)  # type: ndarray

    def supply(self, start: float, end: float = None):
        # TODO: use searchsorted
        if end is None:
            return self.items[self.items == start]
        else:
            return self.items[(self.items > start) & (self.items <= end)]

    def contains(self, t: float):
        return t in self.items


class DiscontinuitySupplier:
    def supply(self, start: float, end: float = None):
        """Subset of discontinuities starting with the first one past start and first one at or past end."""
        raise NotImplementedError()

    def contains(self, t: float):
        raise NotImplementedError()


class ListDiscontinuitySupplier(DiscontinuitySupplier):
    def __init__(self, items: Union[ndarray, List[float]]):
        """items must be sorted, positive, finite, and unique"""
        self.items = np.asarray(items)  # type: ndarray

    def supply(self, start: float, end: float = None):
        if end is None or start == end:
            last_index = np.searchsorted(self.items, start, side='right')
            if last_index == len(self.items):
                return np.reshape(inf, (1,))
            else:
                return np.reshape(self.items[last_index], (1,))
        else:
            first_index = np.searchsorted(self.items, start, side='right')
            last_index = np.searchsorted(self.items, end, side='left')

            found_discontinuities = self.items[first_index:last_index + 1]

            if last_index == len(self.items):
                # There is no discontinuity beyond end
                found_discontinuities = np.append(found_discontinuities, inf)

            return found_discontinuities

    def contains(self, t: float):
        return t in self.items


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

        self.e_dir = np.asarray([event[1] for event in events])

        self.e_syms = [event[0] for event in events]
        self.e_func = [lambdify(all_parameters, e_i) for e_i in self.e_syms]

        self.e_dt_syms = [e_i.diff('t') for e_i in self.e_syms]
        self.e_dt_func = [lambdify(all_parameters, e_dt_i) for e_dt_i in self.e_dt_syms]

        self.e_dx_syms = [multidimensional_derivative(np.asarray(e_i), self.x_names) for e_i in self.e_syms]
        self.e_dx_func = [multidimensional_lambdify(all_parameters, e_dx_i) for e_dx_i in self.e_dx_syms]

        self.e_dk_syms = [multidimensional_derivative(np.asarray(e_i), self.k_names) for e_i in self.e_syms]
        self.e_dk_func = [multidimensional_lambdify(all_parameters, e_dk_i) for e_dk_i in self.e_dk_syms]

        self.j_syms = [event[2] for event in events]
        self.j_func = [lambdify(all_parameters, j_i) for j_i in self.j_syms]

        self.j_dt_syms = [np.asarray([j_i_i.diff('t') for j_i_i in j_i]) for j_i in self.j_syms]
        self.j_dt_func = [multidimensional_lambdify(all_parameters, j_dt_i) for j_dt_i in self.j_dt_syms]

        self.j_dx_syms = [multidimensional_derivative(j_i, self.x_names) for j_i in self.j_syms]
        self.j_dx_func = [multidimensional_lambdify(all_parameters, j_dx_i) for j_dx_i in self.j_dx_syms]

        self.j_dk_syms = [multidimensional_derivative(j_i, self.k_names) for j_i in self.j_syms]
        self.j_dk_func = [multidimensional_lambdify(all_parameters, j_dk_i) for j_dk_i in self.j_dk_syms]

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
        return np.asarray(self.f_func(t, *x, *self.k))

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
    def e_dt(self):
        return EventGetter(self.e_dt_func, self.k)

    @property
    def e_dx(self):
        return EventGetter(self.e_dx_func, self.k)

    @property
    def e_dk(self):
        return EventGetter(self.e_dk_func, self.k)

    @property
    def j(self):
        return EventGetter(self.j_func, self.k)

    @property
    def j_dt(self):
        return EventGetter(self.j_dt_func, self.k)

    @property
    def j_dx(self):
        return EventGetter(self.j_dx_func, self.k)

    @property
    def j_dk(self):
        return EventGetter(self.j_dk_func, self.k)

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


class EventDetection:
    def __init__(self, time: float, index: int):
        self.time = time
        self.index = index


def right_before(value: float):
    if value == inf:
        return value
    else:
        return np.nextafter(value, -np.inf)


class LazyIntegrableSolution:
    def __init__(self, initials: ndarray, odes: Callable[[float, ndarray], ndarray],
                 jacobian: Callable[[float, ndarray], ndarray], discontinuity_supplier: DiscontinuitySupplier,
                 dose_supplier: DoseSupplier, dose: Callable[[float, ndarray], ndarray],
                 triggers: List[Callable[[float, ndarray], int]], directions: ndarray,
                 effects: List[Callable[[float, ndarray], ndarray]]):
        self.initials = initials
        self.odes = odes
        self.jacobian = jacobian
        self.discontinuities = discontinuity_supplier
        self.dose_supplier = dose_supplier
        self.dose = dose
        self.triggers = triggers
        self.directions = directions
        self.effects = effects

        # Test for a dose at time 0
        if self.dose_supplier.supply(0.0):
            y0 = self.dose(0.0, self.initials)
        else:
            y0 = self.initials

        # First discontinuity is guaranteed to have length == 1
        first_discontinuity = right_before(self.discontinuities.supply(0.0)[0])

        self.solver = Radau(self.odes, 0.0, y0, first_discontinuity, jac=self.jacobian)

        self.solutions: List[DenseOutput] = []

        self.dose_times: List[float] = []
        self.dose_states: List[ndarray] = []

        self.detection_indexes: List[int] = []
        self.detection_times: List[float] = []
        self.detection_pre_states: List[ndarray] = []
        self.detection_post_states: List[ndarray] = []

        self.last_event_value = np.asarray([event(0.0, y0) for event in self.triggers])

    def integrate_to(self, requested_time):
        if requested_time <= self.solver.t:
            return

        dose_times = self.dose_supplier.supply(self.solver.t, requested_time)
        i_dose = 0

        # Guaranteed to end with a point at or beyond requested_time
        discontinuities = self.discontinuities.supply(self.solver.t, requested_time)
        i_discontinuity = 0

        while requested_time > self.solver.t:
            # Advance the solver
            if self.solver.status == 'running':
                # Take step with existing solver
                self.solver.step()

                # Die immediately if solver failed
                if self.solver.status == 'failed':
                    raise IntegrationFailureException(self.solver.t)

                # Save state fom successful step
                last_time = self.solver.t_old
                this_time = self.solver.t
                this_state = self.solver.y
                sol = self.solver.dense_output()
                self.solutions.append(sol)

                # Gather any extra doses if solver runs beyond requested_time
                # Do not do this if there are no parameters, otherwise it will run to infinity
                if requested_time < self.solver.t and self.solver.n != 0:
                    dose_times = np.concatenate([dose_times, self.dose_supplier.supply(requested_time, self.solver.t)])

                # Detect doses
                # Use dose_time_i == nan when no doses are available
                dose_time_i = dose_times[i_dose] if len(dose_times) > i_dose else nan
                if dose_time_i <= this_time:
                    # Dose encountered during this step, truncate solution
                    this_time = dose_time_i
                    this_state = sol(this_time)

                # Detect events
                last_event_value = self.last_event_value
                this_event_value = np.asarray([event(this_time, this_state)
                                               for event in self.triggers])
                self.last_event_value = this_event_value

                i_active_events = find_active_events(last_event_value, this_event_value, self.directions)

                # Apply events and doses
                if i_active_events.size > 0 or dose_time_i <= this_time:
                    # In the rare case of event occurring exactly on dose, apply event first then dose

                    if i_active_events.size > 0:
                        # Assume that there is a maximum of one root per event per step
                        roots = [solve_event_equation(self.triggers[i_event], sol, last_time, this_time)
                                 for i_event in i_active_events]
                        i_first_active = np.argmin(roots)
                        this_time = roots[i_first_active]
                        this_state = sol(this_time)
                        i_event = i_active_events[i_first_active]

                        self.detection_indexes.append(i_event)
                        self.detection_times.append(this_time)
                        self.detection_pre_states.append(this_state)

                        this_state = self.effects[i_event](this_time, this_state)

                        self.detection_post_states.append(this_state)

                    if dose_time_i == this_time:
                        # This will be triggered if (1) there is no event, (2) event occurred exactly on dose
                        self.dose_times.append(this_time)
                        self.dose_states.append(this_state)

                        this_state = self.dose(dose_times[i_dose], this_state)
                        i_dose += 1

                    # Initialize a new solver
                    self.solver = Radau(self.odes, this_time, this_state,
                                        right_before(discontinuities[i_discontinuity]), jac=self.jacobian)

            elif self.solver.status == 'finished':
                # Reached a discontinuity, maybe apply a dose and initialize a new solver

                if dose_times and dose_times[i_dose] == discontinuities[i_discontinuity]:
                    # Dose time coincides with discontinuity
                    y0 = self.dose(dose_times[i_dose], self.solver.y)
                    i_dose += 1
                else:
                    y0 = self.solver.y

                this_discontinuity = discontinuities[i_discontinuity]
                i_discontinuity += 1
                next_discontinuity = right_before(discontinuities[i_discontinuity])

                # Initialize a new solver
                self.solver = Radau(self.odes, this_discontinuity, y0, next_discontinuity, jac=self.jacobian)
            else:
                raise IntegrationFailureException(self.solver.t)

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


# For providing the discovered events of a simulation as doses to the sensitivities
class LazyIntegrableSolutionDoseSupplier(DoseSupplier):
    def __init__(self, integrable: LazyIntegrableSolution):
        self.integrable = integrable

    def supply(self, start: float, end: float = None):
        self.integrable.integrate_to(end if end is not None else start)

        dose_times = self.integrable.dose_supplier.supply(start, end)
        event_times = [time for time in self.integrable.detection_times if start < time <= end]

        return np.unique(np.concatenate([dose_times, event_times]))

    def contains(self, t: float):
        self.integrable.integrate_to(t)

        return self.integrable.dose_supplier.contains(t) or t in self.integrable.detection_times
