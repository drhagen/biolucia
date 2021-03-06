from numbers import Real
from typing import Union, List

import numpy as np
from numpy import ndarray
from scipy import sparse

from multipledispatch import Dispatcher

from .model import Model, update
from .experiment import Experiment, InitialValueExperiment, SteadyStateExperiment
from .ode import LazyIntegrableSolution, ListDiscontinuitySupplier, ListDoseSupplier, LazyIntegrableSolutionDoseSupplier


class Simulation:
    """Abstract base class for simulations, which lazily produce values for all components at all
    time points 0 to infinity"""

    def matrix_values(self, when: Union[Real, List[Real]], which: Union[str, List[str], None] = None) -> ndarray:
        raise NotImplementedError()

    def vector_values(self, when: Union[Real, List[Real]], which: Union[str, List[str]]) -> ndarray:
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_values(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes]

    def matrix_sensitivities(self, when: Union[Real, List[Real]], which: Union[str, List[str], None] = None) -> ndarray:
        raise NotImplementedError()

    def vector_sensitivities(self, when: Union[Real, List[Real]], which: Union[str, List[str]]) -> ndarray:
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_sensitivities(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes, :]

    def matrix_curvatures(self, when: Union[Real, List[Real]], which: Union[str, List[str], None] = None) -> ndarray:
        raise NotImplementedError()

    def vector_curvatures(self, when: Union[Real, List[Real]], which: Union[str, List[str]]) -> ndarray:
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_curvatures(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes]


class BioluciaSystemSimulation(Simulation):
    def __init__(self, system: Model, final_time: float = 0.0, parameters: List[str] = ()):
        self._observable_names = system.observable_names()
        self.ode_system = system.build_odes(parameters)
        self.parameters = parameters

        self.solution = LazyIntegrableSolution(self.ode_system.x0, self.ode_system.f, self.ode_system.f_dx,
                                               ListDiscontinuitySupplier(self.ode_system.discontinuities),
                                               ListDoseSupplier(self.ode_system.t_dose), self.ode_system.d,
                                               self.ode_system.e, self.ode_system.e_dir, self.ode_system.j)
        self.solution.integrate_to(final_time)

        self.sensitivities = LazyIntegrableSolution(self.ode_system.x0_dk.flatten(), self.odes_dk, self.jacobian_dk,
                                                    ListDiscontinuitySupplier(self.ode_system.discontinuities),
                                                    LazyIntegrableSolutionDoseSupplier(self.solution),
                                                    self.dose_effect_dk, [], np.empty((0,)), [])

    def matrix_values(self, when: Union[Real, List[Real]], which: Union[str, List[str]] = None) -> ndarray:
        which = self._observable_names if which is None else which

        # Extract values from solution
        output_fun = self.ode_system.y
        if isinstance(which, str) and isinstance(when, Real):
            states = self.solution(when)
            return output_fun(which, when, states)
        elif isinstance(which, str):
            # when is a sequence
            states = self.solution(when)
            outputs = np.zeros((len(when),))
            for i_when in range(len(when)):
                outputs[i_when] = output_fun(which, when[i_when], states[i_when])
            return outputs
        elif isinstance(when, Real):
            states = self.solution(when)
            return np.fromiter((output_fun(which_i, when, states) for which_i in which),
                               'float', count=len(which))
        else:
            def values():
                for when_i in when:
                    states = self.solution(when_i)
                    for which_i in which:
                        yield output_fun(which_i, when_i, states)

            values = np.fromiter(values(), 'float', count=len(which) * len(when))
            return np.reshape(values, [len(when), len(which)])

    def odes_dk(self, t, x_dk) -> ndarray:
        nx = len(self.ode_system.x0)
        nk = len(self.ode_system.k)
        x = self.solution(t)

        return (self.ode_system.f_dx(t, x) @ x_dk.reshape(nx, nk) + self.ode_system.f_dk(t, x)).flatten()

    def jacobian_dk(self, t, x_dk) -> ndarray:
        nk = len(self.ode_system.k)
        x = self.solution(t)

        return sparse.kron(sparse.identity(nk), self.ode_system.f_dx(t, x), format='csc')

    def dose_effect_dk(self, t, x_dk) -> ndarray:
        nx = len(self.ode_system.x0)
        nk = len(self.ode_system.k)

        # In the rare case that a dose and event occur at the same time, apply event first
        if t in self.solution.detection_times:
            i_time = self.solution.detection_times.index(t)
            i = self.solution.detection_indexes[i_time]
            x_pre = self.solution.detection_pre_states[i_time]
            x_post = self.solution.detection_post_states[i_time]

            e_dk_pre = self.ode_system.e_dk[i](t, x_pre).reshape((1, -1))
            e_dx_pre = self.ode_system.e_dx[i](t, x_pre).reshape((1, -1))
            e_dt_pre = self.ode_system.e_dt[i](t, x_pre)
            x_dt_pre = self.ode_system.f(t, x_pre).reshape((-1, 1))

            t_dk = - (e_dk_pre + e_dx_pre @ x_dk) / (e_dt_pre + e_dx_pre @ x_dt_pre)

            j_dk_pre = self.ode_system.j_dk[i](t, x_pre)
            j_dx_pre = self.ode_system.j_dx[i](t, x_pre)
            j_dt_pre = self.ode_system.j_dt[i](t, x_pre).reshape((-1, 1))

            j_dk = j_dk_pre + j_dx_pre @ x_dk.reshape((-1, 1)) + j_dt_pre @ t_dk + (j_dx_pre @ x_dt_pre) * t_dk

            x_dt_post = self.ode_system.f(t, x_post).reshape((-1, 1))

            x_dk = (j_dk - x_dt_post * t_dk).flatten()

        if t in self.solution.dose_times:
            i_time = self.solution.dose_times.index(t)
            x = self.solution.dose_states[i_time]
            x_dk = (self.ode_system.d_dx(t, x) @ x_dk.reshape(nx, nk) + self.ode_system.d_dk(t, x)).flatten()

        return x_dk

    def matrix_sensitivities(self, when: Union[Real, List[Real]], which: Union[str, List[str], None] = None) -> ndarray:
        which = self._observable_names if which is None else which

        # Extract values from solution
        if isinstance(which, str) and isinstance(when, Real):
            states = self.solution(when)
            sensitivities = self.sensitivities(when).reshape((len(states), len(self.ode_system.k)))
            return self.ode_system.y_dx(which, when, states) @ sensitivities + self.ode_system.y_dk(which, when, states)
        elif isinstance(which, str):
            # when is a sequence
            outputs = np.empty((len(when), len(self.parameters)))
            for i_when in range(len(when)):
                states = self.solution(when[i_when])
                sensitivities = self.sensitivities(when[i_when]).reshape((len(states), len(self.ode_system.k)))
                outputs[i_when, :] = (self.ode_system.y_dx(which, when[i_when], states) @ sensitivities
                                      + self.ode_system.y_dk(which, when[i_when], states))
            return outputs
        elif isinstance(when, Real):
            states = self.solution(when)
            sensitivities = self.sensitivities(when).reshape((len(states), len(self.ode_system.k)))
            outputs = np.empty((len(which), len(self.parameters)))
            for i_which in range(len(which)):
                outputs[i_which, :] = (self.ode_system.y_dx(which[i_which], when, states) @ sensitivities
                                       + self.ode_system.y_dk(which[i_which], when, states))
            return outputs
        else:
            outputs = np.empty((len(when), len(which), len(self.parameters)))
            for i_when in range(len(when)):
                states = self.solution(when[i_when])
                sensitivities = self.sensitivities(when[i_when]).reshape((len(states), len(self.ode_system.k)))
                for i_which in range(len(which)):
                    outputs[i_when, i_which, :] = (self.ode_system.y_dx(which[i_which], when[i_when], states)
                                                   @ sensitivities
                                                   + self.ode_system.y_dk(which[i_which], when[i_when], states))
            return outputs


# Multiple dispatch contraption for simulate
def simulate(model: Model, experiments: Union[Experiment, List[Experiment]] = InitialValueExperiment(), **kwargs) -> \
        Union[Simulation, List[Experiment]]:
    if isinstance(experiments, Experiment):
        return simulate.dispatcher(model, experiments, **kwargs)
    else:
        return [simulate.dispatcher(model, experiment, **kwargs) for experiment in experiments]


simulate.dispatcher = Dispatcher('simulate')


@simulate.dispatcher.register(Model, InitialValueExperiment)
def simulate_analytic_initial(model: Model, experiment: InitialValueExperiment, *,
                              final_time: float = 0.0, parameters: List[str] = ()):
    system = update(model, experiment.variant)
    return BioluciaSystemSimulation(system, final_time, parameters)


@simulate.dispatcher.register(Model, SteadyStateExperiment)
def simulate_analytic_steady_state(model: Model, experiment: SteadyStateExperiment, *,
                                   final_time: float = 0.0, parameters: List[str] = ()):
    starter = update(model, update(experiment.starter))
    system = update(model, experiment.variant)

    # TODO: or something like this...
    starter = run_to_steady_state(starter)
    system = system.update_initial(starter)

    return BioluciaSystemSimulation(system, final_time, parameters)
