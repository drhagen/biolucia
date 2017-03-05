from numbers import Real
from typing import Union, Sequence, Dict, Tuple

import numpy as np
from scipy import sparse

from .model import Model
from .ode import LazyIntegrableSolution


class Simulation:
    """Abstract base class for simulations, which lazily produce values for all components at all
    time points 0 to infinity"""
    def matrix_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None):
        raise NotImplementedError

    def vector_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_values(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes]

    def matrix_sensitivities(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None):
        raise NotImplementedError

    def vector_sensitivities(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_sensitivities(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes, :]

    def matrix_curvatures(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None):
        raise NotImplementedError

    def vector_curvatures(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_curvatures(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes]


class BioluciaSystemSimulation(Simulation):
    def __init__(self, system: Model, final_time: float = 0.0, parameters = ()):
        self._observable_names = system.observable_names()
        self.ode_system = system.build_odes(parameters)
        self.parameters = parameters

        self.solution = LazyIntegrableSolution(self.ode_system.x0, self.ode_system.f, self.ode_system.f_dx,
                                               self.ode_system.discontinuities, self.ode_system.t_dose,
                                               self.ode_system.d, self.ode_system.e, self.ode_system.e_dir,
                                               self.ode_system.e_eff)
        self.solution.integrate_to(final_time)

        self.sensitivities = LazyIntegrableSolution(self.ode_system.x0_dk.flatten(), self.odes_dk, self.jacobian_dk,
                                                    self.ode_system.discontinuities, [], [],
                                                    [], np.empty((0,)), [])

    def matrix_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]] = None):
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

            values = np.fromiter(values(), 'float', count=len(which)*len(when))
            return np.reshape(values, [len(when), len(which)])

    def odes_dk(self, t, x_dk):
        nx = len(self.ode_system.x0)
        nk = len(self.ode_system.k)
        x = self.solution(t)

        return (self.ode_system.f_dx(t, x) @ x_dk.reshape(nx, nk) + self.ode_system.f_dk(t, x)).flatten()

    def jacobian_dk(self, t, x_dk):
        nk = len(self.ode_system.k)
        x = self.solution(t)

        return sparse.kron(sparse.identity(nk), self.ode_system.f_dx(t, x), format='csc')

    def matrix_sensitivities(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None] = None):
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
