from numbers import Real
from typing import Union, Sequence

import numpy as np
from numpy import inf

from .model import Model
from .ode import LazyIntegrableSolution


class Simulation:
    """Abstract base class for simulations, which lazily produce values for all components at all
    time points 0 to infinity"""
    def matrix_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None):
        raise NotImplementedError()

    def vector_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)

        result_matrix = self.matrix_values(when_unique, which_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes]

    def matrix_sensitivities(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None,
                             parameters: Union[str, Sequence[str], None] = None):
        raise NotImplementedError()

    def vector_sensitivities(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]],
                             parameters: Union[str, Sequence[str], None] = None):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)
        parameters_unique, parameters_unique_indexes = np.unique(parameters, return_inverse=True)

        result_matrix = self.matrix_sensitivities(when_unique, which_unique, parameters_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes, parameters_unique]

    def matrix_curvature(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str], None]=None,
                         parameters1: Union[str, Sequence[str], None] = None,
                         parameters2: Union[str, Sequence[str], None] = None):
        raise NotImplementedError()

    def vector_curvature(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]],
                         parameters1: Union[str, Sequence[str], None] = None,
                         parameters2: Union[str, Sequence[str], None] = None):
        when_unique, when_unique_indexes = np.unique(when, return_inverse=True)
        which_unique, which_unique_indexes = np.unique(which, return_inverse=True)
        parameters1_unique, parameters1_unique_indexes = np.unique(parameters1, return_inverse=True)
        parameters2_unique, parameters2_unique_indexes = np.unique(parameters2, return_inverse=True)

        result_matrix = self.matrix_sensitivities(when_unique, which_unique, parameters1_unique, parameters2_unique)

        return result_matrix[when_unique_indexes, which_unique_indexes, parameters1_unique, parameters2_unique]


class BioluciaSystemSimulation(Simulation):
    def __init__(self, system: Model, final_time: float = 0.0):
        self._observable_names = system.observable_names()
        self.ode_system = system.build_odes()

        self.solution = LazyIntegrableSolution(self.ode_system)
        self.solution.integrate_to(final_time)

    def matrix_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]=None):
        which = self._observable_names if which is None else which

        # Extract values from solution
        output_fun = self.ode_system.outputs
        if isinstance(which, str) and isinstance(when, Real):
            states = self.solution(when)
            return output_fun(which, when, states)
        elif isinstance(which, str):
            return np.fromiter((output_fun(which, when_i, self.solution(when_i)) for when_i in when),
                               'float', count=len(when))
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
