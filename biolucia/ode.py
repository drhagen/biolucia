from biolucia.helpers import *
from typing import Sequence
from numpy import concatenate


class IntegrableSystem:
    def __init__(self, n_parameters, n_states, parameters, ics, odes, jacobian, discontinuities, doses, dose_times,
                 outputs):
        self.n_parameters = n_parameters
        self.n_states = n_states
        self.parameters = parameters
        self._ics = ics
        self._odes = odes
        self._jacobian = jacobian
        self.discontinuities = discontinuities
        self._doses = doses
        self.dose_times = dose_times
        self._outputs = outputs

    def ics(self):
        return self._ics(*self.parameters)

    def odes(self, time, states):
        return self._odes(*concatenate(((time,), states, self.parameters)))

    def jacobian(self, time, states):
        return self._jacobian(*concatenate(((time,), states, self.parameters)))

    def doses(self, time, states):
        return self._doses[time](*concatenate(((time,), states, self.parameters)))

    def outputs(self, name, time, states):
        return self._outputs[name](*concatenate(((time,), states, self.parameters)))

    def update(self, parameters):
        return IntegrableSystem(self.n_parameters, self.n_states, parameters, self._ics, self._odes, self._jacobian,
                                self.discontinuities, self._doses, self.dose_times, self._outputs)
