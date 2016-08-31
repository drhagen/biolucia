from typing import Sequence, Union
from numbers import Real

import numpy as np
from numpy import concatenate, inf
from scipy_ode import Radau, SolverStatus

from .helpers import eps


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


class LazyIntegrableSolution:
    def __init__(self, system: IntegrableSystem):
        self.system = system

        sorted_disc = np.asarray(sorted(set(self.system.discontinuities + self.system.dose_times + (inf,))))
        self.all_discontinuities = sorted_disc[sorted_disc > 0]

        self.i_next_discontinuity = 0
        self.i_next_dose = 0

        if self.next_dose() == 0:
            y0 = self.system.doses(0.0, system.ics())
            self.i_next_dose += 1
        else:
            y0 = system.ics()

        self.solver = Radau(system.odes, y0, 0.0, self.next_discontinuity(short=True), jac=system.jacobian)

        # TODO: deconstruct state and write our own interpolator unless we give users choice of solver???
        self.states = [[self.solver.state]]
        self.solutions = [self.solver.interpolator(self.states[-1])]

    def integrate_to(self, requested_time):
        if requested_time > self.final_time:
            while requested_time > self.final_time:
                # Advance the solver
                if self.solver.status == SolverStatus.running or self.solver.status == SolverStatus.started:
                    # Take step with existing solver
                    self.solver.step()
                    self.states[-1].append(self.solver.state)

                    # Detect events
                    # TODO
                elif self.solver.status == SolverStatus.finished:
                    # Reached a discontinuity, apply doses and initialize a new solver

                    # Replace last solution with final solution
                    self.solutions[-1] = self.solver.interpolator(self.states[-1])

                    current_discontinuity = self.next_discontinuity()
                    self.i_next_discontinuity += 1

                    # Handle doses
                    if current_discontinuity == self.next_dose():
                        y0 = self.system.doses(current_discontinuity, self.solver.state.y)
                        self.i_next_dose += 1
                    else:
                        y0 = self.solver.state.y

                    # Initialize a new solver
                    self.solver = Radau(self.system.odes, y0, current_discontinuity,
                                        self.next_discontinuity(short=True), jac=self.system.jacobian)

                    self.states.append([self.solver.state])
                    self.solutions.append(None)
                else:
                    raise IntegrationFailureException(self.solver.state.t)

            # Update the interpolator for the final segment
            self.solutions[-1] = self.solver.interpolator(self.states[-1])

    @property
    def final_time(self):
        return self.states[-1][-1].t

    def next_discontinuity(self, short=False):
        next_discontinuity = self.all_discontinuities[self.i_next_discontinuity]

        if short:
            if next_discontinuity == inf:
                return next_discontinuity
            else:
                return next_discontinuity - 16 * eps * next_discontinuity
        else:
            return next_discontinuity

    def next_dose(self):
        if self.i_next_dose < len(self.system.dose_times):
            return self.system.dose_times[self.i_next_dose]
        else:
            return inf

    def __call__(self, when: Union[Real, Sequence[Real]]):
        max_when = when if isinstance(when, Real) else max(when)
        self.integrate_to(max_when)

        inds = np.searchsorted(self.all_discontinuities, when, side='right')
        if isinstance(when, Real):
            return self.solutions[inds](when)
        else:
            pass


class IntegrationFailureException(Exception):
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return 'Integration failure at t = {t}'.format(t=self.t)
