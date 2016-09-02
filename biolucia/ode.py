from typing import Sequence, Union
from numbers import Real

import numpy as np
from numpy import concatenate, inf
from scipy_ode import Radau, SolverStatus
from scipy_ode.ivp import get_active_events, solve_event_equation

from .helpers import eps


class IntegrableSystem:
    """Class that stores parameter values and wraps ODE functions"""
    def __init__(self, n_parameters, n_states, parameters, ics, odes, jacobian, discontinuities, doses, dose_times,
                 events, directions, effects, outputs):
        self.n_parameters = n_parameters
        self.n_states = n_states
        self.parameters = parameters
        self._ics = ics
        self._odes = odes
        self._jacobian = jacobian
        self.discontinuities = discontinuities
        self._doses = doses
        self.dose_times = dose_times
        self.events = [lambda time, states: event(*concatenate(((time,), states, self.parameters))) for event in events]
        self.effects = [lambda time, states: effect(*concatenate(((time,), states, self.parameters))) for effect in effects]
        self.directions = directions
        self._effects = effects
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
        self.last_event_value = np.asarray([event(0.0, y0) for event in system.events])

    def integrate_to(self, requested_time):
        if requested_time > self.final_time:
            while requested_time > self.final_time:
                # Advance the solver
                if self.solver.status == SolverStatus.running or self.solver.status == SolverStatus.started:
                    # Take step with existing solver
                    self.solver.step()

                    # Die immediately if solver failed
                    if self.solver.status == SolverStatus.failed:
                        raise IntegrationFailureException(self.solver.state.t)

                    # Save state fom successful step
                    last_time = self.states[-1][-1].t
                    this_time = self.solver.state.t
                    self.states[-1].append(self.solver.state)

                    # Detect events
                    last_event_value = self.last_event_value
                    this_event_value = np.asarray([event(self.solver.state.t, self.solver.state.y)
                                                   for event in self.system.events])
                    self.last_event_value = this_event_value

                    i_active_events = get_active_events(last_event_value, this_event_value, self.system.directions)
                    if i_active_events.size > 0:
                        sol = self.solver.interpolator(self.states[-1][-2:])
                        # Assume that there is a maximum of one root per event per step
                        roots = [solve_event_equation(self.system.events[i_event], sol, last_time, this_time)
                                 for i_event in i_active_events]
                        i_first_active = np.argmin(roots)
                        t_first_active = roots[i_first_active]
                        i_event = i_active_events[i_first_active]

                        y0 = self.system.effects[i_event](t_first_active, sol(t_first_active))

                        # Initialize a new solver
                        self.solver = Radau(self.system.odes, y0, t_first_active,
                                            self.next_discontinuity(short=True), jac=self.system.jacobian)

                        self.solutions[-1] = self.solver.interpolator(self.states[-1])
                        self.states.append([self.solver.state])
                        self.solutions.append(None)
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

        t_firsts = [state_list[0].t for state_list in self.states[1:]]

        inds = np.searchsorted(t_firsts, when, side='right')
        if isinstance(when, Real):
            return self.solutions[inds](when)
        else:
            output = np.zeros((len(when), self.system.n_states))
            for i_when in range(len(when)):
                output[i_when] = self.solutions[inds[i_when]](when[i_when])
            return output


class IntegrationFailureException(Exception):
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return 'Integration failure at t = {t}'.format(t=self.t)
