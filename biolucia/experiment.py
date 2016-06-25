from biolucia.base import Experiment
from biolucia.model import Model
from biolucia.base import Simulation
from typing import Sequence, Union
import numpy as np
from numpy import inf, nextafter
from scipy.integrate import ode
from scipy.interpolate import interp1d
from numbers import Real


class InitialValueExperiment(Experiment):
    def __init__(self, variant: Model = Model()):
        self.variant = variant

    def _simulate(self, model: Model, final_time=0.0):
        system = model.update(self.variant)
        return BioluciaSystemSimulation(system, final_time)


class SteadyStateExperiment(Experiment):
    def __init__(self, starter: Model = Model(), variant: Model = Model()):
        self.starter = starter
        self.variant = variant

    def _simulate(self, model: Model, final_time=0.0):
        starter = model.update(self.starter)
        system = model.update(self.variant)

        starter = starter.run_to_steady_state()
        system = system.update_initial(starter)

        return BioluciaSystemSimulation(system, final_time)


class BioluciaSystemSimulation(Simulation):
    def __init__(self, system: Model, final_time: float = 0.0):
        self._observable_names = system.observable_names()
        self.ode_system = system.build_odes()

        self.final_time = 0.0
        self.final_state = self.ode_system.ics()

        self.solution_times = [0.0]
        self.solution_states = [self.final_state]

        # Ensure the discontinuities are sorted and there is always an infinity and never a 0
        sorted_disc = sorted(set(self.ode_system.discontinuities + self.ode_system.dose_times + (inf,)))
        zeroless_disc = sorted_disc if sorted_disc[0] != 0.0 else sorted_disc[1:]
        self.all_discontinuities = zeroless_disc

        self.i_next_discontinuity = 0

        self.integrate_to(final_time)

    def integrate_to(self, requested_time):
        if requested_time > self.final_time:
            # Create ODE system
            obj = ode(self.ode_system.odes, self.ode_system.jacobian)
            # TODO (drhagen): Save this obj on self when reentrant solvers become available

            while True:
                next_discontinuity = self.all_discontinuities[self.i_next_discontinuity]
                if next_discontinuity <= requested_time:
                    # Going to hit a discontinuity before reaching the requested time
                    soft_stop = False  # Do not run over the discontinuity
                    end_time = next_discontinuity
                    integration_end_time = nextafter(end_time, -inf)
                else:
                    # Going to finish the integration in this segment
                    soft_stop = True  # It is OK to step over a requested time
                    end_time = requested_time
                    integration_end_time = end_time
                    # TODO (drhagen): It is possible for requested_time to be slightly less than a hard stop,
                    # causing soft_step to be True and then stepping over both requested_time and the discontinuity.
                    # There is no good solution to this with scipy solvers.

                # Start where we left off last time
                obj.set_initial_value(self.final_state, self.final_time)

                # Integrate until we pass needed point
                ts_new = []
                ys_new = []

                while obj.t < integration_end_time:
                    y_new = obj.integrate(integration_end_time, step=True, relax=soft_stop)

                    if not obj.successful():
                        raise IntegrationFailureException(obj.t)

                    if obj.t <= integration_end_time or soft_stop:
                        ts_new.append(obj.t)
                        ys_new.append(y_new)
                    else:
                        # Integrator stepped over discontinuity, restart from previous point and take one step to end
                        # TODO (drhagen): Find a better ODE integrator because this is hacky and expensive
                        t_last = ts_new[-1]
                        obj = ode(self.ode_system.odes, self.ode_system.jacobian)
                        obj.set_initial_value(ys_new[-1], t_last)
                        y_new = obj.integrate(integration_end_time, step=False, relax=False)
                        ts_new.append(obj.t)
                        ys_new.append(y_new)

                if not soft_stop:
                    # This is a discontinuity, which needs to be recorded
                    ts_new.append(end_time)
                    if end_time in self.ode_system.dose_times:
                        # A dose discontinuity
                        ys_new.append(self.ode_system.doses(end_time, y_new))
                    else:
                        # Just a regular discontinuity
                        # (Keep the time because it is needed to plot the discontinuities even if the state is
                        # continuous)
                        ys_new.append(y_new)
                    self.i_next_discontinuity += 1

                self.final_time = ts_new[-1]
                self.final_state = ys_new[-1]

                self.solution_times += ts_new
                self.solution_states += ys_new

                if self.final_time >= requested_time:
                    # Done with integrating for now
                    break

    def system_values(self, when: Union[Real, Sequence[Real]], which: Union[str, Sequence[str]]=None):
        which = self._observable_names if which is None else which
        max_when = when if isinstance(when, Real) else max(when)

        self.integrate_to(max_when)

        if len(self.solution_times) == 1:
            # Handle scipy bug when there is only one time point
            # TODO (drhagen): super hacky solution here
            state_interpolator = lambda t: self.solution_states[0]
        else:
            state_interpolator = interp1d(self.solution_times, self.solution_states, axis=0, assume_sorted=True,
                                          copy=False)

        # Extract values from solution
        output_fun = self.ode_system.outputs
        if isinstance(which, str) and isinstance(when, Real):
            states = state_interpolator(when)
            return output_fun(which, when, states)
        elif isinstance(which, str):
            return np.fromiter((output_fun(which, when_i, state_interpolator(when_i)) for when_i in when),
                               'float', count=len(when))
        elif isinstance(when, Real):
            states = state_interpolator(when)
            return np.fromiter((output_fun(which_i, when, states) for which_i in which),
                               'float', count=len(which))
        else:
            def values():
                for when_i in when:
                    states = state_interpolator(when_i)
                    for which_i in which:
                        yield output_fun(which_i, when_i, states)

            values = np.fromiter(values(), 'float', count=len(which)*len(when))
            return np.reshape(values, [len(when), len(which)])


class IntegrationFailureException(Exception):
    def __init__(self, t):
        self.t = t

    def __str__(self):
        return 'Integration failure at t = {t}'.format(t=self.t)
