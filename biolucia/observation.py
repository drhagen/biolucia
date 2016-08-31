from numbers import Real
from typing import Sequence, Callable
from sympy import Symbol

import numpy as np
from numpy import log
from numpy.random import normal
from scipy.stats import multivariate_normal

from .helpers import tau
from .simulation import Simulation


class Observation:
    """Abstract base class for observation schemes, which define a probability distribution over simulation space"""
    def probability(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError()

    def log_probability(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError()

    def sample(self, simulation: Simulation):
        raise NotImplementedError()

    def objective(self, measurements: Sequence[Real]):
        raise NotImplementedError()


class Objective:
    def G(self, sim: Simulation):
        raise NotImplementedError()


class LinearWeightedSumOfSquaresObservation(Observation):
    def __init__(self, ts: Sequence[Real], ys: Sequence[Symbol], uncertainty_function: Callable[[Real, Symbol, Real], Real]):
        ts = np.asarray(ts)
        ys = np.asarray(ys)

        if ts.ndim != 1:
            raise ValueError("ts must be a 1-dimensional array")
        if ys.ndim != 1:
            raise ValueError("ys must be a 1-dimensional array")

        self.n = ts.size
        self.ts = ts
        self.ys = ys
        self.uncertainty_function = uncertainty_function
        self.t_max = np.amax(ts) if ts.size > 0 else 0

    def probability(self, simulation: Simulation, measurements: Sequence[Real]):
        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = np.asarray(self.uncertainty_function(self.ts, self.ys, ybars))

        return multivariate_normal.pdf(measurements, ybars, sigmas ** 2)

    def log_probability(self, simulation: Simulation, measurements: Sequence[Real]):
        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = np.asarray(self.uncertainty_function(self.ts, self.ys, ybars))

        # es = ybars - measurements
        # return -self.n / 2 * log(tau) + -sum(log(sigmas)) + -1 / 2 * sum((es / sigmas)**2)
        return multivariate_normal.logpdf(measurements, ybars, sigmas ** 2)

    def sample(self, simulation: Simulation):
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = np.asarray(self.uncertainty_function(self.ts, self.ys, ybars))

        if self.n == 1:
            # Work around Numpy bug #7983
            return normal(list(ybars), sigmas)
        else:
            return normal(ybars, sigmas)

    def objective(self, measurements: Sequence[Real]):
        return LinearWeightedSumOfSquaresObjective(self, measurements)


# TODO: make this inner class to its observation?
class LinearWeightedSumOfSquaresObjective(Objective):
    def __init__(self, observation: LinearWeightedSumOfSquaresObservation, measurements: Sequence[Real]):
        self.observation = observation
        self.measurements = measurements

        # Undo the constant when computing G from logp so that G is ~= 0 when ybar ~= measurements
        sigmas = np.asarray(observation.uncertainty_function(observation.ts, observation.ys, measurements))
        self.offset = -observation.n / 2 * log(tau) + -sum(log(sigmas))

    def G(self, sim: Simulation):
        return -2 * (self.observation.log_probability(sim, self.measurements) - self.offset)
