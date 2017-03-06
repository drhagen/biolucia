from numbers import Real
from typing import Sequence, Callable
from sympy import Symbol

import numpy as np
from numpy import log
from numpy.random import normal
from scipy.stats import multivariate_normal

from math import tau
from .simulation import Simulation


class MeasurementUncertainty:
    def sigma(self, ts: np.array, ys: np.array, ybars: np.array) -> np.array:
        raise NotImplementedError

    def sigma_dy(self, ts: np.array, ys: np.array, ybars: np.array) -> np.array:
        raise NotImplementedError

    def sigma_dy_dy(self, ts: np.array, ys: np.array, ybars: np.array) -> np.array:
        raise NotImplementedError


class AffineMeasurementUncertainty(MeasurementUncertainty):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sigma(self, ts: np.array, ys: np.array, ybars: np.array) -> np.array:
        return self.a + self.b * ybars

    def sigma_dy(self, ts: np.array, ys: np.array, ybars: np.array) -> np.array:
        return np.full(ybars.shape, self.b)

    def sigma_dy_dy(self, ts: np.array, ys: np.array, ybars: np.array):
        return np.zeros(ybars.shape)


class Observation:
    """Abstract base class for observation schemes, which define a probability distribution over simulation space"""
    def probability(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError

    def log_probability(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError

    def log_probability_dk(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError

    def log_probability_dk_dk(self, simulation: Simulation, measurements: Sequence[Real]):
        raise NotImplementedError

    def sample(self, simulation: Simulation):
        raise NotImplementedError

    def objective(self, measurements: Sequence[Real]):
        raise NotImplementedError


# Parameters are ys also
# These are all partial derivatives
class Objective:
    def G(self, sim: Simulation):
        raise NotImplementedError

    def G_dy(self, t: float, sim: Simulation):
        raise NotImplementedError

    def G_dy_dy(self, t: float, sim: Simulation):
        raise NotImplementedError


class LinearWeightedSumOfSquaresObservation(Observation):
    def __init__(self, ts: Sequence[Real], ys: Sequence[Symbol],
                 uncertainty_function: MeasurementUncertainty):
        ts = np.asarray(ts)
        ys = np.asarray(ys)

        if ts.ndim != 1:
            raise ValueError("ts must be a 1-dimensional array")
        if ys.ndim != 1:
            raise ValueError("ys must be a 1-dimensional array")

        self.n = ts.size
        self.ts = ts
        self.ys = ys
        self.uncertainty = uncertainty_function
        self.t_max = np.amax(ts) if ts.size > 0 else 0

    def probability(self, simulation: Simulation, measurements: Sequence[Real]):
        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = self.uncertainty.sigma(self.ts, self.ys, ybars)

        return multivariate_normal.pdf(measurements, ybars, sigmas ** 2)

    def log_probability(self, simulation: Simulation, measurements: Sequence[Real]):
        # Returns scalar

        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = self.uncertainty.sigma(self.ts, self.ys, ybars)

        # es = ybars - measurements
        # return -self.n / 2 * log(tau) + -sum(log(sigmas)) + -1 / 2 * sum((es / sigmas)**2)
        return multivariate_normal.logpdf(measurements, ybars, sigmas ** 2)

    def log_probability_dk(self, simulation: Simulation, measurements: Sequence[Real]):
        # Returns (nk,)

        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        ybars_dk = simulation.vector_sensitivities(self.ts, self.ys)
        sigmas = np.expand_dims(self.uncertainty.sigma(self.ts, self.ys, ybars), axis=1)
        sigmas_dy = np.expand_dims(self.uncertainty.sigma_dy(self.ts, self.ys, ybars), axis=1)

        es = np.expand_dims(ybars - measurements, axis=1)
        sigmas_dk = sigmas_dy * ybars_dk

        return -np.sum(sigmas_dk / sigmas, 0) - np.sum((es * (sigmas * ybars_dk - es * sigmas_dk)) / sigmas**3, 0)

    def log_probability_dk_dk(self, simulation: Simulation, measurements: Sequence[Real]):
        # Returns (nk, nk)

        measurements = np.asarray(measurements)
        ybars = simulation.vector_values(self.ts, self.ys)
        ybars_dk = simulation.vector_sensitivities(self.ts, self.ys)
        ybars_dk_dk = simulation.vector_curvatures(self.ts, self.ys)
        sigmas = self.uncertainty.sigma(self.ts, self.ys, ybars)
        sigmas_dy = self.uncertainty.sigma_dy(self.ts, self.ys, ybars)

        es = ybars - measurements
        sigmas_dk = sigmas_dy * ybars_dk
        sigmas_dk_dk = sigmas_dy * ybars_dk_dk

        # TODO: recheck this equation because the previous one was wrong
        temp1 = sigmas - ybars_dk - es * sigmas_dk
        return (np.sum(2 * sigmas_dk / sigmas ** 2, 0)
                - np.sum(sigmas_dk_dk / sigmas, 0)
                + np.sum(3 * es * temp1 / sigmas ** 4, 0)
                - np.sum((ybars_dk * temp1 + es * (sigmas_dk - ybars_dk_dk - ybars_dk * sigmas_dk - es * sigmas_dk_dk)) / sigmas ** 3), 0)

    def sample(self, simulation: Simulation):
        ybars = simulation.vector_values(self.ts, self.ys)
        sigmas = self.uncertainty.sigma(self.ts, self.ys, ybars)

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
        self.measurements = np.asarray(measurements)

        # Undo the constant when computing G from logp so that G is ~= 0 when ybar ~= measurements
        sigmas = observation.uncertainty.sigma(observation.ts, observation.ys, self.measurements)
        self.offset = -observation.n / 2 * log(tau) + -sum(log(sigmas))

    def G(self, sim: Simulation):
        return -2 * (self.observation.log_probability(sim, self.measurements) - self.offset)
