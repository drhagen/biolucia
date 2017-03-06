from typing import Dict, List
from collections import OrderedDict

from biolucia.model import Model

from biolucia.simulation import Simulation, BioluciaSystemSimulation


class Experiment:
    """Abstract base class for experimental conditions, which when combined with a model make various simulations
    available"""
    def simulate(self, model, final_time=0.0) -> Simulation:
        raise NotImplementedError

    def default_parameters(self) -> 'OrderedDict[str, float]':
        raise NotImplementedError

    # def parameter_values(self, parameter_names: List[str]):
    #     raise NotImplementedError
    #
    def update_parameters(self, parameters: Dict[str, float]) -> 'Experiment':
        raise NotImplementedError

    # def simulate_variance(self, model, final_time=0.0, method='mfk'):
    #     raise NotImplementedError
    #
    # def simulate_stochastic(self, model, final_time=0.0):
    #     raise NotImplementedError


class InitialValueExperiment(Experiment):
    def __init__(self, variant: Model = Model()):
        self.variant = variant

    def simulate(self, model: Model, final_time: float = 0.0, parameters: List[str] = ()) -> BioluciaSystemSimulation:
        system = model.update(self.variant)
        return BioluciaSystemSimulation(system, final_time, parameters)

    def default_parameters(self) -> 'OrderedDict[str, float]':
        return self.variant.default_parameters()

    def update_parameters(self, parameters: Dict[str, float]) -> 'InitialValueExperiment':
        variant = self.variant.update_parameters(parameters)
        return InitialValueExperiment(variant)


class SteadyStateExperiment(Experiment):
    def __init__(self, starter: Model = Model(), variant: Model = Model()):
        self.starter = starter
        self.variant = variant

    def simulate(self, model: Model, final_time=0.0, parameters: List[str] = ()) -> BioluciaSystemSimulation:
        starter = model.update(self.starter)
        system = model.update(self.variant)

        starter = starter.run_to_steady_state()
        system = system.update_initial(starter)

        return BioluciaSystemSimulation(system, final_time, parameters)

    def default_parameters(self) -> 'OrderedDict[str, float]':
        return self.variant.default_parameters()

    def update_parameters(self, parameters: Dict[str, float]) -> 'SteadyStateExperiment':
        starter_dictionary = dict(entry for entry in parameters.items()
                                  if entry[0] in self.starter.default_parameters().keys())
        starter = self.starter.update_parameters(starter_dictionary)
        variant = self.variant.update_parameters(parameters)
        return SteadyStateExperiment(starter, variant)
