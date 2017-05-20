from typing import Dict, List
from collections import OrderedDict

from biolucia.model import Model


class Experiment:
    """Abstract base class for experimental conditions, which when combined with a model make various simulations
    available"""
    def default_parameters(self) -> 'OrderedDict[str, float]':
        raise NotImplementedError

    # def parameter_values(self, parameter_names: List[str]):
    #     raise NotImplementedError
    #
    def update_parameters(self, parameters: Dict[str, float]) -> 'Experiment':
        raise NotImplementedError


class InitialValueExperiment(Experiment):
    def __init__(self, variant: Model = Model()):
        self.variant = variant

    def default_parameters(self) -> 'OrderedDict[str, float]':
        return self.variant.default_parameters()

    def update_parameters(self, parameters: Dict[str, float]) -> 'InitialValueExperiment':
        variant = self.variant.update_parameters(parameters)
        return InitialValueExperiment(variant)


class SteadyStateExperiment(Experiment):
    def __init__(self, starter: Model = Model(), variant: Model = Model()):
        self.starter = starter
        self.variant = variant

    def default_parameters(self) -> 'OrderedDict[str, float]':
        return self.variant.default_parameters()

    def update_parameters(self, parameters: Dict[str, float]) -> 'SteadyStateExperiment':
        starter_dictionary = dict(entry for entry in parameters.items()
                                  if entry[0] in self.starter.default_parameters().keys())
        starter = self.starter.update_parameters(starter_dictionary)
        variant = self.variant.update_parameters(parameters)
        return SteadyStateExperiment(starter, variant)
