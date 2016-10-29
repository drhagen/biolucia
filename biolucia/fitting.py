from typing import List, Union, Sequence
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from .model import Model
from .experiment import Experiment
from .observation import Objective


class ActiveParameters:
    def __init__(self, model_parameters: 'OrderedDict[str, float]', experiment_parameters: 'Sequence[OrderedDict[str, float]]'):
        if isinstance(model_parameters, OrderedDict):
            pass
        elif isinstance(model_parameters, dict):
            model_parameters = OrderedDict(model_parameters)

        if isinstance(experiment_parameters, OrderedDict):
            experiment_parameters = [experiment_parameters]
        elif isinstance(experiment_parameters, dict):
            experiment_parameters = [OrderedDict(experiment_parameters)]

        self.model_parameters = model_parameters  # type: OrderedDict[str, float]
        self.experiment_parameters = experiment_parameters  # type: List[OrderedDict[str, float]]

    @property
    def values(self) -> np.array:
        k = list(self.model_parameters.values())

        for experiment in self.experiment_parameters:
            k.extend(experiment.values())

        return np.asarray(k)

    def update_parameters(self, values: np.array):
        n_model_parameters = len(self.model_parameters)
        n_experiment_parameters = [len(exp_i) for exp_i in self.experiment_parameters]
        n_parameters = n_model_parameters + sum(n_experiment_parameters)

        if len(values) != n_parameters:
            raise ValueError('{} parameters required, but {} received'.format(n_parameters, len(values)))

        model_parameters = []
        for name_i, i_parameter in zip(self.model_parameters.keys(), range(n_model_parameters)):
            model_parameters.append((name_i, values[i_parameter]))
        model_parameters = OrderedDict(model_parameters)

        experiment_parameters = []
        start_position = n_model_parameters
        for current_experiment_i, n_current_parameters in zip(self.experiment_parameters, n_experiment_parameters):
            experiment_parameters_i = []
            for name_i, i_parameter in zip(current_experiment_i.keys(),
                                           range(start_position, start_position + n_current_parameters)):
                experiment_parameters_i.append((name_i, values[i_parameter]))
            experiment_parameters_i = OrderedDict(experiment_parameters_i)
            start_position += n_current_parameters
            experiment_parameters.append(experiment_parameters_i)

        return ActiveParameters(model_parameters, experiment_parameters)

    @staticmethod
    def from_model_experiments(model, experiment, model_parameter_names, experiment_parameter_names):
        default_model_parameters = model.default_parameters()
        active_model_parameters = OrderedDict((name, default_model_parameters[name]) for name in model_parameter_names)

        if not experiment_parameter_names or isinstance(experiment_parameter_names[0], str):
            experiment_parameter_names = [experiment_parameter_names] * len(experiment)

        active_experiment_parameters = []
        for experiment_i, experiment_i_parameter_names in zip(experiment, experiment_parameter_names):
            default_experiment_i_parameters = experiment_i.default_parameters()
            active_experiment_parameters.append(OrderedDict((name, default_experiment_i_parameters[name])
                                                            for name in experiment_i_parameter_names))

        return ActiveParameters(active_model_parameters, active_experiment_parameters)

    def __eq__(self, other):
        if isinstance(other, ActiveParameters):
            return (self.model_parameters == other.model_parameters
                    and self.experiment_parameters == other.experiment_parameters)
        else:
            return NotImplemented

    def __repr__(self):
        return 'ActiveParameters({}, {})'.format(self.model_parameters, self.experiment_parameters)


def fit_parameters(model: Model, experiments: Union[Experiment, Sequence[Experiment]], objectives: Objective, *,
                   model_parameters, experiment_parameters=()):
    if isinstance(experiments, Experiment):
        experiments = [experiments]

    active_parameters = ActiveParameters.from_model_experiments(model, experiments, model_parameters,
                                                                experiment_parameters)

    def objective_function(k):
        nonlocal active_parameters, model
        active_parameters = active_parameters.update_parameters(k)
        model = model.update_parameters(active_parameters.model_parameters)
        G = 0

        for experiment_i, objective_i, experiment_parameters_i in zip(experiments, objectives,
                                                                      active_parameters.experiment_parameters):
            experiment_i = experiment_i.update_parameters(experiment_parameters_i)
            sim_i = model.simulate(experiment_i)
            G = G + objective_i.G(sim_i)

        return G

    res = minimize(objective_function, active_parameters.values)

    active_parameters = active_parameters.update_parameters(res.x)

    return active_parameters
