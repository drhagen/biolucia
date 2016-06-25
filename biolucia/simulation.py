# from biolucia.helpers import *
# from typing import Sequence
# from biolucia.model import Model
# from biolucia.base import Experiment,Simulation
# from biolucia.experiment import InitialValueExperiment,SteadyStateExperiment
# from scipy.integrate import ode
#
#
# def simulate_system(model: Model, experiment: Experiment, final_time: float = 0):
#     """
#     This function emulates multiple dispatch to allow for many model and experiment types
#     TODO: maybe make this read from a mutable dictionary so that types can be added outside
#     Biolucia
#     """
#     if isinstance(model, Model) and isinstance(experiment, InitialValueExperiment):
#         return simulate_initial_value_system(model, experiment, final_time)
#     elif isinstance(model, Model) and isinstance(experiment, SteadyStateExperiment):
#         return simulate_steady_state_system(model, experiment, final_time)
#     else:
#         raise TypeError(
#             'No simulation engine found for model of type "{m}" and experiment of type "{con}"'
#             .format(m=type(model), con=type(experiment))
#         )
#
#
