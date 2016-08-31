from biolucia.model import Model

from biolucia.simulation import BioluciaSystemSimulation


class Experiment:
    """Abstract base class for experimental conditions, which when combined with a model make various simulations
    available"""
    def _simulate(self, model, final_time=0.0):
        raise NotImplementedError()

    # def _simulate_variance(self, model, final_time=0.0, method='mfk'):
    #     raise NotImplementedError()

    # def _simulate_stochastic(self, model, final_time=0.0):
    #     raise NotImplementedError()


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
