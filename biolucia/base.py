class Experiment:
    def _simulate(self, model, final_time=0.0):
        raise NotImplementedError

    # def _simulate_variance(self, model, final_time=0.0, method='mfk'):
    #     raise NotImplementedError

    # def _simulate_stochastic(self, model, final_time=0.0):
    #     raise NotImplementedError


class Simulation:
    """Abstract base class for simulations, which lazily produce values for all components at all 
    time points 0 to infinity"""
    pass
