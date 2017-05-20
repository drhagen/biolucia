from biolucia.parser import *
from biolucia.experiment import InitialValueExperiment
from biolucia.simulation import simulate
import matplotlib.pyplot as plt
import numpy as np

filename = '../models/discontinuity.txt'
m = read_model(filename)
con = InitialValueExperiment()
sim = simulate(m, con)

times = np.linspace(0, 10, 1000)
values = sim.matrix_values(times)
lines = plt.plot(times, values)
plt.show()
