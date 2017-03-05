from biolucia.parser import *
import matplotlib.pyplot as plt
import numpy as np

filename = '../models/egfngf.txt'
m = read_model(filename)
sim = m.simulate()

times = np.linspace(0, 120, 1000)
values = sim.matrix_values(times)
lines = plt.plot(times, values)
plt.ylim((0, 2000000))
plt.show()
