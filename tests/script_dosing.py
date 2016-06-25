from biolucia.parser import *
import matplotlib.pyplot as plt
import numpy as np

filename = '../models/dosing.txt'
m = read_model(filename)
sim = m.simulate()

times = np.linspace(0, 10, 1000)
values = sim.system_values(times)
lines = plt.plot(times, values)
plt.show()
