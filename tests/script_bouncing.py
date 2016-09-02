from biolucia.parser import *
import matplotlib.pyplot as plt
import numpy as np

filename = '../models/bouncing.txt'
m = read_model(filename)
sim = m.simulate()

times = np.linspace(0, 100, 1000)
values = sim.matrix_values(times, 'x')
lines = plt.plot(times, values)
plt.show()
