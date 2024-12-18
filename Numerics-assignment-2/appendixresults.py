import numpy as np
from timeEvolvedPollutionOverReading import convergence

# global wind unit-vectors
north = np.array([0,1])

directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
directed_at_reading = (1/np.linalg.norm(directed_at_reading)
                       *directed_at_reading)
# Diffusion coefficient
D = 10000

# Reading coords
reading = np.array([473993, 171625])

# Max runtime (secs)
t_max = 15000

convergence(t_max, -10*north, D, reading)
convergence(t_max, -10*directed_at_reading, D, reading)