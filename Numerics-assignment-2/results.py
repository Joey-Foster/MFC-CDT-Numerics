import numpy as np
import matplotlib.pyplot as plt

#Make the plot text and aspect ratio look nice
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 
# pts-to-inches conversion * #pts in width of latex doc with 1in margins
fig_width_inches = 1 / 72.27 * 443.57848
fig_height_inches = fig_width_inches * ( -1 + np.sqrt(5) ) / 2
figsize = (fig_width_inches, fig_height_inches)

# Load in example mesh figure
plt.figure(figsize=figsize)

nodes = np.loadtxt('las_grids/las_nodes_10k.txt')
IEN = np.loadtxt('las_grids/las_IEN_10k.txt', dtype=np.int64)
boundary_nodes = np.loadtxt('las_grids/las_bdry_10k.txt', 
                            dtype=np.int64)

plt.triplot(nodes[:,0], nodes[:,1], triangles=IEN)
plt.plot(nodes[boundary_nodes, 0], nodes[boundary_nodes, 1], '.', c='orange')
plt.plot([442365],[115483],'x',c='r')
plt.plot([473993], [171625],'x',c='pink')
plt.axis('equal')
plt.savefig('example_mesh.pdf')
plt.show()

from TwoDimStaticAdvDiffFESolver import TwoDimStaticAdvDiffFESolver
from staticPollutionOverReading import S_sotonfire, pollutionExtractor, convergence

# global wind unit-vectors
north = np.array([0,1])

directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
directed_at_reading = (1/np.linalg.norm(directed_at_reading)
                       *directed_at_reading)
# Diffusion coefficient
D = 10000

# Reading coords
reading = np.array([473993, 171625])

# Max res north static plot
plt.figure(figsize=figsize)
max_res_data = TwoDimStaticAdvDiffFESolver(S_sotonfire, -10*north, D, '1_25')
nodes, IEN, southern_boarder, psi = max_res_data

plt.tripcolor(nodes[0,:], nodes[1,:], psi, triangles=IEN)
plt.plot([442365],[115483],'x',c='r')
plt.plot([473993], [171625],'x',c='pink')
plt.axis('equal')
plt.colorbar()
plt.plot(nodes[0, southern_boarder], nodes[1,southern_boarder], '.', c='orange')
plt.savefig('static_north.pdf')
plt.show()
print('North wind:')
print(f'Pollution over reading = {pollutionExtractor(psi, nodes, IEN, reading)}')

convergence(max_res_data, reading, -10*north, 10000, figsize, 
            'static_north_convergence_Ns', 'static_north_convergence_xs')

print('')
# Max res reading-directed static plot
plt.figure(figsize=figsize)
max_res_data = TwoDimStaticAdvDiffFESolver(S_sotonfire, -10*directed_at_reading,
                                           D, '1_25')
nodes, IEN, southern_boarder, psi = max_res_data

plt.tripcolor(nodes[0,:], nodes[1,:], psi, triangles=IEN)
plt.plot([442365],[115483],'x',c='r')
plt.plot([473993], [171625],'x',c='pink')
plt.axis('equal')
plt.colorbar()
plt.plot(nodes[0, southern_boarder], nodes[1,southern_boarder], '.', c='orange')
plt.savefig('static_directed_at_reading.pdf')
plt.show()
print('Reading wind:')
print(f'Pollution over reading = {pollutionExtractor(psi, nodes, IEN, reading)}')

convergence(max_res_data, reading, -10*directed_at_reading, 10000, figsize,
            'static_reading_convergence_Ns', 'static_reading_convergence_xs')


from timeEvolvedPollutionOverReading import pollutionTimeSeries

# Max runtime (secs)
t_max = 15000

# 10k res is the mas I can do without it taking ages
pollutionTimeSeries(t_max, -10*north, D, '10', reading, 
                    figsize=figsize, filename='timeseries_north')

pollutionTimeSeries(t_max, -10*directed_at_reading, D, '10', reading, 
                    figsize=figsize, filename='timeseries_reading')