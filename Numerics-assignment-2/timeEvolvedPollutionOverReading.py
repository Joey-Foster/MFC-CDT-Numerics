import numpy as np
import matplotlib.pyplot as plt
from TwoDimStaticAdvDiffFESolver import S_sotonfire
from TwoDimTimeEvolvedAdvDiffFESolver import TwoDimTimeEvolvedAdvDiffFESolver
from staticPollutionOverReading import pollutionExtractor

def doTimeEvolution(t_max, u, D, resolution):
    '''
    Plots one figure for every 100 timesteps - this may lag VScode, use spyder
    until I port this into a GIF maker
    '''
    nodes, IEN, southern_boarder, ts, ys = TwoDimTimeEvolvedAdvDiffFESolver(S_sotonfire, u, D, resolution, t_max)

    for i, t in enumerate(ts):
        if i % 100 == 0:
            plt.figure()
            plt.tripcolor(nodes[0,:], nodes[1,:], ys[:,i], triangles=IEN)
            plt.plot([442365],[115483],'x',c='r')
            plt.plot([473993], [171625],'x',c='pink')
            plt.axis('equal')
            plt.colorbar()
            plt.title(f'Time = {int(t)}')
            
            plt.plot(nodes[0, southern_boarder], nodes[1,southern_boarder], '.', c='orange')
        
        plt.show()
        
def pollutionTimeSeries(t_max, u, D, resolution, coords):
    
    nodes, IEN, southern_boarder, ts, ys = TwoDimTimeEvolvedAdvDiffFESolver(S_sotonfire, u, D, resolution, t_max)
    
    psi_at_reading = np.zeros_like(ts)
    for i in range(len(ts)):
        psi_at_reading[i] = pollutionExtractor(ys[:,i], nodes, IEN, coords)
        
    plt.plot(ts, psi_at_reading)
    plt.xlabel('t')
    plt.ylabel(r'$\psi(reading)$')
    plt.title(f'Time series of pollution concerntration over reading\n u = [{u[0]:.2f},{u[1]:.2f}], D = {D}')
    plt.show()
        
if __name__ == '__main__':
    
    directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
    directed_at_reading = 1/np.linalg.norm(directed_at_reading)*directed_at_reading
    
    #doTimeEvolution(15000, 10*directed_at_reading, 10000, '10')
    
    reading = np.array([473993, 171625])
    pollutionTimeSeries(15000, 10*directed_at_reading, 10000, '10', reading)