import numpy as np
import matplotlib.pyplot as plt
from TwoDimTimeEvolvedAdvDiffFESolver import TwoDimTimeEvolvedAdvDiffFESolver
from staticPollutionOverReading import pollutionExtractor, S_sotonfire

def doTimeEvolution(t_max, u, D, resolution):
    '''
    Plots one figure for each of the 201 timesteps - this may lag VScode, use spyder
    until I port this into a GIF maker
    '''
    nodes, IEN, southern_boarder, ts, ys = TwoDimTimeEvolvedAdvDiffFESolver(S_sotonfire, 
                                                                            u, D, resolution, 
                                                                            t_max)

    for i, t in enumerate(ts):
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
    
    nodes, IEN, southern_boarder, ts, ys = TwoDimTimeEvolvedAdvDiffFESolver(S_sotonfire, 
                                                                            u, D, resolution, 
                                                                            t_max)
    
    psi_at_reading = np.zeros_like(ts)
    for i in range(len(ts)):
        psi_at_reading[i] = pollutionExtractor(ys[:,i], nodes, IEN, coords)
        
    plt.plot(ts, psi_at_reading)
    plt.xlabel('t')
    plt.ylabel(r'$\psi(Reading)$')
    plt.title('Time series of pollution concerntration over Reading\n'
              f'u = [{abs(u[0]):.2f},{abs(u[1]):.2f}], D = {D}')
    plt.grid()
    plt.show()
    
    
    return ts, psi_at_reading
        
def convergence(t_max, u, D, coords):
    
    ts, soln_N = pollutionTimeSeries(t_max, u, D, '20', coords)
    _, soln_2N = pollutionTimeSeries(t_max, u, D, '10', coords)
    __, soln_4N = pollutionTimeSeries(t_max, u, D, '5', coords)
    
   # indicies = np.arange(len(ts))
    
    y_2N_N = np.linalg.norm(soln_2N - soln_N, 2)
    y_4N_2N = np.linalg.norm(soln_4N - soln_2N, 2)
    
    s = np.log2(y_2N_N/y_4N_2N)
    return y_2N_N, y_4N_2N, s
    
if __name__ == '__main__':
    
    directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
    directed_at_reading = 1/np.linalg.norm(directed_at_reading)*directed_at_reading
    
    doTimeEvolution(15000, -10*directed_at_reading, 1000, '10')
   
    reading = np.array([473993, 171625])
   # print(convergence(15000, -10*directed_at_reading, 1000, reading))
    
    
    #pollutionTimeSeries(15000, -10*directed_at_reading, 1000, '10', reading)