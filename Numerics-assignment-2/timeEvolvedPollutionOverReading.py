import numpy as np
import matplotlib.pyplot as plt
from TwoDimTimeEvolvedAdvDiffFESolver import TwoDimTimeEvolvedAdvDiffFESolver
from staticPollutionOverReading import pollutionExtractor, S_sotonfire

def doTimeEvolution(t_max, u, D, resolution):
    """
    Plots the time evolution of the pollution distribution over the specified
    time period.

    Parameters:
    t_max (float): Maximum runtime of the simulation [s].
    u (array-like): Advection velocity vector [ms^-1].
    D (float): Diffusion coefficient [m^2s^-1].
    resolution (string): Grid resolution, one of ['1_25', '2_5', '5', '10', '20', '40'].

    Returns:
    None. 
        
    Note: I didn't have time (and/or bother) to implement GIF generation, so this
    just dumps 200 figures - use spyder to not crash your IDE
    """
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
        
def pollutionTimeSeries(t_max, u, D, resolution, coords, figsize=None, filename=None):
    """
    Plots the time series of pollution concentration at the specified coordinates
    over the given time period.

    Parameters:
    t_max (float): Maximum runtime of the simulation [s].
    u (array-like): Advection velocity vector [ms^-1].
    D (float): Diffusion coefficient [m^2s^-1].
    resolution (string): Grid resolution, one of ['1_25', '2_5', '5', '10', '20', '40'].
    coords (array-like): A 2-element array containing the coordinates where the
                         pollution value is to be extracted.
    figsize (tuple, optional): Figure size for the plot. Default is None.
    filename (str, optional): Filename for saving the plot. Default is None.

    Returns:
    psi_at_reading (np.ndarray): Array of pollution concentration values at the
                                 specified coordinates over time.
    """
    nodes, IEN, southern_boarder, ts, ys = TwoDimTimeEvolvedAdvDiffFESolver(S_sotonfire, 
                                                                            u, D, resolution, 
                                                                            t_max)
    # really this should say psi_at_coords if we're being completely general...
    psi_at_reading = np.zeros_like(ts)
    for i in range(len(ts)):
        psi_at_reading[i] = pollutionExtractor(ys[:,i], nodes, IEN, coords)
    
    if figsize != None:
        plt.figure(figsize=figsize)    
        plt.plot(ts, psi_at_reading)
        plt.xlabel('t')
        plt.ylabel(r'$\psi(Reading)$')
        plt.title('Time series of pollution concerntration over Reading\n'
                  f'u = [{abs(u[0]):.2f},{abs(u[1]):.2f}], D = {D}')
        plt.grid()
        if filename != None:
            plt.savefig(f'{filename}.pdf')
        plt.show()
        
    return psi_at_reading
        
def convergence(t_max, u, D, coords):
    """
    Analyzes the convergence of the time-evolved finite element solution for 
    different grid resolutions.

    Parameters:
    t_max (float): Maximum runtime of the simulation [s].
    u (array-like): Advection velocity vector [ms^-1].
    D (float): Diffusion coefficient [m^2s^-1].
    coords (array-like): A 2-element array containing the coordinates where the
                         pollution value is to be extracted.

    Returns:
    None.
    
    Note: this is hard-coded to use the 5,10,20 triple of resolutions because 
    the 40k one is useless and any higher than 5k would take hours.
    """
    
    soln_N = pollutionTimeSeries(t_max, u, D, '20', coords)
    soln_2N = pollutionTimeSeries(t_max, u, D, '10', coords)
    soln_4N = pollutionTimeSeries(t_max, u, D, '5', coords)
    
    y_2N_N = np.linalg.norm(soln_2N - soln_N, 2)
    y_4N_2N = np.linalg.norm(soln_4N - soln_2N, 2)
    
    s = np.log2(y_2N_N/y_4N_2N)
    
    abs_error = y_2N_N/(1-2**(-s))
    rel_error = abs_error/np.linalg.norm(soln_N,2) * 100
    
    textual_data = (f'theoretical convergence order = {s}\n'
                    f'theoretical absolute error = {abs_error}\n'
                    f'theoretical relative error = {rel_error}%')
    
    # save the data to a .txt file instead of just dumping to the console
    with open(f'time_dep_convergence_results_for_u=[{abs(u[0]):.2f}, {abs(u[1]):.2f}].txt',
              'w') as file:
        file.write(textual_data)

if __name__ == '__main__':
    
    directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
    directed_at_reading = 1/np.linalg.norm(directed_at_reading)*directed_at_reading
    
    # This will produce 200 figures - use spyder's embeded plot viewer!
    doTimeEvolution(15000, -10*directed_at_reading, 10000, '10')
