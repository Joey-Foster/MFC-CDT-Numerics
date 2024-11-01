import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from NLSWE_specialisedIC import plottingSetup, justPlotTheseTimesteps, doEvolution


#%% Function Definitions

# Initial data
def h0(x):
    """
    Initial datum for the height
    """
    return 1 + np.exp(-5*x**2)
    #return 1 + 0.5* np.power(np.cos(np.pi * x),2)

def u0(x):
    """
    Initial datum for the velocity
    """
    return 0*x

def processInitialData(h0, u0, nx, g, H, mu, t_end=False):
    """
    Processes the initial data for the simulation:
        setting up the spatial grid, initial conditions, timestep and
        mapping into characteristic variables

    Parameters:
    h0 (function): Function defining the initial water height distribution.
    u0 (function): Function defining the initial velocity distribution.
    nx (int): Number of spatial grid points.
    g (float): Gravitational constant.
    H (float): Mean height of the water distribution.
    t_end (float, optional): End time for the simulation. 
        If provided, the number of timesteps is calculated to reach this time.

    Returns:
    tuple: A tuple containing the previous and current timestep arrays for characteristic
           variables Q1 and Q2, the current arrays for the height and velocity,
           the spatial grid, grid spacing, timestep, and optionally the number of timesteps.
    """
    
    # Define space discretisation
    dx = 1/nx
    x = np.linspace(-1, 1, nx+1)
    
    # Define initial data
    h = h0(x)
    u = u0(x)
    
    # Map to characteristic variables
    Q1 = 0.5 * (u + np.sqrt(g*h))
    Q2 = 0.5 * (-u + np.sqrt(g*h))

    # Set up previous-timestep arrays
    Q1Old = Q1.copy()
    Q2Old = Q2.copy()

    # Obey the CFL condition (derived from linearised SWE)
    stableWaveSpeed = np.sqrt(g*H)
    dt = mu * dx / stableWaveSpeed 
     
    if t_end:
        # Ensure number of timesteps is reachable
        nt = int(np.ceil(t_end/dt))
    
        return Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt
    else: 
        return Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt

def produceStaticPlot(h0, h_lims, u_lims, nx, g, H, mu, t_simulation_range, t_plotting_range, t_sample, suptitle, filename):
    """
    Produces a static plot of the simulation results over a specified time range.

    Parameters:
    h0 (function): Function defining the initial water height distribution.
    h_lims (list): The limits for the water height plot [min, max].
    u_lims (list): The limits for the velocity plot [min, max].
    nx (int): Number of spatial grid points.
    g (float): Gravitational constant.
    H (float): Mean height of the water distribution.
    mu (float): The desired courant number.
    t_simulation_range (int): The total number of timesteps for the simulation.
    t_plotting_range (tuple): The range of timesteps to plot (start, end).
    t_sample (int): The number of timesteps to sample for plotting.
    suptitle (str): The title for the entire figure.
    filename (str): The filename to save the plot as a PDF.

    Returns:
    None
    """
    current_time=0

    fig, ax = plottingSetup(h_lims, u_lims)
    
    fig.suptitle(suptitle)
    
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt = processInitialData(h0, u0, nx, g, H, mu)
    
    # uOld and hOld are carry-overs from the doEvolution parent class but are
    # not used in this implementation, so they are just defined here as a 
    # placeholder string. There is definitely a more elegant way to deal with
    # this, but this works for now. If I have time, I will come back and try to
    # make this nicer.
    uOld = hOld = 'placeholder'
    
    for t in range(t_simulation_range + 1):
        justPlotTheseTimesteps(t_plotting_range, t_sample, t, current_time, ax, x, h, u)
        evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, dt, 
                                             t, current_time, g, H, nx, Q1, Q2, Q1Old, Q2Old)
        Q1Old, Q2Old, Q1, Q2, h, u, dt, t, current_time = evolution.timestep('')
    fig.legend(loc='upper center', bbox_to_anchor=(0.525, 0.95), ncol=t_sample, frameon=False)
    plt.savefig(f'{filename}.pdf')
    plt.show()


def GIFtime():
    """
    Produces a GIF of the simulation results up to a specified final time.
    
    It uses most of the same parameters as produceStaticPlot(), except 
    t_simulation_range, t_plotting_range and t_sample, and additionally 
    depends on t_end. I didn't have time to clean this up but it does work
    if you run it locally in this file.
    """
    t=0
    current_time=0

    # Set up axes and initial data
    fig, line_h, line_u = plottingSetup([0.9,2], [-1,1], GIF=True)
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt = processInitialData(h0, u0, nx, g, H, mu, t_end)
   
    suptitle = fig.suptitle(r'Non-linear 1-D SWE with IC $u_0 = 0$')
   
    print('Rendering GIF. Please stand by.')
   
    uOld = hOld = 'placeholder'
   
    evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, dt, t, current_time,
                                          g, H, nx, Q1, Q2, Q1Old, Q2Old, nt, 
                                          t_end, line_h, line_u, suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
    ani.save('characteristic_WTF.gif', writer='pillow', fps=20)


#%% Time stepping

class doCharacteristicEvolution(doEvolution):
    """
    Class responsible for the timestepping and handlling all the variables of 
    the FTBFS characteristic-based finite difference implementation.
    
    Due to the inheritance of doEvolution being.... not the best, some functions
    in this file have had to be duplicated. See the docstring for the doEvolution
    class in NLSWE_specialsedIC.py for my reflections on how this was implemented.
    """
    
    def __init__(self, hOld, uOld, h, u, x, dx, dt, t, current_time, g, H, nx, 
                 Q1, Q2, Q1Old, Q2Old, nt=False, t_end=False, line_h=False,
                 line_u=False, suptitle=False):
        """
        Initialise all the variables needed for the class. Most variables are
        inherited. See docstring for the parent class.
        
        New Parameters:
        H (float): Mean height of the water distribution.
        nx (int): Number of spatial grid points.
        Q1 (np.ndarray): Current timestep array for the Q1 variable
        Q2 (np.ndarray): Current timestep array for the Q2 variable
        Q1Old (np.ndarray): Previous timestep array for the Q1 variable
        Q2Old (np.ndarray): Previous timestep array for the Q2 variable
        """
        super().__init__(hOld, uOld, h, u, x, dx, dt, t, current_time, g, nt = nt, 
                         t_end = t_end, line_h = line_h, line_u = line_u, suptitle = suptitle)
        self.H = H
        self.nx = nx
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q1Old = Q1Old
        self.Q2Old = Q2Old
        

    def timestep(self, frame):
        """
        Evolves the simulation by one timestep
        
        Parmeters:
        frame (int): An implict frame counter that is only used by FuncAnimation()
        inside GIFtime()
        
        Returns:
        tuple: A tuple containing the previous and current timestep arrays for 
               characteristic variables Q1 and Q2, current timestep arrays for
               height and velocity, the timestep, current number of timesteps taken,
               current time and optionally, the total number of timesteps for 
               the whole simulation.
        """
        # FTBS for Q1 and FTFS for Q2
        self.Q1[1:-1] = self.Q1Old[1:-1] - 2*self.dt/self.dx * self.Q1Old[1:-1] * (self.Q1Old[1:-1] - self.Q1Old[:-2])
        self.Q2[1:-1] = self.Q2Old[1:-1] + 2*self.dt/self.dx * self.Q2Old[1:-1] * (self.Q2Old[2:] - self.Q2Old[1:-1])
        
        # Manually update the missing boundary point 
        self.Q1[-1] = self.Q1Old[-1] - 2*self.dt/self.dx * self.Q1Old[-1] * (self.Q1Old[-1] - self.Q1Old[-2])
        self.Q2[0] = self.Q2Old[0] + 2*self.dt/self.dx * self.Q2Old[0] * (self.Q2Old[1] - self.Q2Old[0])
        
        # Apply PBCs
        self.Q1[0] = self.Q1[-1] 
        self.Q2[-1] = self.Q2[0]
        
        # Update previous timestep
        self.Q1Old = self.Q1.copy()
        self.Q2Old = self.Q2.copy()
    
        # Map back to coordinate variables for plotting
        self.h = 1/self.g * (self.Q1 + self.Q2)**2
        self.u = self.Q1 - self.Q2
        
        # Set axis data
        if bool(self.line_h) & bool(self.line_u):
            self.line_h.set_data(self.x, self.h)
            self.line_u.set_data(self.x, self.u)
            
        if bool(self.suptitle):
            self.suptitle.set_text(fr'Non-linear 1-D SWE with IC $u_0 = 0$'
                               f'\n Time = {self.current_time:.3f}')
        print(f't={self.t}, dt={self.dt:.5f}, '
              f'current_time={self.current_time:.3f}, ' 
              f'CFL={np.sqrt(self.g*self.H)*self.dt/self.dx:.2f}')
        
        if bool(self.t_end) & bool(self.nt):
            self.timeOvershootChecker(self.t_end)
            self.current_time += self.dt
            self.t += 1
            return self.Q1Old, self.Q2Old, self.Q1, self.Q2, self.h, self.u, self.dt, self.nt, self.t, self.current_time
        else:
            self.current_time += self.dt
            return self.Q1Old, self.Q2Old, self.Q1, self.Q2, self.h, self.u, self.dt, self.t, self.current_time

#%% Params

if __name__ == '__main__':
    g = 9.81     # Gravitational constant [ms^-2]
    H = 1.4      # Mean height distribution [m]
    nx = 100     # Number of spatial grid points
    t_end = 1  # End point of the simulation runtime [s]
    mu = 0.888    # Desired courant number 

    GIFtime()
    # produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [-1,1], nx = nx, g = g, H = H, mu = mu, 
    #                   t_end = t_end, t_simulation_range = 200, 
    #                   t_plotting_range = [180,200], t_sample = 4, 
    #                   suptitle = r'Non-linear 1-D SWE with $h_0 = 1 + e^{-5x^2}$, $u_0 = 0$', 
    #                   filename = 'NLSWE_arbitraryIC')