import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Function Defintions

def plottingSetup(h_lims, u_lims, GIF=False):
    """
    Sets up the plotting environment for the simulation.

    Parameters:
    h_lims (list): The range for the water height plot [min, max].
    u_lims (list): The range for the velocity plot [min, max].
    GIF (bool): Flag to indicate if the setup is for GIF creation. Default is False.

    Returns:
    tuple: A tuple containing the figure object and line objects for height and velocity.
           or the figure and axes objects, depending on the value of GIF
    """
    
    #Make the plot text and aspect ratio look nice
    params = {'text.usetex' : True,
              'font.size' : 11,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params) 
    # pts-to-inches conversion * #pts in width of latex doc with 1in margins
    fig_width_inches = 1 / 72.27 * 443.57848
    fig_height_inches = fig_width_inches * ( -1 + np.sqrt(5) ) / 2
    
    # Set up empty axes with appropriate labels
    fig, ax = plt.subplots(1,2,figsize=(fig_width_inches, fig_height_inches))
    if GIF:
        line_h, = ax[0].plot([], [],'b', label='h')
        line_u, = ax[1].plot([], [],'r', label='u')
        
    ax[0].set_ylabel('$h$')
    ax[1].set_ylabel('$u$')
    ax[0].set_ylim(h_lims)
    ax[1].set_ylim(u_lims)
    for a in ax:
        if GIF:
            a.legend(loc='upper left')
        a.set_xlabel('$x$')
        a.set_xlim([-1,1])
        a.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    if GIF:
        return fig, line_h, line_u
    else:
        return fig, ax

def h0(x):
    """
    Initial datum for the height
    """
    return 1 + np.exp(-5*x**2)
    #return 1 + 0.5 * np.power(np.cos(np.pi*x),2)

def u0(x, U0):
    """
    Initial datum for the velocity
    """
    return U0*np.ones_like(x)

def processInitialData(h0, u0, U0, nx, g, t_end=False):
    """
    Processes the initial data for the simulation:
        setting up the spatial grid, initial conditions, and timestep.

    Parameters:
    h0 (function): Function defining the initial water height distribution.
    u0 (function): Function defining the initial velocity distribution.
    U0 (float): Initial velocity parameter.
    nx (int): Number of spatial grid points.
    g (float): Gravitational constant.
    t_end (float, optional): End time for the simulation. 
        If provided, the number of timesteps is calculated to reach this time.

    Returns:
    tuple: A tuple containing the previous and current timestep arrays for height and velocity,
          the spatial grid, grid spacing, timestep, and optionally the number of timesteps.
    """
    
    # Define space discretisation
    dx = 1/nx
    x = np.linspace(-1, 1, nx+1)
    
    # Define initial data and previous-timestep arrays
    h = h0(x)
    u = u0(x, U0)
    hOld = h.copy()
    uOld = u.copy()

    # Obey intial CFL condition 
    intialStableWaveSpeed = np.max(uOld) + np.sqrt(np.max(g*hOld))
    dt = 0.99 * dx / intialStableWaveSpeed
     
    # Using the truthiness of t_end
    if t_end:
        # Ensure number of timesteps is reachable
        nt = int(np.ceil(t_end/dt))
        return hOld, uOld, h, u, x, dx, dt, nt
    else:
        return hOld, uOld, h, u, x, dx, dt

def justPlotTheseTimesteps(t_plotting_range, t_sample, t, current_time, ax, x, h, u):
    """
    Plots the simulation data at specific timesteps within the plotting range.
    
    Regarding the raise Exception: 
    Obviously there are plenty of other exceptions I could catch such as any 
    of the t_range's being a non-integer, or t_plotting_range[1] > t_simulation_range
    but to an expert in the field reading this code, those restrictions should
    be obvious. The only non-obvious limitation is the number of sample points
    being limited to at most 4, so it deserves a raise Exception.

    Parameters:
    t_plotting_range (tuple): The range of timesteps to plot (start, end).
    t_sample (int): The number of timesteps to sample for plotting.
    t (int): The current timestep.
    current_time (float): The current simulation time.
    ax (list): List of axes objects for plotting.
    x (np.ndarray): The spatial grid points.
    h (np.ndarray): The water height data.
    u (np.ndarray): The velocity data.

    Returns:
    None
    """
    if t_sample > 4:
        raise Exception("t_sample must be between 1 and 4 because I didn't have time to generalise this")
        
    colours = ['b','r','g','orange'] #This does not support t_sample > 4...
    if t_plotting_range[0] <= t <= t_plotting_range[1]:
        quotient, remainder = divmod(t - t_plotting_range[0], (t_plotting_range[1]
                                       - t_plotting_range[0]) // (t_sample - 1))
        if remainder==0:
            ax[0].plot(x,h,c=colours[quotient],label=f't = {current_time:.3f}')
            ax[1].plot(x,u,c=colours[quotient])

def produceStaticPlot(h0, h_lims, u_lims, U0, nx, g, t_simulation_range, 
                      t_plotting_range, t_sample, suptitle, filename):
    """
    Produces a static plot of the simulation results over a specified time range.

    Parameters:
    h0 (function): Function defining the initial water height distribution.
    h_lims (list): The limits for the water height plot [min, max].
    u_lims (list): The limits for the velocity plot [min, max].
    U0 (float): Initial velocity parameter.
    nx (int): Number of spatial grid points.
    g (float): Gravitational constant.
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
    
    hOld, uOld, h, u, x, dx, dt = processInitialData(h0, u0, U0, nx, g)
    
    for t in range(t_simulation_range + 1):
        justPlotTheseTimesteps(t_plotting_range, t_sample, t, current_time, ax, x, h, u)
        evolution = doEvolution(hOld, uOld, h, u, x, dx, dt, t, current_time, g)
        hOld, uOld, h, u, dt, t, current_time = evolution.timestep('')
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
    fig, line_h, line_u = plottingSetup([0.9,2], [3,6], GIF=True)
    hOld, uOld, h, u, x, dx, dt, nt = processInitialData(h0, u0, U0, nx, g, t_end)
    
    suptitle = fig.suptitle(fr'Non-linear 1-D SWE with specialised IC $u_0$ = {U0}')
   
    print('Rendering GIF. Please stand by.')
   
    evolution = doEvolution(hOld, uOld, h, u, x, dx, dt, t, current_time, g, nt, 
                            t_end, line_h, line_u, suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
    ani.save('cheating_attempt_unstable.gif', writer='pillow', fps=20)

#%% Time stepping


class doEvolution:
    """
    Class responsible for the timestepping and handlling all the variables of 
    the FTBS finite difference implementation.
    
    Note: This is not really written in the best way. If I were to start from 
    strach, I would implement a more basic doEvolution class that did not assume
    any particular parameters about the FD scheme, and then have both this file
    and the other one inherit from that parent class.
    As it stands, this is the parent class so when the other file inherits it,
    it picks up unnecssary variables, namely uOld and hOld, that it does not use
    in that implementation. 
    This was my first time implementing a non-trivial class structure and by the
    time I realised what a mess I had made, it was too late to refactor... :(
    """
    def __init__(self, hOld, uOld, h, u, x, dx, dt, t, current_time, g, nt = False, t_end=False,
                 line_h=False, line_u=False, suptitle=False):
        """
        Initialise all the variables needed for the class.
        
        Note: The optional arguments used here are a dirty hack to avoid produceStaticPlot()
        requiring a dependency on them when it doesn't actually use them in the 
        body of the function. They are only used when GIFtime() is called and I 
        am using the truthiness of their given values as a flag.
        
        Parameters:
        hOld (np.ndarray): Previous water height distribution.
        uOld (np.ndarray): Previous velocity distribution.
        h (np.ndarray): Current water height distribution.
        u (np.ndarray): Current velocity distribution.
        x (np.ndarray): Spatial grid points.
        dx (float): Grid spacing.
        dt (float): Timestep size.
        t (int): The current timestep.
        current_time (flaat): The current time of the simulation
        g (float): Gravitational constant.
        nt (int, optional): Number of timesteps.
        t_end (float, optional): The final runtime of the simulation.
        line_h (plt.line.lines2D, optional): Object for storting height data
        line_u (plt.line.lines2D, optional): Object for storting velocity data
        suptitle (str, optional): The title for the entire figure.
        
        Returns:
        None.
        """
        self.hOld = hOld
        self.uOld = uOld
        self.h = h
        self.u = u
        self.x = x
        self.dx = dx
        self.dt = dt
        self.nt = nt
        self.t = t
        self.current_time = current_time
        self.t_end = t_end
        self.g = g
        self.line_h = line_h
        self.line_u = line_u
        self.suptitle = suptitle
        
    def timeOvershootChecker(self, t_end):
        """
        Sets the correct value for dt and nt such that the simulation stops at
        exactly t_end.
        
            If current_time happens to exactly reach t_end, stop the evolution 
            and do not update nt else a division by zero error would occur.
            Else, if the next timestep is about to overshoot t_end, update dt and
            nt to exactly stop at the right time.
        
        Parameters:
        t_end (float): The final runtime of the simulation.
        
        Returns:
        None.
        """
        if self.current_time == t_end:
            self.dt = t_end - self.current_time
        elif self.current_time + self.dt > t_end:
            self.dt = t_end - self.current_time
            self.nt = int(np.ceil(t_end/self.dt))
            
    
    ## Using Durran's NL-SWE formualtion
    def timestep(self, frame):
        """
        Evolves the simulation by one timestep
        
        Parmeters:
        frame (int): An implict frame counter that is only used by FuncAnimation()
        inside GIFtime()
        
        Returns:
        tuple: A tuple containing the previous and current timestep arrays for 
               height and velocity, the current timestep, current
               number of timesteps taken, current time and optionally, the total
               number of timesteps for the whole simulation.
        """
        # Vectorised FTBS
        self.h[1:] = self.hOld[1:] - self.dt/self.dx * (self.uOld[1:] * 
                    (self.hOld[1:] - self.hOld[:-1]) + self.hOld[1:] * 
                    (self.uOld[1:] - self.uOld[:-1]))
        self.u[1:] = self.uOld[1:] - self.dt/self.dx * (self.uOld[1:] *
                    (self.uOld[1:] - self.uOld[:-1]) + self.g * 
                    (self.hOld[1:] - self.hOld[:-1]))
    
        # Apply PBCs
        self.h[0] = self.h[-1] 
        self.u[0] = self.u[-1]
    
        # Update previous timestep
        self.hOld = self.h.copy()
        self.uOld = self.u.copy()
        
        # Determine the new maximum stable wavespeed and update dt accordingly
        uTemp = abs(self.uOld) + np.sqrt(self.g*self.hOld)
        self.dt = min(self.dx / max(uTemp), self.dt)   
    
        # Set axis data
        if bool(self.line_h) & bool(self.line_u):
            self.line_h.set_data(self.x, self.h)
            self.line_u.set_data(self.x, self.u)
            
        if bool(self.suptitle):
            self.suptitle.set_text(f'Non-linear 1-D SWE with IC $u_0$ = {U0}'
                                   f'\n Time = {self.current_time:.3f}')
        print(f't={self.t}, dt={self.dt:.5f}, '
              f'current_time={self.current_time:.3f}, ' 
              f'CFL={max(abs(self.u))*self.dt/self.dx:.2f}')
        
        if bool(self.t_end) & bool(self.nt):
            self.timeOvershootChecker(self.t_end)
            self.current_time += self.dt
            self.t += 1

            return self.hOld, self.uOld, self.h, self.u, self.dt, self.nt, self.t, self.current_time
        else:
            self.current_time += self.dt
            return self.hOld, self.uOld, self.h, self.u, self.dt, self.t, self.current_time
#%% Params 

if __name__ == '__main__':
    g = 9.81     # Gravitational constant [ms^-2]
    nx = 100     # Number of spatial grid points
    t_end = 0.2  # End point of the simulation runtime [s]
    U0 = 4.3     # Initial velocity parameter [ms^-1]
    
    GIFtime()
    # produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [0.5*U0,1.5*U0], U0=U0, nx=nx, g=g, 
    #                   t_simulation_range = 250, t_plotting_range = [180, 200] , t_sample = 4, 
    #                   suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + e^{{-5x^2}}$, $u_0$ = {U0}',
    #                   filename='NLSWE_specialisedIC')