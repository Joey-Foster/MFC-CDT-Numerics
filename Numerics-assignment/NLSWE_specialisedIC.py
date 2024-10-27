import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Function Defintions

def plottingSetup(h_lims, u_lims, GIF=False):
    #Make the plot aspect ratios and text look nice
    params = {'text.usetex' : True,
              'font.size' : 11,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params) 
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
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.85)
    
    if GIF:
        return fig, line_h, line_u
    else:
        return fig, ax

# Initial data
def h0(x):
    return 1 + np.exp(-5*x**2)

def u0(x,U0):
    return U0*np.ones_like(x)

def processInitialData(h0, u0, U0, nx, t_end, g):
    
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
     
    # Ensure number of timesteps is reachable
    nt = int(np.ceil(t_end/dt))
    
    return hOld, uOld, h, u, x, dx, dt, nt

def justPlotTheseTimesteps(t_simulation_range, t_plotting_range, t_sample, t, current_time, ax, x, h, u):
    # Obviously there are plenty of other exceptions I could catch such as any 
    # of the t_range's being a non-integer, or t_plotting_range[1] > t_simulation_range
    # but to an expert in the field reading this code, those restrictions should
    # be obvious. The only non-obvious limitation is the number of sample points
    # being limited to at most 4, so it deserves a raise Exception.
    if t_sample > 4:
        raise Exception("t_sample must be between 1 and 4 because I didn't have time to generalise this")
        
    colours = ['b','r','g','orange'] #This does not support t_sample > 4...
    if t_plotting_range[0] <= t <= t_plotting_range[1]:
        quotient, remainder = divmod(t - t_plotting_range[0], (t_plotting_range[1] - t_plotting_range[0]) // (t_sample - 1))
        if remainder==0:
            ax[0].plot(x,h,c=colours[quotient],label=f't = {current_time:.3f}')
            ax[1].plot(x,u,c=colours[quotient])

def produceStaticPlot(h0, h_lims, u_lims, U0, nx, g, t_end, t_simulation_range, t_plotting_range, t_sample, suptitle, filename):
    current_time=0

    fig, ax = plottingSetup(h_lims, u_lims)
    
    fig.suptitle(suptitle)
    
    hOld, uOld, h, u, x, dx, dt, nt = processInitialData(h0, u0, U0, nx, t_end, g)
    
    for t in range(t_simulation_range + 1):
        justPlotTheseTimesteps(t_simulation_range, t_plotting_range, t_sample, t, current_time, ax, x, h, u)
        evolution = doEvolution(hOld, uOld, h, u, x, dx, dt, nt, t, current_time, t_end, g)
        hOld, uOld, h, u, dt, nt, t, current_time = evolution.timestep('')
    fig.legend(loc='upper center', bbox_to_anchor=(0.525, 0.95), ncol=t_sample, frameon=False)
    plt.savefig(f'{filename}.pdf')
    plt.show()

def GIFtime():
    t=0
    current_time=0

    # Set up axes and initial data
    fig, line_h, line_u = plottingSetup([0.9,2], [0.5*U0,1.5*U0], GIF=True)
    hOld, uOld, h, u, x, dx, dt, nt = processInitialData(h0, u0, U0, nx, t_end, g)
    
    suptitle = fig.suptitle(fr'Non-linear 1-D SWE with specialised IC $u_0$ = {U0}')
   
    print('Rendering GIF. Please stand by.')
   
    evolution = doEvolution(hOld, uOld, h, u, x, dx, dt, nt, t, current_time, 
                           line_h, line_u, suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
    ani.save('FTBS_NLSWE_specialIC.gif', writer='pillow', fps=20)

#%% Time stepping


class doEvolution:
    
    def __init__(self, hOld, uOld, h, u, x, dx, dt, nt, t, current_time, t_end, g, 
                 line_h=0, line_u=0, suptitle=''):
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
        # If current_time happens to exactly reach t_end, stop the evolution 
        # and do not update nt else a division by zero error would occur.
        # Else, if the next timestep is about to overshoot t_end, update dt and
        # nt to exactly stop at the right time.
        if self.current_time == t_end:
            self.dt = t_end - self.current_time
        elif self.current_time + self.dt > t_end:
            self.dt = t_end - self.current_time
            self.nt = int(np.ceil(t_end/self.dt))
            
    
    ## Using Durran's NL-SWE formualtion
    def timestep(self, frame):
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
        if (self.line_h != 0) & (self.line_u != 0):
            self.line_h.set_data(self.x, self.h)
            self.line_u.set_data(self.x, self.u)
            
        if self.suptitle !='':
            self.suptitle.set_text(fr'Non-linear 1-D SWE with specialised IC $u_0$ = {U0}\n Time = {self.current_time:.3f}')
        print(f't={self.t}, dt={self.dt:.5f}, '
              f'current_time={self.current_time:.3f}, ' 
              f'CFL={max(abs(self.u))*self.dt/self.dx:.2f}')
        
        self.timeOvershootChecker(self.t_end)
        self.current_time += self.dt
        self.t += 1
  
        return self.hOld, self.uOld, self.h, self.u, self.dt, self.nt, self.t, self.current_time,
#%% Params 

if __name__ == '__main__':
    g = 9.81
    nx = 100
    t_end = 0.25
    U0 = 10
    
    #GIFtime()
    produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [0.5*U0,1.5*U0], U0=U0, nx=nx, g=g, t_end=t_end, t_simulation_range = 250, t_plotting_range =[180, 200] , t_sample = 4, suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + e^{{-5x^2}}$, $u_0$ = {U0}',  filename='NLSWE_specialisedIC')