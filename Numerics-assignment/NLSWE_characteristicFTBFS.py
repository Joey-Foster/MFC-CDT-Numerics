import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from NLSWE_specialisedIC import plottingSetup, justPlotTheseTimesteps, doEvolution


#%% Function Definitions

# Initial data
def h0(x):
    return 1 + np.exp(-5*x**2)
    #return 1 + 0.5* np.power(np.cos(np.pi * x),2)

def u0(x):
    return 0*x

def processInitialData(h0, u0, nx, g, H, t_end=False):
    
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
    dt =  0.85 * dx / stableWaveSpeed 
     
    if t_end:
        # Ensure number of timesteps is reachable
        nt = int(np.ceil(t_end/dt))
    
        return Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt
    else: 
        return Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt

def produceStaticPlot(h0, h_lims, u_lims, nx, g, H, t_simulation_range, t_plotting_range, t_sample, suptitle, filename):
    current_time=0

    fig, ax = plottingSetup(h_lims, u_lims)
    
    fig.suptitle(suptitle)
    
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt = processInitialData(h0, u0, nx, g, H)
    
    # uOld and hOld are carry-overs from the doEvolution parent class but are
    # not used in this implementation, so they are just defined here as a 
    # placeholder string. There is definitely a more elegant way to deal with
    # this, but this works for now. If I have time, I will come back and try to
    # make this nicer.
    uOld = hOld = 'placeholder'
    
    for t in range(t_simulation_range + 1):
        justPlotTheseTimesteps(t_simulation_range, t_plotting_range, t_sample, t, current_time, ax, x, h, u)
        evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, dt, 
                                             t, current_time, g, H, nx, Q1, Q2, Q1Old, Q2Old)
        Q1Old, Q2Old, Q1, Q2, h, u, dt, t, current_time = evolution.timestep('')
    fig.legend(loc='upper center', bbox_to_anchor=(0.525, 0.95), ncol=t_sample, frameon=False)
    plt.savefig(f'{filename}.pdf')
    plt.show()


def GIFtime():
    t=0
    current_time=0

    # Set up axes and initial data
    fig, line_h, line_u = plottingSetup([0.9,2], [-1,1], GIF=True)
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt = processInitialData(h0, u0, nx, g, H, t_end)
   
    suptitle = fig.suptitle('Non-linear 1-D SWE with arbitrary smooth IC')
   
    print('Rendering GIF. Please stand by.')
   
    
    uOld = hOld = 'placeholder'
   
    evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, dt, 
                                         t, current_time, g, H, nx, Q1, Q2, Q1Old, Q2Old, nt, t_end, line_h, line_u, suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
    ani.save('FTBFS_NLSWE_arbitrarysmoothIC.gif', writer='pillow', fps=20)


#%% Time stepping

class doCharacteristicEvolution(doEvolution):
    
    def __init__(self, hOld, uOld, h, u, x, dx, dt, t, current_time, g, H, nx, Q1, Q2, Q1Old, Q2Old, nt=False, t_end=False, line_h=False, line_u=False, suptitle=False):
        super().__init__(hOld, uOld, h, u, x, dx, dt, t, current_time, g, 
                     nt = nt, t_end = t_end, line_h = line_h, line_u = line_u, suptitle = suptitle)
        self.H = H
        self.nx = nx
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q1Old = Q1Old
        self.Q2Old = Q2Old
        

    def timestep(self, frame):
        
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
            self.suptitle.set_text(f'Non-linear 1-D SWE with arbitrary smooth IC'
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
    g = 9.81
    H = 1.4
    nx = 100
    t_end = 0.5

    GIFtime()
    #produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [-1,1], nx = nx, g = g, H = H, t_end = t_end, t_simulation_range = 200, t_plotting_range = [180,200], t_sample = 4, suptitle = r'Non-linear 1-D SWE with $h_0 = 1 + e^{-5x^2}$, $u_0 = 0$', filename = 'NLSWE_arbitraryIC')