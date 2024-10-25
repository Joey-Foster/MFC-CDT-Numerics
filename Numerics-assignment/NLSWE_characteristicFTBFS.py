import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from NLSWE_specialisedIC import plottingSetup ,doEvolution


#%% Function Definitions

# Initial data
def h0(x):
    return 1 + np.exp(-10*x**2)

def u0(x):
    return 0*x

def processInitialData(h0, u0, nx, t_end, g, H):
    
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
    dt =  0.99 * dx / (2 * stableWaveSpeed) 
     
    # Ensure number of timesteps is reachable
    nt = int(np.ceil(t_end/dt))
    
    return Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt

def produceStaticPlot(h_lims, u_lims, t_range, t_sample):
    if t_sample > 4:
        raise Exception("t_sample must be between 1 and 4 because I didn't have time to generalise this")
    t=0
    current_time=0

    fig, ax = plottingSetup(h_lims, u_lims)
    
    fig.suptitle('Non-linear 1-D SWE with arbitrary smooth IC')
    
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt = processInitialData(h0, u0, nx, t_end, g, H)
    
    uOld = hOld = 'placeholder'
    
    for t in range(t_range + 1):
        colours = ['b','r','g','orange'] #This does not support t_sample > 4...
        quotient, remainder = divmod(t,t_range//(t_sample-1))
        if remainder==0:
            ax[0].plot(x,h,c=colours[quotient-1],label=f't = {current_time:.3f}')
            ax[1].plot(x,u,c=colours[quotient-1])
        evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, nx, dt, nt, 
                                             t, current_time, Q1, Q2, Q1Old, Q2Old)
        Q1Old, Q2Old, Q1, Q2, h, u, dt, nt, t, current_time = evolution.timestep('')
    fig.legend(loc='upper center', bbox_to_anchor=(0.525, 0.95), ncol=t_sample, frameon=False)
    plt.savefig('NLSWE_arbitrarysmoothIC.pdf')
    plt.show()


def GIFtime():
    t=0
    current_time=0

    # Set up axes and initial data
    fig, line_h, line_u = plottingSetup([0.9,2], [-1,1], GIF=True)
    Q1Old, Q2Old, Q1, Q2, h, u, x, dx, dt, nt = processInitialData(h0, u0, nx, 
                                                                  t_end, g, H)
   
    suptitle = fig.suptitle('Non-linear 1-D SWE with arbitrary smooth IC')
   
    print('Rendering GIF. Please stand by.')
   
    # uOld and hOld are carry-overs from the doEvolution parent class but are
    # not used in this implementation, so they are just defined here as a 
    # placeholder string. There is definitely a more elegant way to deal with
    # this, but this works for now. If I have time, I will come back and try to
    # make this nicer.
    uOld = hOld = 'placeholder'
   
    evolution = doCharacteristicEvolution(hOld, uOld, h, u, x, dx, nx, dt, nt, 
                                         t, current_time, Q1, Q2, Q1Old, Q2Old, line_h, line_u, 
                                         suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
    ani.save('FTBS_NLSWE_arbitrarysmoothIC.gif', writer='pillow', fps=20)


#%% Time stepping

class doCharacteristicEvolution(doEvolution):
    
    def __init__(self, hOld, uOld, h, u, x, dx, nx, dt, nt, t, current_time, Q1, Q2, Q1Old, Q2Old,
                 line_h=0, line_u=0, suptitle=''):
        super().__init__(hOld, uOld, h, u, x, dx, dt, nt, t, current_time, 
                     line_h = line_h, line_u = line_u, suptitle = suptitle)
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
        self.h = 1/g * (self.Q1 + self.Q2)**2
        self.u = self.Q1 - self.Q2
        
        # Set axis data
        if (self.line_h != 0) & (self.line_u != 0):
            self.line_h.set_data(self.x, self.h)
            self.line_u.set_data(self.x, self.u)
            
        if self.suptitle != '':
            self.suptitle.set_text(f'Non-linear 1-D SWE with arbitrary smooth IC'
                               f'\n Time = {self.current_time:.3f}')
        print(f't={self.t}, dt={self.dt:.5f}, '
              f'current_time={self.current_time:.3f}, ' 
              f'CFL={2*np.sqrt(g*H)*self.dt/self.dx:.2f}')
        
        self.timeOvershootChecker(t_end)
        self.current_time += self.dt
        self.t += 1

        return self.Q1Old, self.Q2Old, self.Q1, self.Q2, self.h, self.u, self.dt, self.nt, self.t, self.current_time
#%% Params

if __name__ == '__main__':
    g = 9.81
    H = 1
    nx = 100
    t_end = 0.1

    GIFtime()
    #produceStaticPlot(h_lims = [0.9,2], u_lims = [-1,1], t_range = 50, t_sample = 4)