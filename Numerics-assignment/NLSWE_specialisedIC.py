import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Function Defintions

def plottingSetup(h_lims, u_lims):
    #Make the plot aspect ratios and text look nice
    params = {'text.usetex' : True,
              'font.size' : 11,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params) 
    fig_width_inches = 1 / 72.27 * 443.57848
    fig_height_inches = fig_width_inches * ( -1 + np.sqrt(5) ) / 2
    
    fig, ax = plt.subplots(1,2,figsize=(fig_width_inches, fig_height_inches))
    line_h, = ax[0].plot([], [],'b', label='h')
    line_u, = ax[1].plot([], [],'r', label='u')
    
    ax[0].set_ylabel('$h$')
    ax[1].set_ylabel('$u$')
    ax[0].set_ylim(h_lims)
    ax[1].set_ylim(u_lims)
    for a in ax:
        a.legend(loc='upper left')
        a.set_xlabel('$x$')
        a.set_xlim([-1,1])
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.85)
    
    return fig, line_h, line_u

# Initial data
def h0(x):
    return 1+ np.exp(-5*x**2)

def u0(x):
    return 10*np.ones_like(x)

def processInitialData(h0, u0, nx, t_end, g):
    
    dx = 1/nx
    x = np.linspace(-1, 1, nx+1)
    
    h = h0(x)
    u = u0(x)
    hOld = h.copy()
    uOld = u.copy()

    intialStableWaveSpeed = np.max(uOld) + np.sqrt(np.max(g*hOld))
    dt =  0.99 *dx / intialStableWaveSpeed #CFL 
     
    nt = int(np.ceil(t_end/dt))
    
    return hOld, uOld, h, u, x, dx, dt, nt


def GIFtime():
   t=0
   current_time=0

   fig, line_h, line_u = plottingSetup([0.9,2], [7.5,12])
   hOld, uOld, h, u, x, dx, dt, nt = processInitialData(h0, u0, nx, t_end, g)
    
   suptitle = fig.suptitle('Non-linear 1-D SWE with specialised IC to make '
                           'FTBS work')
   
   print('Rendering GIF. Please stand by.')
   
   evolution = doEvolution(hOld, uOld, h, u, x, dx, dt, nt, t, current_time, 
                           line_h, line_u, suptitle)
    
   ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, 
                       repeat=False)
   ani.save('FTBS_NLSWE_specialIC.gif', writer='pillow', fps=20)
   plt.show()
#%% Time stepping


class doEvolution:
    
    def __init__(self, hOld, uOld, h, u, x, dx, dt, nt, t, current_time, 
                 line_h, line_u, suptitle):
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
        self.line_h = line_h
        self.line_u = line_u
        self.suptitle = suptitle
        
    ## Using Durran's NL-SWE formualtion
    def timestep(self, frame):
        #Vectorised FTBS
        self.h[1:] = self.hOld[1:] - self.dt/self.dx * (self.uOld[1:] * 
                    (self.hOld[1:] - self.hOld[:-1]) + self.hOld[1:] * 
                    (self.uOld[1:] - self.uOld[:-1]))
        self.u[1:] = self.uOld[1:] - self.dt/self.dx * (self.uOld[1:] *
                    (self.uOld[1:] - self.uOld[:-1]) + g * 
                    (self.hOld[1:] - self.hOld[:-1]))
    
        #PBCs
        self.h[0] = self.h[-1] 
        self.u[0] = self.u[-1]
    
        self.hOld = self.h.copy()
        self.uOld = self.u.copy()
        
        uTemp = abs(self.uOld) + np.sqrt(g*self.hOld)
        self.dt = min(self.dx / max(uTemp), self.dt)   
    
        
        self.line_h.set_data(self.x, self.h)
        self.line_u.set_data(self.x, self.u)
        
        self.suptitle.set_text(f'Non-linear 1-D SWE with specialised IC to '
                               f'make FTBS work\n Time = {self.current_time:.3f}')
        print(f't={self.t}, dt={self.dt:.5f}, '
              f'current_time={self.current_time:.3f}, ' 
              f'CFL={max(abs(self.u))*self.dt/self.dx:.2f}')
        plt.pause(0.01)
        
        if self.current_time == t_end:
            self.dt = t_end - self.current_time
        elif self.current_time + self.dt > t_end:
            self.dt = t_end - self.current_time
            self.nt = int(np.ceil(t_end/self.dt))
        self.current_time += self.dt
        self.t += 1
  

#%% Params 

if __name__ == '__main__':
    g = 9.81
    nx = 100
    t_end = 0.1
    
    GIFtime()