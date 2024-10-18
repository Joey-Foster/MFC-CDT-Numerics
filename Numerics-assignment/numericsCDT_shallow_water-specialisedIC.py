import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Make the plot aspect ratios and text look nice
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 
fig_width_inches = 1 / 72.27 * 443.57848
fig_height_inches = fig_width_inches * ( -1 + np.sqrt(5) ) / 2

#%% Initial data
def h0(x):
    return 1+ np.exp(-5*x**2)

def u0(x):
    return 10*np.ones_like(x)

#%% Params 

g = 9.81
nx = 100
t_end = 0.1

#def doEvolution(g,nx,h0,u0,t_end):
dx = 1/nx

x = np.linspace(-1, 1, nx+1)

h = h0(x)
u = u0(x)
hOld = h.copy()
uOld = u.copy()

def GIFtime(hOld,uOld,t=0,current_time=0):
    fig, ax = plt.subplots()

    intialStableWaveSpeed = np.max(uOld) + np.sqrt(np.max(g*hOld))
    dt =  0.99 *dx / intialStableWaveSpeed #CFL 
    
    nt = int(t_end/dt)
    
    #%% Animation/plotting setup
    
    fig, ax = plt.subplots(1,2,figsize=(fig_width_inches, fig_height_inches))
    line_h, = ax[0].plot([], [],'b', label='h')
    line_u, = ax[1].plot([], [],'r', label='u')
    
    ax[0].set_ylabel('$h$')
    ax[1].set_ylabel('$u$')
    ax[0].set_ylim([0.9,2])
    ax[1].set_ylim([7.5,12])
    for a in ax:
        a.legend(loc='upper left')
        a.set_xlabel('$x$')
        a.set_xlim([-1,1])
    plt.tight_layout()
    suptitle = fig.suptitle('Non-linear 1-D SWE with specialised IC to make FTBS work')
    
    plt.subplots_adjust(top=0.85)

    evolution = doEvolution(hOld, uOld, dt, t, current_time, line_h, line_u, suptitle)
    
    ani = FuncAnimation(fig, evolution.timestep, frames=nt, blit=False, repeat=False)
    ani.save('FTBS_NLSWE_specialIC.gif', writer='pillow', fps=20)
    plt.show()
#%% Time stepping

## Using Duran's NL-SWE formualtion

t=0
current_time = 0

class doEvolution:
    
    def __init__(self, hOld, uOld, dt, t, current_time, line_h, line_u, suptitle):
        self.hOld = hOld
        self.uOld = uOld
        self.dt = dt
        self.t = t
        self.current_time = current_time
        self.line_h = line_h
        self.line_u = line_u
        self.suptitle = suptitle
        

    def timestep(self, frame):
    #while current_time < t_end: 
        #Vectorised FTBS
        h[1:] = self.hOld[1:] - self.dt/dx * (self.uOld[1:] * (self.hOld[1:] - self.hOld[:-1]) + self.hOld[1:] * (self.uOld[1:] - self.uOld[:-1]))
        u[1:] = self.uOld[1:] - self.dt/dx * (self.uOld[1:] * (self.uOld[1:] - self.uOld[:-1]) + g * (self.hOld[1:] - self.hOld[:-1]))
    
        #PBCs
        h[0] = h[-1] 
        u[0] = u[-1]
    
        self.hOld = h.copy()
        self.uOld = u.copy()
        
        uTemp = abs(self.uOld) + np.sqrt(g*self.hOld)
        self.dt = min(dx / max(uTemp), self.dt)   
    
        
        self.line_h.set_data(x, h)
        self.line_u.set_data(x, u)
        
        self.suptitle.set_text(f'Non-linear 1-D SWE with specialised IC to make FTBS work\n Time = {self.current_time:.3f}')
        print(f"t={self.t}, dt={self.dt:.5f}, current_time={self.current_time:.3f}, CFL={max(abs(u))*self.dt/dx:.2f}")
        plt.pause(0.01)
        
        self.current_time += self.dt
        self.t += 1
  

GIFtime(hOld, uOld)