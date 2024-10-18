# BRANCH TEST!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

#Make the plot aspect ratios and text look nice
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 
fig_width_inches = 1 / 72.27 * 443.57848
fig_height_inches = fig_width_inches * ( -1 + np.sqrt(5) ) / 2

fig, ax = plt.subplots()

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

#H = np.mean(h)

intialStableWaveSpeed = np.max(u) + np.sqrt(np.max(g*h))
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

#%% Time stepping

## Using Duran's NL-SWE formualtion

t=0
current_time = 0
def timestep(frame):
#while current_time < t_end:
    global hOld, uOld, dt, t, current_time
    #Vectorised FTBS
    h[1:] = hOld[1:] - dt/dx * (uOld[1:] * (hOld[1:] - hOld[:-1]) + hOld[1:] * (uOld[1:] - uOld[:-1]))
    u[1:] = uOld[1:] - dt/dx * (uOld[1:] * (uOld[1:] - uOld[:-1]) + g * (hOld[1:] - hOld[:-1]))

    #PBCs
    h[0] = h[-1] 
    u[0] = u[-1]

    hOld = h.copy()
    uOld = u.copy()
    
    uTemp = abs(uOld) + np.sqrt(g*hOld)
    dt = min(dx / max(uTemp), dt)   

    
    line_h.set_data(x, h)
    line_u.set_data(x, u)
    
    suptitle.set_text(f'Non-linear 1-D SWE with specialised IC to make FTBS work\n Time = {current_time:.3f}')
    print(f"t={t}, dt={dt:.5f}, current_time={current_time:.3f}, CFL={max(abs(u))*dt/dx:.2f}")
    plt.pause(0.01)
    
    current_time += dt
    t += 1
    
    
    
    #return t, current_time, dt, hOld, uOld
#  
#partial(timestep,hOld=hOld,uOld=uOld,dt=dt,t=t,current_time=current_time
ani = FuncAnimation(fig, timestep, frames=int(t_end/dt), blit=False, repeat=False)
ani.save('FTBS_NLSWE_specialIC.gif', writer='pillow', fps=20)
plt.show()
    
#doEvolution(g, nx, h0, u0, t_end)