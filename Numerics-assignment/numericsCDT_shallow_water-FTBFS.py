import numpy as np
import matplotlib.pyplot as plt

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
    return 1 + np.exp(-10*x**2)

def u0(x):
    return 0*x

#%% Params 

g = 9.81

nx = 100
dx = 1/nx

x = np.linspace(-1, 1, nx+1)

h = h0(x)
u = u0(x)
hOld = h.copy()
uOld = u.copy()

intialStableWaveSpeed = np.max(u) + np.sqrt(np.max(g*h))
dt =  dx / intialStableWaveSpeed #CFL 
t_end = 0.2
nt = int(t_end/dt)



plt.figure(figsize=(fig_width_inches,fig_height_inches))

plt.plot(x,h,'b',label='h')
plt.legend(loc='upper left')
plt.ylabel('$h$')
plt.xlabel('$x$')
#plt.ylim([0.5*H,2*H])
plt.xlim([-1,1])
plt.pause(0.01)

#%% Time stepping

## Using Durran's NL-SWE formualtion

#FT-B/FS   
t=0
current_time = 0


for t in range(10):
    for j in range(1,nx):
        if uOld[j] >= 0:

            #Upwind in \partial_x
            h[j] = hOld[j] - dt/dx * (uOld[j] * (hOld[j] - hOld[j-1]) + hOld[j] * (uOld[j] - uOld[j-1]))
            u[j] = uOld[j] - dt/dx * (uOld[j] * (uOld[j] - uOld[j-1]) + g * (hOld[j] - hOld[j-1]))
        
            #Manually update the end point 
            h[-1] = hOld[-1] - dt/dx * (uOld[-1] * (hOld[-1] - hOld[-2]) + hOld[-1] * (uOld[-1] - uOld[-2]))
            u[-1] = uOld[-1] - dt/dx * (uOld[-1] * (uOld[-1] - uOld[-2]) + g * (hOld[-1] - hOld[-2]))
            
            #PBCs
            h[0] = h[-1] 
            u[0] = u[-1]
            
        else:
            
            #Downwind in \partial_x
            h[j] = hOld[j] - dt/dx * (uOld[j] * (hOld[j+1] - hOld[j]) + hOld[j] * (uOld[j+1] - uOld[j]))
            u[j] = uOld[j] - dt/dx * (uOld[j] * (uOld[j+1] - uOld[j]) + g * (hOld[j+1] - hOld[j]))
            
            #PBCs
            h[-1] = h[1]
            u[-1] = u[1]
            
            h[0] = h[-2]
            u[0] = u[-2]

    hOld = h.copy()
    uOld = u.copy()

    uTemp = abs(uOld) + np.sqrt(g*hOld)
    dt = min(dx / max(uTemp), dt)
    current_time += dt
    t += 1

#%% Plotting

    fig, ax = plt.subplots(1,2)
    print(f"t={t}, dt={dt:.5f}, current_time={current_time:.3f}, CFL={max(abs(u))*dt/dx:.2f}")
    plt.cla()
    ax[0].plot(x,h,'b',label='h')
    ax[1].plot(x,u,'r',label='u')
    ax[0].set_ylabel('$h$')
    ax[1].set_ylabel('$u$')
    for a in ax:
        a.legend(loc='upper left')
        a.set_xlabel('$x$')
        #plt.ylim([0.5*H,2*H])
        a.set_xlim([-1,1])
    plt.tight_layout()
    plt.pause(0.01)

plt.show()