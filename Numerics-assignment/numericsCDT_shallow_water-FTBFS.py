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
# hOld = h.copy()
# uOld = u.copy()

H = 1

intialStableWaveSpeed = np.sqrt(np.max(g*H))
dt =  0.99 * dx / (2 * intialStableWaveSpeed) #CFL 
t_end = 0.5
nt = int(np.ceil(t_end/dt))


#%% Conversion to Characteristic Variables

Q1 = 0.5 * (u + np.sqrt(g*h))
Q2 = 0.5 * (-u + np.sqrt(g*h))

Q1Old = Q1.copy()
Q2Old = Q2.copy()

#%% Initial state plotting

# plt.figure(figsize=(fig_width_inches,fig_height_inches))

# plt.plot(x,h,'b',label='h')
# plt.legend(loc='upper left')
# plt.ylabel('$h$')
# plt.xlabel('$x$')
# #plt.ylim([0.5*H,2*H])
# plt.xlim([-1,1])
# plt.pause(0.01)

#%% Time stepping

## Using Durran's NL-SWE formualtion

t=0
current_time = 0

while current_time < t_end:
    for j in range(1,nx):

        #FTBS for Q1 and FTFS for Q2
        Q1[j] = Q1Old[j] - 2*dt/dx * Q1Old[j] * (Q1Old[j] - Q1Old[j-1])
        Q2[j] = Q2Old[j] + 2*dt/dx * Q2Old[j] * (Q2Old[j+1] - Q2Old[j])
    
        #Manually update the end point 
        Q1[-1] = Q1Old[-1] - 2*dt/dx * Q1Old[-1] * (Q1Old[-1] - Q1Old[-2])
        Q2[0] = Q2Old[0] + 2*dt/dx * Q2Old[0] * (Q2Old[1] - Q2Old[0])
        
        #PBCs
        Q1[0] = Q1[-1] 
        Q2[-1] = Q2[0]
    

    Q1Old = Q1.copy()
    Q2Old = Q2.copy()

    # uTemp = abs(uOld) + np.sqrt(g*hOld)
    # uTemp = Q1
    # dt = min(dx / max(uTemp), dt)
    if current_time + dt > t_end:
        dt = t_end - current_time
    nt = int(np.ceil(t_end/dt))
    current_time += dt
    t += 1

#%% Plotting

    # Map back to coordinate variables for plotting
    h = 1/g * (Q1 + Q2)**2
    u = Q1 - Q2

    fig, ax = plt.subplots(1,2)
    print(f"t={t}, dt={dt:.5f}, current_time={current_time:.3f}, CFL={2*np.sqrt(g*H)*dt/dx:.2f}")
    plt.cla()
    ax[0].plot(x,h,'b',label='h')
    ax[1].plot(x,u,'r',label='u')
    ax[0].set_ylabel('$h$')
    ax[1].set_ylabel('$u$')
    ax[0].set_ylim([0.9,2])
    for a in ax:
        a.legend(loc='upper left')
        a.set_xlabel('$x$')
        #plt.ylim([0.5*H,2*H])
        a.set_xlim([-1,1])
    plt.tight_layout()
    plt.pause(0.01)

plt.show()