import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sp
from PIL import Image

#params
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }

pts_to_inches = 1/72.27

fig_width_inches = pts_to_inches*443.57848
fig_height_inches = fig_width_inches*(-1+np.sqrt(5))/2

def phi0(x):
    return np.power(np.sin(2*np.pi*x),2)

def EndPointTwoNormAbsoluteDifferenceCalculator(dx,dt,CFL,initial_data):

    #params
    nx = int(1/dx)
    nt = int(1/dt)
    x=np.linspace(0, 1, nx+1)    
    u=CFL*dx/dt
    
    #initial conditions
    phi = initial_data(x)
    phiOld = phi.copy()
    
    #plot inital conditions
    plt.plot(x,phi, 'k',label="initial conditions")
    plt.legend()
    plt.ylabel('$\phi$')
    plt.axhline(0,linestyle=':',color='black')
    plt.ylim([0,1])
    
    def analytic(x,t):
        return initial_data(x-u*t)
    
    #images = []
    #fig = plt.figure(figsize=(fig_width_inches,fig_height_inches))
    #loop over all time steps
    #FTBS scheme
    for t in range(nt+1):
        for j in range(1,nx+1):
            phi[j]=phiOld[j]-u*dt/dx*(phiOld[j]-phiOld[j-1])
        phi[0]=phi[-1]
        phiOld = phi.copy()
        
        print(t, t*dt)
        plt.cla()
        plt.plot(x,phi,'b',label='Time_'+str(np.around(t*dt,2)))
        plt.plot(x,analytic(x,(t+1)*dt),'r',label='analytic')
        plt.legend(loc='upper left')
        plt.ylabel('$\phi$')
        plt.ylim([0,1]) 
        plt.pause(0.01)
        
    
    AnalyticAtEnd = analytic(x, (nt+1)*dt)
    NumericalAtEnd = phi
    
    return np.linalg.norm(abs(AnalyticAtEnd-NumericalAtEnd),2) 

#print(EndPointTwoNormAbsoluteDifferenceCalculator(1/2000, 1/100, 1, phi0))
# dts = 0.1/2**(np.arange(1,10))
# dxs=0.1/2**(np.arange(1,10))
# #dt=0.001

# errors = np.zeros_like(dxs)

# for i, dx in enumerate(dxs):
#       for dt in dts:
     
#           errors[i]= EndPointTwoNormAbsoluteDifferenceCalculator(dx, dt, 0.5, phi0) 
#           #print(dt, dx)

# errors_polyfit_coeffs = np.polyfit(np.log(dxs),np.log(errors),1)

# def trendline(data,x):
#     return np.poly1d(data)(x)

# plt.figure(figsize=(fig_width_inches,fig_height_inches))
# plt.loglog(dxs,np.exp(trendline(errors_polyfit_coeffs,np.log(dxs))),'-r',label=rf'$\propto {{\Delta x}}^{{{np.around(errors_polyfit_coeffs[0],1)}}}$')
# plt.loglog(dxs, errors, 'x',lw=2,label='Error',c='k')
# plt.xlabel(r'$\Delta x$')
# plt.ylabel(r'$\left\Vert\mathrm{Error}\right\Vert_{2}$')
# plt.title(r'Error against $\Delta x$')
# plt.legend()

plt.show()
#plt.savefig("figure3.pdf", format="pdf", bbox_inches="tight")
  
        # fig.savefig('tmp.png')
        # images.append(Image.open('tmp.png'))
# images[0].save('advection_gif.gif', save_all=True, append_images=images, duration=200, loop=0)
