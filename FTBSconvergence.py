import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sp

params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
#Make the plot aspect ratios look nice
pts_to_inches = 1/72.27
fig_width_inches = pts_to_inches*443.57848
fig_height_inches = fig_width_inches*(-1+np.sqrt(5))/2


#Initial data
def phi0(x):
    return np.power(np.sin(2*np.pi*x),2)

#%% Error Calculation

def EndPointRMSCalculator(nx,scaling,u,initial_data):

    #params
    dx = 1/nx
    x=np.linspace(0, 1, nx+1)    
    
    nt=scaling*nx
    dt=1/nt
    
    print("CFL = ", u*dt/dx)
    
    #initial conditions
    phi = initial_data(x)
    phiOld = phi.copy()

    def analytic(x,t):
        return initial_data(x-u*t)
    
    #Vectorized FTBS
    for t in range(nt + 1):
        phi[1:] = phiOld[1:] - u*dt/dx * (phiOld[1:] - phiOld[:-1])
        phi[0] = phi[-1]
        phiOld[:] = phi[:]
   
    #FTBS scheme
    # for t in range(nt+1):
    #     for j in range(1,nx+1):
    #         phi[j]=phiOld[j]-u*dt/dx*(phiOld[j]-phiOld[j-1])
    #     phi[0]=phi[-1]
    #     phiOld = phi.copy()
        
        # print(t, t*dt)
        # plt.cla()
        # plt.plot(x,phi,'b',label='Time_'+str(np.around(t*dt,2)))
        # plt.plot(x,analytic(x,(t+1)*dt),'r',label='analytic')
        # plt.legend(loc='upper left')
        # plt.ylabel('$\phi$')
        # plt.ylim([0,1])
        # plt.pause(0.01)
        
    
    AnalyticAtEnd = analytic(x, (nt+1)*dt)[:-1]
    NumericalAtEnd = phi[:-1]
    
    return 1/nx**(1/4)*np.linalg.norm(AnalyticAtEnd-NumericalAtEnd,4) 
    #return np.sum(abs(AnalyticAtEnd-NumericalAtEnd)**2)**(1/2)
    #'return np.sqrt(np.mean((AnalyticAtEnd-NumericalAtEnd)**2))

print(EndPointRMSCalculator(100, 2, 2, phi0))
# print(EndPointRMSCalculator(200, 1, 0.9, phi0))

#%% Parameters of simulation
CFL=0.9
scaling = 2

nxs= 2**(np.arange(6,15))
nts = scaling*nxs

u=CFL*scaling


#%% Run error for loop

errors = np.zeros_like(nxs,dtype=np.float64)

for i, nx in enumerate(nxs):
        
        errors[i] = EndPointRMSCalculator(nx, scaling, u, phi0)
        print(f"{i+1} out of {len(nxs)} done")
        
#%% Compute trendline
        
dxs=1/nxs
dts=1/nts

errors_polyfit_coeffs = np.polyfit(np.log(dxs),np.log(errors),1)
errors2_polyfit_coeffs = np.polyfit(np.log(dts),np.log(errors),1)

def trendline(data,x):
    return np.poly1d(data)(x)

#%% doPlotting
plt.figure(1)
fig, ax = plt.subplots(1,1,figsize=(fig_width_inches,fig_height_inches))
plt.loglog(dxs,np.exp(trendline(errors_polyfit_coeffs,np.log(dxs))),'-r',label=rf'$\propto {{\Delta x}}^{{{errors_polyfit_coeffs[0]:.2f}}}$')
plt.loglog(dxs, errors, 'x',lw=2,label=r'Error wrt $\Delta x$',c='k')
plt.loglog(dts,np.exp(trendline(errors2_polyfit_coeffs,np.log(dts))),'--',c='orange',label=rf'$\propto {{\Delta t}}^{{{errors2_polyfit_coeffs[0]:.2f}}}$')
plt.loglog(dts, errors, '.',lw=2,label=r'Error wrt $\Delta t$',c='b')
plt.xlabel('Discretisation Step Size')
plt.ylabel(r'$\mathrm{RMS(Error)}$')
#plt.set_title(r'Error against $\Delta x$ and $\Delta t$')
plt.legend()

plt.title(rf"Courant number $\mu$ = {CFL}, wave speed a = {u}",y=1.02)

plt.savefig("FTBScourant=0.9.pdf", format="pdf", bbox_inches="tight")

plt.show()