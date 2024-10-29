import numpy as np
from NLSWE_specialisedIC import produceStaticPlot


#Params
g = 9.81
H=1
nx = 100
U0 = 0

guassian, cosine  = [lambda x: 1 + np.exp(-5*x**2), lambda x: 1 + 0.5*np.power(np.cos(np.pi *x),2)]

h0 = guassian

print('Logging console data for figure 1:')
#Figure 1
produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [-1,1], U0=U0, nx=nx, g=g, 
                  t_simulation_range = 10, t_plotting_range =[0, 10], t_sample = 4, 
                  suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + e^{{-5x^2}}$, $u_0$ = {U0}', 
                  filename = 'unstable_attempt_guassian')

h0 = cosine

print('\nLogging console data for figure 2:')
#Figure 2
produceStaticPlot(h0, h_lims = [1,2], u_lims = [-1,1], U0=U0, nx=nx, g=g, 
                  t_simulation_range = 35, t_plotting_range =[20, 32] , t_sample = 4, 
                  suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + \frac{{1}}{{2}}\cos^2(\pi x)$, $u_0$ = {U0}', 
                  filename = 'unstable_attempt_cosine')

h0 = guassian
U0 = 5

print('\nLogging console data for figure 3:')
#Figure 3
produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [3.5,6.5], U0=U0, nx=nx, g=g,
                  t_simulation_range = 300, t_plotting_range =[280, 300] , t_sample = 4, 
                  suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + e^{{-5x^2}}$, $u_0$ = {U0}', 
                  filename = 'cheating_attempt_stable')


U0 = 4.3

print('\nLogging console data for figure 4:')
#Figure 4
produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [2.5,5.5], U0=U0, nx=nx, g=g,
                  t_simulation_range = 160, t_plotting_range =[125, 155] , t_sample = 4, 
                  suptitle = fr'Non-linear 1-D SWE with $h_0 = 1 + e^{{-5x^2}}$, $u_0$ = {U0}', 
                  filename = 'cheating_attempt_unstable')


# from NLSWE_characteristicFTBFS import produceStaticPlot

# produceStaticPlot(h0, h_lims = [0.9,2], u_lims = [-1,1], nx = nx, g = g, H = H, 
#                   t_simulation_range = 200, t_plotting_range = [180,200], t_sample = 4, 
#                   suptitle = r'Non-linear 1-D SWE with $h_0 = 1 + e^{-5x^2}$, $u_0 = 0$', 
#                   filename = 'NLSWE_arbitraryIC')