import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate 

#%% Exercise 3.1(c)
Ne = 2
# = 1/Ne
psi = np.zeros(Ne + 1)

K = np.array([[4, -2],[-2, 2]])
F = np.array([1/4, 1/24])

psi[0] = 0 #Just for clarity - serves no purpose in this implementation
psi[1:] = np.linalg.solve(K,F)

#print(psi)

def psi_analytic1(x):
    return x/6 * (x**2 - 3*x + 3)

x_analytic = np.linspace(0, 1, 100)
x_numeric = np.linspace(0, 1, Ne+1)

plt.plot(x_analytic, psi_analytic1(x_analytic),'-b',label='analytic')
plt.plot(x_numeric, psi, '--r', label='numeric')
plt.grid()
plt.legend()
plt.title('Exercise 3.1(c) solution: Ne=2 solved globally over nodes')
plt.show()

#%% Exercise 3.2
plt.figure()

def OneDimFESolver(Ne, S, DirichletL=False, DirichletR=False, NeumannL=False, NeumannR=False):
    
    nodes, dx = np.linspace(0, 1, Ne+1, retstep=True)
    #nodes = np.array([0, *sorted(np.random.rand(Ne-1)), 1])
    print(len(nodes))
    
    DirichletCount = 0
    if type(DirichletL)!=bool:
        DirichletCount += 1
    if type(DirichletR)!=bool:
        DirichletCount += 1
        
    MaxSolvingDimension = Ne+1-DirichletCount
    
    LM = np.zeros((2, MaxSolvingDimension), dtype=np.int64)
    for e in range(MaxSolvingDimension):
        if e==0:
            if type(DirichletL)!=bool:
                LM[0, e] = -1 # Don't consider the left BP
                LM[1, e] = 0
            else:
                LM[0, e] = 0
                LM[1, e] = 1
        elif 0 < e < MaxSolvingDimension - 1:
            LM[0, e] = LM[1, e-1]
            LM[1, e] = LM[0, e] + 1
        else: 
            LM[0, e] = LM[1, e-1]
            if type(DirichletR)!=bool:
                LM[1, e] = -1 # Don't consider right BP
            else:
                LM[1, e] = LM[0, e]
            
    print(LM)
    
    def stiffness(element):
        dx = element[-1]-element[0]
        return 1/dx * np.array([[1,-1],[-1,1]])
            
    def force(element, S):
        dx = element[-1]-element[0]
        def integrand(xi,xi_a):
            return S(1/2 * (dx*xi + element[0]+element[-1])) * (1/2*(1+xi_a*xi))
        integrand1 = lambda xi: integrand(xi, -1)
        integrand2 = lambda xi: integrand(xi, 1)
        force_vector = np.zeros(2)
        force_vector[0] = dx/2 * scipy.integrate.quad(integrand1, -1,1)[0]
        force_vector[1] = dx/2 * scipy.integrate.quad(integrand2, -1,1)[0]
        return force_vector
            
    
    K = np.zeros((MaxSolvingDimension, MaxSolvingDimension))
    F = np.zeros((MaxSolvingDimension,))
    for e in range(MaxSolvingDimension):
        k_e = stiffness(nodes[e:e+2])
        f_e = force(nodes[e:e+2], S)
        for a in range(2):
            A = LM[a, e]
            for b in range(2):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
        
  
        # Modify force vector for Dirichlet BC
        if e == 0 and type(DirichletL)!=bool:
            F[0] -= DirichletL * k_e[1, 0]
        if e == MaxSolvingDimension - 1 and type(DirichletR)!=bool:
            F[-1] -= DirichletR * k_e[1, 0]
    print(f'K = {K}')

    # Modify force vector for Neumann BC
    if type(NeumannL)!=bool:
        F[0] += NeumannL
    if type(NeumannR)!=bool:
        F[-1] += NeumannR
   
    
    Psi_A = np.zeros_like(nodes)
    Lindex = 0
    Rindex = len(Psi_A)+1
    if type(DirichletL)!=bool:
        Psi_A[0] = DirichletL
        Lindex = 1
    if type(DirichletR)!=bool:
        Psi_A[-1] = DirichletR
        Rindex = -1
    Psi_A[Lindex:Rindex] = np.linalg.solve(K, F)

    print(Psi_A)

    return nodes, Psi_A

###############################################################################
 
def S1(x):
    return 1-x
        
def S2(x):
    return (1-x)**2

def S3(x):
    if abs(x-1/2) < 1/4:
        return 1
    else:
        return 0

def psi_analytic2(x):
    return x/12 * (4 - 6*x + 4*x**2 - x**3)

def psi_analytic3(x):
    output = np.zeros_like(x)
    for i, x_i in enumerate(x):
        if x_i <= 1/4:
            output[i] = 0.3*x_i+0.1
        elif 1/4 <= x_i <= 3/4:
            output[i] = -1/2 * x_i**2 + 0.55*x_i + 11/160
        else:
            output[i] = -0.2*x_i + 0.35
    return output

nodes, soln = OneDimFESolver(7, S1, DirichletL=0, DirichletR=1)

x_analytic = np.linspace(0, 1, 100)

#plt.plot(x_analytic, psi_analytic3(x_analytic),'-b',label='analytic')
plt.plot(nodes, soln, '--x', color='r', label='numeric')
plt.grid()
plt.legend()
plt.title(f'Exercise 3.2 solution: Ne={Ne} solved locally over elements')
plt.show()
