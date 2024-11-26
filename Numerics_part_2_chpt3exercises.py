import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate 

#%% Exercise 3.1(c)
def exercise3point1c():
    Ne = 2
    # = 1/Ne
    psi = np.zeros(Ne + 1)
    
    K = np.array([[4, -2],[-2, 2]])
    F = np.array([1/4, 1/24])
    
    psi[0] = 0 #Just for clarity - serves no purpose in this implementation
    psi[1:] = np.linalg.solve(K,F)
    
    #print(psi)
    
    x_analytic = np.linspace(0, 1, 100)
    x_numeric = np.linspace(0, 1, Ne+1)
    
    plt.plot(x_analytic, psi_analytic1(x_analytic),'-b',label='analytic')
    plt.plot(x_numeric, psi, '--r', label='numeric')
    plt.grid()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\psi$')
    plt.legend()
    plt.title('Exercise 3.1(c): Solve globally over nodes\n' r'Ne = 2 and $S(x) = 1-x$')
    plt.show()

#%% Exercise 3.2

def OneDimFESolver(Ne, S, DirichletL=None, DirichletR=None, NeumannL=None, NeumannR=None):
    """
    Solves the one-dimensional PDE problem \partial_xx \psi = -S(x) using the 
    finite element method on an unstructured grid.

    Note: Providing exactly two of the 4 possible boundary conditions is necessary 
    for a well-posed PDE problem. Double-Neumann is ill-posed.

    Parameters:
    Ne (int): Number of elements.
    S (function): Source function S(x).
    DirichletL (float, optional): Dirichlet boundary condition at the left end. Default is None.
    DirichletR (float, optional): Dirichlet boundary condition at the right end. Default is None.
    NeumannL (float, optional): Neumann boundary condition at the left end. Default is None.
    NeumannR (float, optional): Neumann boundary condition at the right end. Default is None.

    Returns:
    tuple: A tuple containing the nodes and the solution vector.
    """
    
    #nodes, dx = np.linspace(0, 1, Ne+1, retstep=True)
    nodes = np.array([0, *sorted(np.random.rand(Ne-1)), 1])
    
    DirichletCount = 0
    if DirichletL != None:
        DirichletCount += 1
    if DirichletR != None:
        DirichletCount += 1
        
    MaxSolvingDimension = Ne + 1 - DirichletCount
    
    
    LM = np.zeros((2, Ne), dtype=np.int64)
    for e in range(Ne):
        if e==0:
            if DirichletL != None:
                LM[0, e] = -1 # Don't consider the left BP
                LM[1, e] = 0
            else:
                LM[0, e] = 0
                LM[1, e] = 1
        elif 0 < e < Ne - 1:
            LM[0, e] = LM[1, e-1]
            LM[1, e] = LM[0, e] + 1
        else: 
            LM[0, e] = LM[1, e-1]
            if DirichletR != None:
                LM[1, e] = -1 # Don't consider right BP
            else:
                LM[1, e] = LM[0, e] + 1
    
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
    for e in range(Ne):
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
        if e == 0 and DirichletL != None:
            F[0] -= DirichletL * k_e[1, 0]
        if e == MaxSolvingDimension - 1 and DirichletR != None:
            F[-1] -= DirichletR * k_e[1, 0]
            
    # Modify force vector for Neumann BC
    if NeumannL != None:
        F[0] -= NeumannL
    if NeumannR != None:
        F[-1] += NeumannR
    
    Psi_A = np.zeros_like(nodes)
    Lindex = 0
    Rindex = len(Psi_A) 
    
    if DirichletL != None:
        Psi_A[0] = DirichletL
        Lindex = 1
    if DirichletR != None:
        Psi_A[-1] = DirichletR
        Rindex = -1

    Psi_A[Lindex:Rindex] = np.linalg.solve(K, F)

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

def psi_analytic1(x):
     return x/6 * (x**2 - 3*x + 3)

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

###############################################################################

def exercise3point2(Ne):
    
    sources = [S1, S2, S3]
    analytics = [psi_analytic1, psi_analytic2, psi_analytic3]
    DirichletL = [0, None, 0.1]
    DirichletR = [1/6, 1/12, None]
    NeumannL = [None, 1/3, None]
    NeumannR = [None,None,-0.2]
    latexsourcetitles = ['$S(x) = 1-x$',
                         '$S(x) = (1-x)^2$',
                         '$S(x) = \{1$ in the centre, piecewise}']
    boundarytitles = ['Dirichlet L & R', 
                      'Meumann L & Dirichlet R', 
                      'Dirichlet L & Neumann R']
    
    for i in range(3):
        plt.figure()
    
        nodes, soln = OneDimFESolver(Ne, sources[i], 
                                     DirichletL=DirichletL[i], DirichletR=DirichletR[i],
                                     NeumannL=NeumannL[i], NeumannR=NeumannR[i])
        
        x_analytic = np.linspace(0, 1, 100)
        
        plt.plot(x_analytic, analytics[i](x_analytic),'-b',label='analytic')
        plt.plot(nodes, soln, '--x', color='r', label='numeric')
        plt.grid()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\psi$')
        plt.legend()
        plt.title('Exercise 3.2 : '
                  f'Solve locally over elements\n Ne = {Ne}, ' 
                  f'{latexsourcetitles[i]}\n with {boundarytitles[i]} BCs')
        plt.show()
    
if __name__ == '__main__':
    exercise3point1c()
    exercise3point2(Ne=10)
