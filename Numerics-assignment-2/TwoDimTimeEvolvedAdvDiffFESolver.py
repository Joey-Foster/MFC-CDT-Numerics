import numpy as np
from scipy import sparse as sp
from scipy import integrate
from TwoDimStaticAdvDiffFESolver import globalShapeFunctions, globalQuadrature, stiffness, force

def mass(xe):
    output = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            phi = lambda x: globalShapeFunctions(xe, x)[i] * globalShapeFunctions(xe, x)[j]
            output[i,j] = globalQuadrature(xe, phi)
    return output

def TwoDimTimeEvolvedAdvDiffFESolver(S, u, D, resolution, t_max):
    '''
    S (funciton): source term
    u (array): wind velocity [ms^-1]
    D (float): diffusion coefficient [m^2s^-1]
    resolution (string): one of [1_25, 2_5, 5, 10, 20, 40]
    t_nax (float): max runtime of the simulation [s]
    '''
    nodes = np.loadtxt(f'las_grids/las_nodes_{resolution}k.txt')
    IEN = np.loadtxt(f'las_grids/las_IEN_{resolution}k.txt', dtype=np.int64)
    boundary_nodes = np.loadtxt(f'las_grids/las_bdry_{resolution}k.txt', dtype=np.int64)

    southern_boarder = np.where(nodes[boundary_nodes,1] <= 110000)[0]
     
    ID = np.zeros(len(nodes), dtype=np.int64)
    n_eq = 0
    for i in range(len(nodes[:, 1])):
        if i in southern_boarder:
            ID[i] = -1
        else:
            ID[i] = n_eq
            n_eq += 1
    
    N_equations = np.max(ID)+1
    N_elements = IEN.shape[0]
    N_nodes = nodes.shape[0]

    nodes = nodes.T
    
    # Location matrix
    LM = np.zeros_like(IEN.T)
    for e in range(N_elements):
        for a in range(3):
            LM[a,e] = ID[IEN[e,a]]
            
    # Global matrices and force vector
    M = sp.lil_matrix((N_equations, N_equations))
    K = sp.lil_matrix((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        m_e = mass(nodes[:,IEN[e,:]])
        k_e = stiffness(nodes[:,IEN[e,:]], D, u)
        f_e = force(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    M[A, B] += m_e[a, b]
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    
    # Solve
    K = sp.csr_matrix(K)
    M = sp.csc_matrix(M)
    M_inv = sp.linalg.inv(M)
    
    Psi_A = np.zeros(N_nodes)
   # Psi_A[ID>=0] = 
    def rhs(t, psi):
        dpsidt = np.zeros_like(psi)
        dpsidt[ID >= 0] = M_inv @ (F - K @ psi[ID >= 0])
        return dpsidt
    
    
    if resolution == '1_25':
        numeric_res = 1250
    elif resolution == '2_5':
        numeric_res = 2500
    else:
        numeric_res = int(resolution)*1000

    soln = integrate.solve_ivp(rhs, [0, t_max], Psi_A, method='RK45',
                               max_step= 0.5*np.sqrt(numeric_res)/np.linalg.norm(u)
                               ,dense_output=True)
    ts = np.linspace(0, t_max, 201)
    ys = soln.sol(ts)
    
    #normalising
    Psi = np.zeros_like(ys)
    for i in range(1,201):
        Psi[:,i] = 1/max(ys[:,i]) * ys[:,i]
    
    return nodes, IEN, southern_boarder, ts, Psi
    