import numpy as np
from scipy import sparse as sp
from scipy import integrate
from TwoDimStaticAdvDiffFESolver import (globalShapeFunctions, globalQuadrature, 
                                         stiffness, force)

def mass(xe):
    """
    Computes the mass matrix for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).

    Returns:
    output (np.ndarray): A 3x3 mass matrix for the triangular element.
    """
    output = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            phi = lambda x: globalShapeFunctions(xe, x)[i] * globalShapeFunctions(xe, x)[j]
            output[i,j] = globalQuadrature(xe, phi)
    return output

def TwoDimTimeEvolvedAdvDiffFESolver(S, u, D, resolution, t_max):
    """
    Solves the 2D time-dependent advection-diffusion equation using the finite 
    element method.

    Parameters:
    S (function): Source term function that takes global coordinates as input 
                  and returns a scalar value.
    u (array-like): Advection velocity vector [ms^-1].
    D (float): Diffusion coefficient [m^2s^-1].
    resolution (string): Grid resolution, one of ['1_25', '2_5', '5', '10', '20', '40'].
    t_max (float): Maximum runtime of the simulation [s].

    Returns:
    tuple: A tuple containing the following elements:
           - nodes (np.ndarray): Array of node coordinates.
           - IEN (np.ndarray): Array of element connectivity.
           - southern_boarder (np.ndarray): Array of indices of nodes on the 
                                            southern border.
           - ts (np.ndarray): Array of timesteps at which the solution was evaluated.
           - Psi_A (np.ndarray): Array of computed solution values at the nodes, 
                                 normalised.
    """
    # Read in data
    nodes = np.loadtxt(f'las_grids/las_nodes_{resolution}k.txt')
    IEN = np.loadtxt(f'las_grids/las_IEN_{resolution}k.txt', dtype=np.int64)
    boundary_nodes = np.loadtxt(f'las_grids/las_bdry_{resolution}k.txt', dtype=np.int64)

    # locate southern boarder for Dirichlet BC
    southern_boarder = np.where(nodes[boundary_nodes,1] <= 110000)[0]
     
    # Set all nodes in 'southern_boarder' to be ignored by the solver
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
            
    # Global matrices and force vector. Calling sparse for memory efficiency
    M = sp.lil_matrix((N_equations, N_equations))
    K = sp.lil_matrix((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        # compute the individual contriubtions from each element
        m_e = mass(nodes[:,IEN[e,:]])
        k_e = stiffness(nodes[:,IEN[e,:]], D, u)
        f_e = force(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
            # if not on a BC node, assemble full matrices and force vector
                if (A >= 0) and (B >= 0):
                    M[A, B] += m_e[a, b]
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    
    # Store matrices for timestepping
    K = sp.csr_matrix(K)
    M = sp.csc_matrix(M)
    M_inv = sp.linalg.inv(M)
    
    # Initial condition for Psi_A
    Psi_A = np.zeros(N_nodes)
    def rhs(t, psi):
        dpsidt = np.zeros_like(psi)
        dpsidt[ID >= 0] = M_inv @ (F - K @ psi[ID >= 0])
        return dpsidt
    
    # extract numerical value from string resolution. (There is definitely a more
    # elegant way to do this...)
    if resolution == '1_25':
        numeric_res = 1250
    elif resolution == '2_5':
        numeric_res = 2500
    else:
        numeric_res = int(resolution)*1000

    # Run RK45 timestepping
    soln = integrate.solve_ivp(rhs, [0, t_max], Psi_A, method='RK45',
                               max_step= 0.5*np.sqrt(numeric_res)/np.linalg.norm(u)
                               ,dense_output=True)
    # interpolate y at linearly spaced times for consistent array size
    ts = np.linspace(0, t_max, 201)
    ys = soln.sol(ts)
    
    # normalising
    Psi = np.zeros_like(ys)
    for i in range(1,201):
        Psi[:,i] = 1/max(ys[:,i]) * ys[:,i]
    
    return nodes, IEN, southern_boarder, ts, Psi
    