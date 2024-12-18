import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

'''
Solving u .∇Ψ = S + D ΔΨ
'''

def localShapeFunctions(xi):
    """
    Computes the local shape functions for a triangular element.

    Parameters:
    xi (array-like): Local coordinates (xi1, xi2) within the triangular element.

    Returns:
    np.ndarray: Array of shape functions evaluated at the given local coordinates.
    """
    return np.array([1-xi[0]-xi[1], xi[0], xi[1]])

def localShapeFunctionDerivatives():
    """
    Computes the derivatives of the local shape functions with respect to the 
    local coordinates.

    Returns:
    matrix (mp.ndarray): A 2x3 matrix where each row contains the derivatives of the 
                    shape functions with respect to xi1 and xi2, respectively.
                    - The first row contains the derivatives with respect to xi1.
                    - The second row contains the derivatives with respect to xi2.
    """
    matrix = np.zeros((2,3))
    matrix[0,:] = np.array([-1,1,0])
    matrix[1,:] = np.array([-1,0,1])
    return matrix

def local2globalCoords(xe,xi):
    """
    Transforms local coordinates to global coordinates for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row 
                     represents a coordinate (x or y).
    xi (array-like): Local coordinates (xi1, xi2) within the triangular element.

    Returns:
    globalcoords (np.ndarray): The global coordinates corresponding to the given
                               local coordinates.
    """
    globalcoords = np.zeros(2)
    N = localShapeFunctions(xi)
    for i in range(2):
        globalcoords[i] = xe[i,0]*N[0] + xe[i,1]*N[1] + xe[i,2]*N[2]
    return globalcoords

def jacobian(xe):
    """
    Computes the Jacobian matrix for the transformation from local to global 
    coordinates for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
                     
    Returns:
    output (np.ndarray): A 2x2 Jacobian matrix containing the partial derivatives
                         of the global coordinates with respect to local coordinates.
    """
    Nprime = localShapeFunctionDerivatives()
    output = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            output[i,j] = xe[i,0]*Nprime[j,0] + xe[i,1]*Nprime[j,1] + xe[i,2]*Nprime[j,2]
    return output

def globalShapeFunctionDerivatives(xe):
    """
    Computes the derivatives of the shape functions with respect to global coordinates
    for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).

    Returns:
    np.ndarray: A 2x3 matrix where each column contains the derivatives of the 
                shape functions with respect to global coordinates (x and y).
                - The first row contains the derivatives with respect to x.
                - The second row contains the derivatives with respect to y.
    """
    return np.linalg.inv(jacobian(xe)).T @ localShapeFunctionDerivatives()

def localQuadrature(psi):
    """
    Performs numerical integration using Gauss quadrature over a triangular 
    element in local coordinates.

    Parameters:
    psi (function): A function to be integrated, which takes local coordinates
                    as input and returns a scalar value.

    Returns:
    quadrature (float): The result of the numerical integration.
    """
    quadrature = 0
    
    #Gauss-quadrature evaluation points (2nd order accurate approximation)
    xis = 1/6 * np.array([[1, 4, 1],
                          [1, 1, 4]]) 
    for i in range(3):
        quadrature += 1/6 * psi(xis[:,i])
    return quadrature

def globalQuadrature(xe, phi):
    """
    Performs numerical integration using Gauss quadrature over a triangular 
    element in global coordinates.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    phi (function): A function to be integrated, which takes global coordinates
                    as input and returns a scalar value.

    Returns:
    float: The result of the numerical integration.
    """
    detJ = np.linalg.det(jacobian(xe))
    integrand = lambda xi: abs(detJ)*phi(local2globalCoords(xe, xi))
    return localQuadrature(integrand)    

def diffusion_stiffness(xe):
    """
    Computes the diffusion stiffness matrix for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).

    Returns:
    output (np.ndarray): A 3x3 diffusion stiffness matrix for the triangular element.
    """
    output = np.zeros((3,3))
    dxNa = globalShapeFunctionDerivatives(xe)
    for i in range(3):
        for j in range(3):
            phi = lambda x: dxNa[0,i]*dxNa[0,j] + dxNa[1,i]*dxNa[1,j]
            output[i,j] = globalQuadrature(xe, phi)
    return output
    
def global2localCoords(xe, x):
    """
    Transforms global coordinates to local coordinates for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    x (array-like): Global coordinates to be transformed.

    Returns:
    localcoords (np.ndarray): The local coordinates corresponding to the given 
                              global coordinates.
    """
    # Inverts the equation x = x0^e (1 - xi1 - xi2) + x1^e xi1 + x2^e xi2
    #                      y = y0^e (1 - xi1 - xi2) + y1^e xi1 + y2^e xi2
    # to solve for xi1, xi2.
    diffs = np.array([[xe[0,1]-xe[0,0], xe[0,2]-xe[0,0]],
                      [xe[1,1]-xe[1,0], xe[1,2]-xe[1,0]]])
    localcoords = np.linalg.solve(diffs, x-np.array([xe[0,0],xe[1,0]]))
    return localcoords

def globalShapeFunctions(xe, x):
    """
    Computes the shape functions at given global coordinates for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    x (array-like): Global coordinates where the shape functions are to be evaluated.

    Returns:
    np.ndarray: Array of shape functions evaluated at the given global coordinates.
    """
    return localShapeFunctions(global2localCoords(xe, x))

def advection_stiffness(xe, u):
    """
    Computes the advection stiffness matrix for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    u (array-like): Advection velocity vector.

    Returns:
    output (np.ndarray): A 3x3 advection stiffness matrix for the triangular element.
    """
    output = np.zeros((3,3))
    dxNa = globalShapeFunctionDerivatives(xe)
    for i in range(3):
        for j in range(3):
            integrand = lambda x: globalShapeFunctions(xe, x)[i] * (u[0]*dxNa[0,j]
                                                                    + u[1]*dxNa[1,j])
            output[i,j] = globalQuadrature(xe,integrand)
    return output

def stiffness(xe, D, u):
    """
    Computes the combined stiffness matrix for a triangular element, combining both
    diffusion and advection effects.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    u (array-like): Advection velocity vector.
    D (float): Diffusion coefficient.

    Returns:
    np.ndarray: A 3x3 combined stiffness matrix for the triangular element.
    """
    return D * diffusion_stiffness(xe) - advection_stiffness(xe, u)

def force(xe, S):
    """
    Computes the force vector for a triangular element.

    Parameters:
    xe (np.ndarray): A 2x3 matrix containing the global coordinates of the 
                     element nodes. Each column represents a node, and each row
                     represents a coordinate (x or y).
    S (function): A source term function that takes global coordinates as input
                  and returns a scalar value.

    Returns:
    output (np.ndarray): A 3-element force vector for the triangular element.
    """
    output = np.zeros(3)
    for i in range(3):
        integrand = lambda x: S(x) * globalShapeFunctions(xe, x)[i]
        output[i] = globalQuadrature(xe, integrand)
    return output
        
def TwoDimStaticAdvDiffFESolver(S, u, D, resolution):
    """
    Solves the 2D steady-state advection-diffusion equation using the finite 
    element method.

    Parameters:
    S (function): Source term function that takes global coordinates as input 
                  and returns a scalar value.
    u (array-like): Advection velocity vector [ms^-1].
    D (float): Diffusion coefficient [m^2s^-1].
    resolution (string): Grid resolution, one of ['1_25', '2_5', '5', '10', '20', '40'].

    Returns:
    tuple: A tuple containing the following elements:
           - nodes (np.ndarray): Array of node coordinates.
           - IEN (np.ndarray): Array of element connectivity.
           - southern_boarder (np.ndarray): Array of indices of nodes on the 
                                            southern border.
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
            
    # Global stiffness matrix and force vector. Calling sparse for memory efficiency
    K = sp.lil_matrix((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        # compute the individual contriubtions from each element
        k_e = stiffness(nodes[:,IEN[e,:]], D, u)
        f_e = force(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
            # if not on a BC node, assemble full stiffness and force arrays
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    
    
    # Solve
    K = sp.csr_matrix(K)
    Psi_interior = sp.linalg.spsolve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise, Psi_A is homogeneous Dirichlet
            Psi_A[n] = Psi_interior[ID[n]]
            
    # normalising
    Psi_A = 1/max(Psi_A)*Psi_A
    
    return nodes, IEN, southern_boarder, Psi_A
