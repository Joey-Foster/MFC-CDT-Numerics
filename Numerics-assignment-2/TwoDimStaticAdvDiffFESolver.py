import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

'''
Solving u .∇Ψ = S + D ΔΨ
'''

def localShapeFunctions(xi):
    return np.array([1-xi[0]-xi[1], xi[0], xi[1]])

def localShapeFunctionDerivatives():
    '''
    d_xi1 : N_0, N_1, N_2
    d_xi2 : N_0, N_1, N_2

    '''
    matrix = np.zeros((2,3))
    matrix[0,:] = np.array([-1,1,0])
    matrix[1,:] = np.array([-1,0,1])
    return matrix

def local2globalCoords(xe,xi):
    '''
    xe is a 2x3 matrix containing the global coords of the element nodes
    xi are the local coords
    
    returns x, the global coords
    '''
    globalcoords = np.zeros(2)
    N = localShapeFunctions(xi)
    for i in range(2):
        globalcoords[i] = xe[i,0]*N[0] + xe[i,1]*N[1] + xe[i,2]*N[2]
    return globalcoords

def jacobian(xe):
    Nprime = localShapeFunctionDerivatives()
    output = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            output[i,j] = xe[i,0]*Nprime[j,0] + xe[i,1]*Nprime[j,1] + xe[i,2]*Nprime[j,2]
    return output

def globalShapeFunctionDerivatives(xe):
    return np.linalg.inv(jacobian(xe)).T @ localShapeFunctionDerivatives()

def localQuadrature(psi):
    quadrature = 0
    
    #Gauss-quadrature evaluation points (2nd order accurate approximation)
    xis = 1/6 * np.array([[1, 4, 1],
                          [1, 1, 4]]) 
    for i in range(3):
        quadrature += 1/6 * psi(xis[:,i])
    return quadrature

def globalQuadrature(xe, phi):
    detJ = np.linalg.det(jacobian(xe))
    integrand = lambda xi: abs(detJ)*phi(local2globalCoords(xe, xi))
    return localQuadrature(integrand)    

def diffusion_stiffness(xe):
    output = np.zeros((3,3))
    dxNa = globalShapeFunctionDerivatives(xe)
    for i in range(3):
        for j in range(3):
            phi = lambda x: dxNa[0,i]*dxNa[0,j] + dxNa[1,i]*dxNa[1,j]
            output[i,j] = globalQuadrature(xe, phi)
    return output
    
def global2localCoords(xe, x):
    diffs = np.array([[xe[0,1]-xe[0,0], xe[0,2]-xe[0,0]],
                      [xe[1,1]-xe[1,0], xe[1,2]-xe[1,0]]])
    localcoords = np.linalg.solve(diffs, x-np.array([xe[0,0],xe[1,0]]))
    return localcoords

def globalShapeFunctions(xe, x):
    return localShapeFunctions(global2localCoords(xe, x))

def advection_stiffness(xe, u):
    output = np.zeros((3,3))
    dxNa = globalShapeFunctionDerivatives(xe)
    for i in range(3):
        for j in range(3):
            integrand = lambda x: globalShapeFunctions(xe, x)[i] * (u[0]*dxNa[0,j]
                                                                    + u[1]*dxNa[1,j])
            output[i,j] = globalQuadrature(xe,integrand)
    return output

def stiffness(xe, D, u):
    return D * diffusion_stiffness(xe) - advection_stiffness(xe, u)

def force(xe, S):
    output = np.zeros(3)
    for i in range(3):
        integrand = lambda x: S(x) * globalShapeFunctions(xe, x)[i]
        output[i] = globalQuadrature(xe, integrand)
    return output
        
def TwoDimStaticAdvDiffFESolver(S, u, D, resolution):
    '''
    S (funciton): source term
    u (array): wind velocity [ms^-1]
    D (float): diffusion coefficient [m^2s^-1]
    resolution (string): one of [1_25, 2_5, 5, 10, 20, 40]
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
            
    # Global stiffness matrix and force vector
    K = sp.lil_matrix((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        k_e = stiffness(nodes[:,IEN[e,:]], D, u)
        f_e = force(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
            if (A >= 0):
                F[A] += f_e[a]
    
    
    # Solve
    K = sp.csr_matrix(K)
    Psi_interior = sp.linalg.spsolve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise, Psi_A is homogeneous dirichlet
            Psi_A[n] = Psi_interior[ID[n]]
            
    
    #normalising
    Psi_A = 1/max(Psi_A)*Psi_A
    
    return nodes, IEN, southern_boarder, Psi_A

if __name__ == '__main__':
    
    def S_sotonfire(x):
        sigma = 10000
        return np.exp(-1/(2*sigma**2)*((x[0]-442365)**2 + (x[1]-115483)**2))

    north = np.array([0,1])
    
    directed_at_reading = np.array([473993 - 442365, 171625 - 115483])
    directed_at_reading = (1/np.linalg.norm(directed_at_reading)
                           *directed_at_reading)
    # Diffusion coefficient
    D = 10000
    
    # Reading coords
    reading = np.array([473993, 171625])
    
    # Max res north static plot
 
    max_res_data = TwoDimStaticAdvDiffFESolver(S_sotonfire, -10*directed_at_reading, D, '5')
    nodes, IEN, southern_boarder, psi = max_res_data
    
    plt.tripcolor(nodes[0,:], nodes[1,:], psi, triangles=IEN)
    plt.plot([442365],[115483],'x',c='r')
    plt.plot([473993], [171625],'x',c='pink')
    plt.axis('equal')
    plt.colorbar()
    plt.plot(nodes[0, southern_boarder], nodes[1,southern_boarder], '.', c='orange')
  
    plt.show()
