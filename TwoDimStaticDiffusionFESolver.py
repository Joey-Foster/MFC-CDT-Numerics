import numpy as np
import matplotlib.pyplot as plt


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

def test_local2globalCoords():
    
    default = {
                "xe": np.array([[0, 0, 1],
                               [0, 1, 0]]),
                "ans": np.array([0.5, 0.5])
              }
    
    translated = {
                "xe": np.array([[1, 1, 2],
                               [0, 1, 0]]),
                "ans": np.array([1.5, 0.5])
                 }
    
    scaled = {
                "xe": np.array([[0, 0, 2],
                               [0, 2, 0]]),
                "ans": np.array([1, 1])
             }
    
    rotated = {
                "xe": np.array([[1, 0, 1],
                               [1, 1, 0]]),
                "ans": np.array([0.5, 0.5])
              }
    
    xi = np.array([0.5, 0.5])
    
    for t in [default, translated, scaled, rotated]:
        assert np.allclose(local2globalCoords(t["xe"], xi),t["ans"]), f"element\n {t['xe']} is broken"
  

def jacobian(xe):
    Nprime = localShapeFunctionDerivatives()
    output = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            output[i,j] = xe[i,0]*Nprime[j,0] + xe[i,1]*Nprime[j,1] + xe[i,2]*Nprime[j,2]
    return output

def test_jacobian():
      
    default = {
                "xe": np.array([[0, 0, 1],
                               [0, 1, 0]]),
                "ans": np.array([[0, 1],
                                 [1, 0]])
              }
    
    translated = {
                "xe": np.array([[1, 1, 2],
                               [0, 1, 0]]),
                "ans": np.array([[0, 1],
                                 [1, 0]])
                 }
    
    scaled = {
                "xe": np.array([[0, 0, 2],
                               [0, 2, 0]]),
                "ans": np.array([[0, 2],
                                 [2, 0]])
             }
    
    rotated = {
                "xe": np.array([[1, 0, 1],
                               [1, 1, 0]]),
                "ans": np.array([[-1, 0],
                                 [0, -1]])
              }
    
    for t in [default, translated, scaled, rotated]:
        assert np.allclose(jacobian(t["xe"]),t["ans"]), f"element\n {t['xe']} is broken"


def globalShapeFunctionDerivatives(xe):
    return np.linalg.inv(jacobian(xe)) @ localShapeFunctionDerivatives()
    
def localQuadrature(psi):
    quadrature = 0
    
    #Gauss-quadrature evaluation points (2nd order accurate approximation)
    xis = 1/6 * np.array([[1, 1, 4],
                          [1, 4, 1]]) 
    for i in range(3):
        quadrature += 1/6 * psi(xis[:,i])
    return quadrature

def test_localQuadtrature():
    
    constant = {
             "psi": lambda x: 1,
             "ans": 0.5 
             } 
    
    linearx = {
             "psi": lambda x: 6*x[0],
             "ans": 1,
             }
    
    lineary = {
             "psi": lambda x: x[1],
             "ans": 1/6,
             }
    
    product = {
             "psi": lambda x: x[0]*x[1],
             "ans": 1/24,
             }
    for t in [constant, linearx, lineary, product]:
        assert np.allclose(localQuadrature(t["psi"]),t["ans"]), f"function\n {t['psi']} is broken"

def globalQuadrature(xe, phi):
    detJ = np.linalg.det(jacobian(xe))
    integrand = lambda xi: abs(detJ)*phi(local2globalCoords(xe, xi))
    return localQuadrature(integrand)    

def test_globalQuadrature():   
    
    translated_linear = {
             "xe": np.array([[1, 1, 2],
                            [0, 1, 0]]),
             "phi": lambda x: 3*x[0],
             "ans": 2,
                        }
    
    translated_product = {
             "xe": np.array([[1, 1, 2],
                            [0, 1, 0]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 5/24,
                         }
    
    scaled_linear = {
             "xe": np.array([[0, 0, 2],
                           [0, 2, 0]]),
             "phi": lambda x: 3*x[0],
             "ans": 4,
                     }
    
    scaled_product = {
             "xe": np.array([[0, 0, 2],
                           [0, 2, 0]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 2/3,
                     }
    
    rotated_linear = {
             "xe": np.array([[1, 0, 1],
                           [1, 1, 0]]),
             "phi": lambda x: 3*x[0],
             "ans": 1,
                     }
    
    rotated_product = {
             "xe": np.array([[1, 0, 1],
                           [1, 1, 0]]),
             "phi": lambda x: x[0]*x[1],
             "ans": 5/24,
                     }

    for t in [translated_linear, translated_product, scaled_linear, scaled_product,
              rotated_linear, rotated_product]:
        assert np.allclose(globalQuadrature(t["xe"],t["phi"]),t["ans"]), f"element\n {t['xe']} and/or function {t['phi']} is broken"
   
def stiffness(xe):
    output = np.zeros((3,3))
    dxNa = globalShapeFunctionDerivatives(xe)
    for i in range(3):
        for j in range(3):
            phi = lambda x: dxNa[0,i]*dxNa[0,j] + dxNa[1,i]*dxNa[1,j]
            output[i,j] = globalQuadrature(xe, phi)
    return output

def test_stiffness():

    default = {
                "xe": np.array([[0, 0, 1],
                                [0, 1, 0]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
              }
    
    translated = {
                "xe": np.array([[1, 1, 2],
                                [0, 1, 0]]),
                "ans": np.array([[1, -0.5, -0.5],
                                 [-0.5, 0.5, 0],
                                 [-0.5, 0, 0.5]])
                 }
    
    scaled = {
                "xe": np.array([[0, 0, 2],
                                [0, 2, 0]]),
                "ans": np.array([[2, -1, -1],
                                 [-1, 1, 0],
                                 [-1, 0, 1]])
            }
    rotated = {
                "xe": np.array([[1, 0, 1],
                                [1, 1, 0]]),
                "ans": np.array([[-1, 0.5, 0.5],
                                 [0.5, -0.5, 0],
                                 [0.5, 0, -0.5]])
              }
    
    for t in [default, translated, scaled, rotated]:
        assert np.allclose(stiffness(t["xe"]),t["ans"]), f"element\n {t['xe']} is broken"


def force(xe, S):
    '''
    Cannot just pass globalQuadrature() as expression for globalShapeFunctions()
    is required but not practically obtainable

    '''
    output = np.zeros(3)
    detJ = np.linalg.det(jacobian(xe))
    
    for i in range(3):
        integrand = lambda xi: abs(detJ) * S(local2globalCoords(xe, xi)) * localShapeFunctions(xi)[i]
        output[i] = localQuadrature(integrand)
    return output

def test_force():
    '''
    I need to compute some integrals over trianlges by hand to check this
    '''
    pass

def generate_2d_grid(Nx, alpha, beta):
    '''
    Written by Ian
    '''
    Nnodes = Nx+1
    x = np.linspace(0, 1, Nnodes)
    y = np.linspace(0, 1, Nnodes)
    X, Y = np.meshgrid(x,y)
    nodes = np.zeros((Nnodes**2,2))
    nodes[:,0] = X.ravel()
    nodes[:,1] = Y.ravel()
    ID = np.zeros(len(nodes), dtype=np.int64)
    boundaries = dict()  # Will hold the boundary values
    n_eq = 0
    for nID in range(len(nodes)):
        if np.allclose(nodes[nID, 0], 0):
            ID[nID] = -1
            boundaries[nID] = alpha  # Dirichlet BC
        else:
            ID[nID] = n_eq
            n_eq += 1
            if ( (np.allclose(nodes[nID, 1], 0)) or 
                 (np.allclose(nodes[nID, 0], 1)) or 
                 (np.allclose(nodes[nID, 1], 1)) ):
                boundaries[nID] = beta # Neumann BC
    IEN = np.zeros((2*Nx**2, 3), dtype=np.int64)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2*i+2*j*Nx  , :] = (i+j*Nnodes, 
                                    i+1+j*Nnodes, 
                                    i+(j+1)*Nnodes)
            IEN[2*i+1+2*j*Nx, :] = (i+1+j*Nnodes, 
                                    i+1+(j+1)*Nnodes, 
                                    i+(j+1)*Nnodes)
    return nodes, IEN, ID, boundaries

def TwoDimStaticDiffusionFESolver(Ne, S, alpha, beta):
    
    nodes, IEN, ID, boundaries = generate_2d_grid(Ne, alpha, beta)
    
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
    K = np.zeros((N_equations, N_equations))
    F = np.zeros((N_equations,))
    # Loop over elements
    for e in range(N_elements):
        k_e = stiffness(nodes[:,IEN[e,:]])
        f_e = force(nodes[:,IEN[e,:]], S)
        for a in range(3):
            A = LM[a, e]
            for b in range(3):
                B = LM[b, e]
                if (A >= 0) and (B >= 0):
                    K[A, B] += k_e[a, b]
                if A==0:
                    print(f'B={B}, e={e}, {k_e}')
            if (A >= 0):
                F[A] += f_e[a]
            
    print(f'K={K[:2,:]}')
    print(f'F={F}')
    
    # Solve
    Psi_interior = np.linalg.solve(K, F)
    Psi_A = np.zeros(N_nodes)
    for n in range(N_nodes):
        if ID[n] >= 0: # Otherwise, need to update Psi_A with dirichlet info - TODO
            Psi_A[n] = Psi_interior[ID[n]]
    return nodes, IEN, Psi_A
        
if __name__ == '__main__':
    
    pass
    #nodes, IEN, psi = TwoDimStaticDiffusionFESolver(5, lambda x:1, 0, 0)
    # plt.tripcolor(nodes[0,:], nodes[1,:], psi, triangles=IEN)
    
    
    #x = nodes[0,:]
    #plt.plot(x, psi, 'xk')
    # xe = np.array([[0, 0, 2],
    #               [0, 1, 0]])
    # print(f'K={stiffness(xe)}, F={force(xe,lambda x:1)}')
    