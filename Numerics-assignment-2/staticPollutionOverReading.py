import numpy as np
import matplotlib.pyplot as plt
from TwoDimStaticAdvDiffFESolver import localShapeFunctions, global2localCoords, S_sotonfire, TwoDimStaticAdvDiffFESolver

def elementValidityChecker(IEN, element):
    # np.sort() is necessary as the order of the nodes in 'element' need not be
    # the same as they appear in IEN
    IENindex = np.where(np.all(np.sort(IEN,axis=1) == np.sort(element), axis=1))[0]
    
    if len(IENindex) == 0:
        # With the provided grids, this only occurs on the 2.5k resolution
        print('Triangle does not exist in IEN. If things go wrong, check here')
        existsinIEN = False
    else:
        IENindex = IENindex[0]
        existsinIEN = True
    return IENindex, existsinIEN

def nearestElement2Coords(nodes, IEN, coords):
    '''
    nodes should be 2xN
    coords should be 2x1
    '''
    def locateTriangle(coords):
        error = np.zeros_like(nodes[0,:])
        for i in range(len(error)):
            error[i] = np.sqrt((nodes[0,i]-coords[0])**2 + (nodes[1,i]-coords[1])**2)
    
        tol = 18000
        test = np.where(error < tol)[0]
        while len(test) > 3:
            tol -= 10
            test = np.where(error < tol)[0]
            
        if len(test) < 3:
            raise Exception('Degenerate triangle found!')
            
        return test
    
    candidate_triangle = locateTriangle(coords)
    IENindex, existsinIEN = elementValidityChecker(IEN, candidate_triangle)
    
    if existsinIEN:
        #Use the order of the nodes as the appear in IEN, not 'candidate_triangle' itself
        matching_row = IEN[IENindex,:]
        return matching_row
    else:
        return candidate_triangle
    
def pollutionExtractor(psi, nodes, IEN, coords):
    #find nodes of the triangle that contains reading
    elementnodes = nearestElement2Coords(nodes, IEN, coords)
    #get the coords of the nodes of that triangle
    xe = nodes[:,elementnodes]
    
    #find the local coords of reading in that triangle
    xi = global2localCoords(xe, coords)
    
    N = localShapeFunctions(xi)
    #use basis function representation to determine the value of psi over reading
    pollution = np.dot(psi[elementnodes], N)
    return pollution

def convergence1(coords, u0, D):
    '''
    With current implementation:
    trendline with all points goes like x^1.7
    killing the last point gives 2.47
    killing the last 2 points gives 3.11
    '''
    maxres_nodes, maxres_IEN, maxres_southern_boarder, maxres_psi = TwoDimStaticAdvDiffFESolver(S_sotonfire, 10, 10000, '1_25')
    maxres_pollution = pollutionExtractor(maxres_psi, maxres_nodes, maxres_IEN, coords)
    print('computed max res')
   
    error = np.zeros(5)
    for i, res in enumerate(['2_5','5','10','20','40']):
        nodes, IEN, southern_boarder, psi = TwoDimStaticAdvDiffFESolver(S_sotonfire, u0, D, res)
        pollution = pollutionExtractor(psi, nodes, IEN, coords)
        error[i] = abs(pollution - maxres_pollution)
        print(f'computed error for {res}')
        
    x = np.array([2500, 5000, 10000, 20000, 40000])
    
    polyfit_coeffs = np.polyfit(np.log(x[:-1]),np.log(error[:-1]),1) 

    trendline = lambda data,x: np.poly1d(data)(x)
    
    plt.loglog(x, error, 'xk')
    plt.loglog(x, np.exp(trendline(polyfit_coeffs,np.log(x))),'-r', label=rf'$\propto x^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel('average element side length')
    plt.ylabel('Error relative to max res soln')
    plt.legend()
    plt.grid()
    plt.title(f'Relative Error in static case with u = {u0}, D = {D}')
    
    plt.show()
    
def convergence2(coords, u0, D):
    
    soln_N  = TwoDimStaticAdvDiffFESolver(S_sotonfire, u0, D, '5')
    print('computed 5k')
    soln_2N = TwoDimStaticAdvDiffFESolver(S_sotonfire, u0, D, '2_5')
    print('computed 2.5k')
    soln_4N = TwoDimStaticAdvDiffFESolver(S_sotonfire, u0, D, '1_25')
    print('computed 1.25k')
    
    y_N = pollutionExtractor(soln_N[3], soln_N[0], soln_N[1], coords)
    y_2N = pollutionExtractor(soln_2N[3], soln_2N[0], soln_2N[1], coords)
    y_4N = pollutionExtractor(soln_4N[3], soln_4N[0], soln_4N[1], coords)
    
    y2N_N = abs(y_2N - y_N)
    y4N_2N = abs(y_4N - y_2N)
    
    s = np.log2(y2N_N/y4N_2N)
    
    error = y2N_N/(1-2**(-s))

    return s, error
    
#%%
if __name__ == '__main__':

    nodes, IEN, southern_boarder, psi = TwoDimStaticAdvDiffFESolver(S_sotonfire, 10, 10000, '5')
    
    #normalising
    psi = 1/max(psi)*psi
    
    plt.tripcolor(nodes[0,:], nodes[1,:], psi, triangles=IEN)
    plt.plot([442365],[115483],'x',c='r')
    plt.plot([473993], [171625],'x',c='pink')
    plt.axis('equal')
    plt.colorbar()
    
    plt.plot(nodes[0, southern_boarder], nodes[1,southern_boarder], '.', c='orange')
    
    plt.show()
    
    reading = np.array([473993, 171625])
    ans = pollutionExtractor(psi, nodes, IEN, reading)   
    print(ans)
    
    #convergence1(reading, 10, 10000)
    order, error = convergence2(reading, 10, 10000)
    print(f'theoretical convergence order = {order}\n theoretical error = {error}')