import numpy as np
import matplotlib.pyplot as plt
from TwoDimStaticAdvDiffFESolver import localShapeFunctions, global2localCoords, S_sotonfire, TwoDimStaticAdvDiffFESolver

def elementValidityChecker(IEN, element):
    # np.sort() is necessary as the order of the nodes in 'element' need not be
    # the same as they appear in IEN
    IENindex = np.where(np.all(np.sort(IEN,axis=1) == np.sort(element), axis=1))[0]
    
    if len(IENindex) == 0:
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