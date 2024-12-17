import numpy as np
import matplotlib.pyplot as plt
from TwoDimStaticAdvDiffFESolver import localShapeFunctions, global2localCoords, TwoDimStaticAdvDiffFESolver

def S_sotonfire(x):
    sigma = 10000
    return np.exp(-1/(2*sigma**2)*((x[0]-442365)**2 + (x[1]-115483)**2))

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
    
    Argument for the validity of this construction: 
        
    I am constructing a large circle centred at 'coords' and computing the number 
    of nodes within the circle, then shrinking the radius until there are exactly 
    3 nodes within it.
    In most cases, these 3 nodes are precisely the nodes of the triangle that 
    contains 'coords'. However, it can be shown geometrically that this is not 
    always the case. Nevertheless, the local coordinate construction is valid 
    outside the confines of the reference triangle as it maps all of mathbb{R}^2
    in principle. (We only care about xi_1,2 inside the reference triangle for 
    the purposes of solving, but they exist everywhere anyway). Therefore, 
    relative to a "close enough" triangle, the basis-function construction seen
    in pollutionExtractor() still holds.
    The purpose of elementValidityChecker() is to see if the identified triangle
    (of which the searching radius now defines its circumcircle) is actually
    one of the triangles in IEN. If so, then it relabels the order of the nodes 
    found by the search to match the order they appear in the IEN, for consistency
    reasons. Since for at least one resolution, this test fails (yet I proceed
    anyway), this check is essentially superfluous but remains for sanity reasons.
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
    #find nodes of the triangle that contains 'coords'
    elementnodes = nearestElement2Coords(nodes, IEN, coords)
    #get the coords of the nodes of that triangle
    xe = nodes[:,elementnodes]
    
    #find the local coords of 'coords' in that triangle
    xi = global2localCoords(xe, coords)
    
    N = localShapeFunctions(xi)
    #use basis function representation to determine the value of psi at 'coords'
    pollution = np.dot(psi[elementnodes], N)
    return pollution

def convergence(max_res_data, coords, u, D, figsize, filename1, filename2):
   
    maxres_nodes, maxres_IEN, maxres_southern_boarder, maxres_psi = max_res_data
    
    maxres_pollution = pollutionExtractor(maxres_psi, maxres_nodes, maxres_IEN, coords)
    y_1_25k = maxres_pollution
   
    error = np.zeros(5)
    Ns = np.zeros(5)
    for i, res in enumerate(['2_5','5','10','20','40']):
        nodes, IEN, southern_boarder, psi = TwoDimStaticAdvDiffFESolver(S_sotonfire, u, D, res)
        Ns[i] = len(nodes[0,:] - len(southern_boarder))**(1/2)
        pollution = pollutionExtractor(psi, nodes, IEN, coords)
        if res == '2_5':
            y_2_5k = pollution
        elif res == '5':
            y_5k = pollution
        elif res == '10':
            y_10k = pollution
        elif res == '20':
            y_20k = pollution
        else:
            y_40k = pollution
        error[i] = abs(pollution - maxres_pollution)
        print(f'computed error for {res}')
    
    polyfit_coeffs = np.polyfit(np.log(Ns),np.log(error),1) 

    trendline = lambda data,x: np.poly1d(data)(x)
    
    plt.figure(figsize=figsize)
    plt.loglog(Ns, error, 'xk')
    plt.loglog(Ns, np.exp(trendline(polyfit_coeffs,np.log(Ns))),'-r', 
               label=rf'$\propto N^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel(r'Degrees of freedom $\sim \sqrt{\mathrm{Number\ of\ nodes}}$')
    plt.ylabel('Error relative to max res soln')
    plt.legend()
    plt.grid()
    plt.title(f'Relative Error in static case with u = [{abs(u[0]):.2f}, {abs(u[1]):.2f}], D = {D}')
    plt.savefig(f'{filename1}.pdf')
    
    x = np.array([2500, 5000, 10000, 20000, 40000])
    polyfit_coeffs = np.polyfit(np.log(x),np.log(error),1) 
    
    plt.figure(figsize=figsize)
    plt.loglog(x, error, 'xk')
    plt.loglog(x, np.exp(trendline(polyfit_coeffs,np.log(x))),'-r', 
               label=rf'$\propto x^{{{polyfit_coeffs[0]:.2f}}}$')
    plt.xlabel('Canonical triangle side length (m)')
    plt.ylabel('Error relative to max res soln')
    plt.legend()
    plt.grid()
    plt.title(f'Relative Error in static case with u = [{abs(u[0]):.2f}, {abs(u[1]):.2f}], D = {D}')
    plt.savefig(f'{filename2}.pdf')
    
    plt.show()
    
    ###########################################################################
    
    def theoretic_convergence(y_N, y_2N, y_4N):
    
        y2N_N = abs(y_2N - y_N)
        y4N_2N = abs(y_4N - y_2N)
        
        s = np.log2(y2N_N/y4N_2N)
        
        abs_error = y2N_N/(1-2**(-s))
        rel_error = abs_error/y_N * 100

        return s, abs_error, rel_error
    
    ss = np.zeros(4)
    abs_errors = np.zeros(4)
    rel_errors = np.zeros(4)
    ys = np.array([[y_5k, y_2_5k, y_1_25k],
                   [y_10k, y_5k, y_2_5k],
                   [y_20k, y_10k, y_5k],
                   [y_40k, y_20k, y_10k]])
    
    for i, y in enumerate(ys):
        ss[i], abs_errors[i], rel_errors[i] = theoretic_convergence(y[0], y[1], y[2])
        
    print('##########################################')
    print(f'theoretical convergence order = {ss}\n'
          f'theoretical absolute error = {abs_errors}\n'
          f'theoretical relative error = {rel_errors}%\n'
          f'mean order = {np.mean(ss)}\n'
          f'mean abs error = {np.mean(abs_errors)}\n'
          f'mean rel error = {np.mean(rel_errors)}')

    