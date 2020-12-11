# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    
    # Perform SVD:
    U, s, Vh = np.linalg.svd(I, full_matrices=False)
    
    # Constrain Rank:
    U_hat = U[:,0:3]
    V_hat = Vh[0:3,:]
    S_hat_half = np.diag(s[0:3]**0.5)

    # Factorize!:
    L = (U_hat @ S_hat_half).T
    B = S_hat_half @ V_hat
    
    # Normalize L (make unit vector) so it's formatted like the given L:
    L = L / np.linalg.norm(L, axis=0)

    return B, L

if __name__ == "__main__":

    # Put your main code here
    
    run_q2b = True
    if run_q2b:
        I, L0, s = loadData()
        B, L = estimatePseudonormalsUncalibrated(I)
        albedos, normals = estimateAlbedosNormals(B)
        albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
        
    run_q2c = True
    if run_q2c:
        I, L0, s = loadData()
        B, L = estimatePseudonormalsUncalibrated(I)
        print(np.around(L0,4))
        print(np.around(L,4))
        # Display matrices as images to ease grading:
        vmin = np.min((L,L0))
        vmax = np.max((L,L0))
        plt.figure()
        plt.imshow(L0, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title('$L_0$')
        plt.show()
        plt.figure()
        plt.imshow(L, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title('$\hat{L}$')
        plt.show()
        
    run_q2d = True
    if run_q2d:
        I, L0, s = loadData()
        B, L = estimatePseudonormalsUncalibrated(I)
        albedos, normals = estimateAlbedosNormals(B)
        surface = estimateShape(normals, s)
        plotSurface(surface)
        
    run_q2e = True
    if run_q2e:
        I, L0, s = loadData()
        B, L = estimatePseudonormalsUncalibrated(I)
        B_int = enforceIntegrability(B, s)
        albedos, normals = estimateAlbedosNormals(B_int)
        surface = estimateShape(normals, s)
        plotSurface(surface)
        
    run_q2f = True
    if run_q2f:
        I, L0, s = loadData()
        B, L = estimatePseudonormalsUncalibrated(I)
        
        # Create and apply a G while varying one of the parameters:
        params_vals = np.asarray([ # which values to try for each of the params (each column is mu, nu, lambda)
            [0, 0, 1],
            [-2, -2, 0.05],
            [2, 2, 10]
        ])
        for i in range(params_vals.shape[1]):
            for val in range(params_vals.shape[0]):
                params = params_vals[0,:].copy()
                params[i] = params_vals[val, i]
                
                G = np.asarray([
                    [1, 0, 0],
                    [0, 1, 0],
                    params
                ])
                
                B_int = enforceIntegrability(B, s)
                B_bas = np.linalg.inv(G).T @ B_int
                albedos, normals = estimateAlbedosNormals(B_bas)
                
                displayAlbedosNormals(albedos, normals, s)
                
                surface = estimateShape(normals, s)
                plt.title(r'$\mu$={}, $\nu$={}, $\lambda$={}'.format(params[0], params[1], params[2]))
                plotSurface(surface)
                pass