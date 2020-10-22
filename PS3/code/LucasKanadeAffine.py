import numpy as np
import numpy.linalg
import warnings
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    def getM(p):
        """Calculate the Homogeneous Warp Matrix M associated with the given p."""
        return np.asarray([
            [1+p[0], p[1], p[2]],
            [p[3], 1+p[4], p[5]],
            [0, 0, 1]
        ]);
    
    def getDWDp(x):
        """Calculate the Jacobian of W w.r.t. for a given position x=[x;y]."""
        return np.asarray([
            [x[0], x[1], 1, 0, 0, 0],
            [0, 0, 0, x[0], x[1], 1]
        ]);
    
    # Precompute meshgrid (base):
    x, y = np.meshgrid(np.arange(It.shape[1]), np.arange(It.shape[0]))
    x = x.ravel()
    y = y.ravel()
    X = np.vstack((x,y)) # Matrix of coords of all pixels in image. row 0 is x positions, row 1 is y
    Xh = np.vstack((X,np.ones(X.shape[1]))) # X in homogeneous form (row of ones at bottom)
    X_int = np.round(X).astype(np.int32) # round to ints for use as indices
    
    # Precompute Spline Functor:
    It1_spline = RectBivariateSpline(range(It1.shape[1]),range(It1.shape[0]), It1.T)

    # Solve with Lucas-Kanade:
    p = np.zeros(6)
    
    iter_count = 0
    while True:
        # Warp image coords:
        Xh_warped = getM(p) @ Xh # equivalent to \tilde{X}
        
        # Find indices of any coordinates which land out of bounds after warp:
        x_out = (Xh_warped[0,:] >= It.shape[1]) | (Xh_warped[0,:] < 0) # x index less than 0 or beyond width
        y_out = (Xh_warped[1,:] >= It.shape[0]) | (Xh_warped[1,:] < 0) # y index less than 0 or beyond height
        # Boolean list indicating all columns in Xh_warped whose coordinates represent a pixel which was warped out of bounds:
        warped_out_of_bounds = x_out | y_out
        in_bounds = ~warped_out_of_bounds
        N_in = np.count_nonzero(in_bounds) # number of common points which are in bounds after warping
        
        # Only compute error for the coordinates which are inbounds once warped:
        # (only considering pixels in te region common to It and the warped It1)
        Xh_warped_in = Xh_warped[:,in_bounds] # All warped Xh coordinates which are still in bounds
        Xh_in = Xh[:,in_bounds] # The coordinates Xh of all unwarped pixels which correspond to the post-warp pixels which are still in bounds

        # Compute Error Image (Template It - Warped It1)
        Ierr = It[X_int[1,in_bounds], X_int[0,in_bounds]] - It1_spline(Xh_in[0,:],Xh_in[1,:], grid=False)
        Ierr.shape = (-1,1)

        # Compute Steepest Descent Images (gradI(x')*dW/dp(x)) for every point:
        gradI = np.zeros((N_in, 2))
        gradI[:,0] = It1_spline(Xh_warped_in[0,:],Xh_warped_in[1,:], dx=1, grid=False) # Ix for all points in Xh_warped
        gradI[:,1] = It1_spline(Xh_warped_in[0,:],Xh_warped_in[1,:], dy=1, grid=False) # Iy for all points in Xh_warped
        steepest = np.zeros((N_in,6))
        for i in range(N_in):
            steepest[i,:] = gradI[i,:] @ getDWDp(Xh_in[:2,i])
            
        # Compute Hessian:
        H = steepest.T @ steepest
        
        # Check Eigenvalues of Hessian to See if there are Going to be Any Issues Inverting:
        eigs, _ = np.linalg.eig(H)
        if not np.all(np.abs(eigs / np.max(eigs)) > 1e-9):
            warnings.warn("H is singular or ill-conditioned with eigs: {}".format(eigs))
        
        # Find Least Squares Solution:
        Dp = np.linalg.inv(H) @ steepest.T @ Ierr
        
        # Update:
        p = p + Dp
        
        # Check Termination Conditions:
        iter_count = iter_count + 1
        if np.sum(Dp**2) < threshold or iter_count > num_iters:
            break

    return getM(p)