import numpy as np
import numpy.linalg
import warnings
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """
    
    # put your implementation here
    
    # Note: For reference, this implementation uses the inverse compositional 
    # algorithm as described in Section 2 of the paper referenced by the 
    # assignment (https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2003_3/baker_simon_2003_3.pdf)
    def getM(p):
        """Calculate the Homogeneous Warp Matrix M associated with the given p."""
        return np.asarray([
            [1+p[0], p[1], p[2]],
            [p[3], 1+p[4], p[5]],
            [0, 0, 1]
        ], dtype=float);
    
    # Precompute meshgrid (base):
    x, y = np.meshgrid(np.arange(It.shape[1]), np.arange(It.shape[0]))
    x = x.ravel()
    y = y.ravel()
    X = np.vstack((x,y)) # Matrix of coords of all pixels in image. row 0 is x positions, row 1 is y
    Xh = np.vstack((X,np.ones(X.shape[1]))) # X in homogeneous form (row of ones at bottom)
    X_int = np.round(X).astype(np.int32) # round to ints for use as indices
    N_pts = X.shape[1] # Number of points
    
    # Some nice helpers:
    ones = Xh[2]; # array of ones of size Npts
    zeros = 0*ones; # array of zeros of size Npts
    
    # Precompute Spline Functors:
    It_spline = RectBivariateSpline(range(It.shape[1]),range(It.shape[0]), It.T)
    It1_spline = RectBivariateSpline(range(It1.shape[1]),range(It1.shape[0]), It1.T)

    # Precompute Template Gradient:
    gradI = np.zeros((1, 2, N_pts))
    gradI[0,0,:] = It_spline(Xh[0,:],Xh[1,:], dx=1, grid=False) # Ix for all points in Xh_warped
    gradI[0,1,:] = It_spline(Xh[0,:],Xh[1,:], dy=1, grid=False) # Iy for all points in Xh_warped

    # Precompute Jacobian dWdp at (x,0):
    dWdp = np.asarray([
        [Xh[0], Xh[1], ones, zeros, zeros, zeros],
        [zeros, zeros, zeros, Xh[0], Xh[1], ones]
    ]);
    
    # Precompute Steepest Descent Images:
    # gradI and dWdp are 1*2*N_pts and 2*6*N_pts 3D stacks of the gradI and 
    # dWdp matrices corresponding to each point in Xh.
    # This einsum effectively loops through that third dimensions, computes 
    # gradI[:,:,i]@dWdp[:,:,i], and stacks up the results in a 1*6*N_pts matrix
    # (but *much* more efficiently).
    steepest = np.einsum('mnr,ndr->mdr', gradI, dWdp)
    steepest = steepest.T.reshape((N_pts,6)) # Reshape to more easily work with subsequent linalg operations

    # Precompute Inverse Hessian:
    H = steepest.T @ steepest
    # Check Eigenvalues of Hessian to See if there are Going to be Any Issues Inverting:
    eigs, _ = np.linalg.eig(H)
    if not np.all(np.abs(eigs / np.max(eigs)) > 1e-9):
        warnings.warn("H is singular or ill-conditioned with eigs: {}".format(eigs))
    H_inv = np.linalg.inv(H)

    # Solve with Lucas-Kanade:
    p = np.zeros(6)
    M = getM(p)
    
    iter_count = 0
    while True:
        # Warp image coords:
        Xh_warped = M @ Xh # equivalent to \tilde{X}
        
        # Find indices of any coordinates which land out of bounds after warp:
        x_out = (Xh_warped[0,:] >= It.shape[1]) | (Xh_warped[0,:] < 0) # x index less than 0 or beyond width
        y_out = (Xh_warped[1,:] >= It.shape[0]) | (Xh_warped[1,:] < 0) # y index less than 0 or beyond height
        # Boolean list indicating all columns in Xh_warped whose coordinates represent a pixel which was warped out of bounds:
        warped_out_of_bounds = x_out | y_out
        
        # Compute Error Image (Template It - Warped It1)
        Ierr = It1_spline(Xh[0,:],Xh[1,:], grid=False) - It[X_int[1,:], X_int[0,:]]
        Ierr[warped_out_of_bounds] = 0; # Set error for all out of bounds pixels to 0
        Ierr.shape = (-1,1)
        
        # Find Least Squares Solution:
        Dp = H_inv @ steepest.T @ Ierr
        DM = getM(Dp) # allowed according to Piazza @346
        
        # Update:
        M = M @ np.linalg.inv(DM)
        
        # Check Termination Conditions:
        iter_count = iter_count + 1
        if np.sum(Dp**2) < threshold or iter_count > num_iters:
            break

    return M