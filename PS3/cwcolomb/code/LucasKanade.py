import numpy as np
import numpy.linalg
import warnings
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Extract Grid Data:
    rect = np.asarray(rect).reshape(4) # standardize formatting
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    
    # Precompute meshgrid (base):
    x, y = np.meshgrid(np.arange(x1,(x2+1)), np.arange(y1,(y2+1)))
    x = x.ravel()
    y = y.ravel()
    x_int = np.round(x).astype(np.int32) # round to ints for use as indices
    y_int = np.round(y).astype(np.int32)
    
    D = x.size
    
    # Precompute Spline Functor:
    It1_spline = RectBivariateSpline(range(It1.shape[1]),range(It1.shape[0]), It1.T)

    # Solve with Lucas-Kanade:
    p = p0
    
    iter_count = 0
    while True:
        A = np.zeros((D,2))
        b = np.zeros((D,1))
        
        xp = x + p[0] # all (xi+px)
        yp = y + p[1] # all (yi+py)
        A[:,0] = It1_spline(xp,yp, dx=1, grid=False) # Ix
        A[:,1] = It1_spline(xp,yp, dy=1, grid=False)# Iy
        b = It[y_int,x_int] - It1_spline(xp,yp, grid=False)
        
        eigs, _ = np.linalg.eig(A.T@A)
        if not np.all(np.abs(eigs / np.max(eigs)) > 1e-6):
            warnings.warn("A.T@A is singular or ill-conditioned with eigs: {}".format(eigs))
            
        # Update:
        Dp = np.linalg.inv(A.T@A) @ A.T @ b
        p = p + Dp
        
        # Check Termination Conditions:
        iter_count = iter_count + 1
        if np.sum(Dp**2) < threshold or iter_count > num_iters:
            break

    return p
