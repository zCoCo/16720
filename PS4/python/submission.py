"""
Homework4.
Replace 'pass' by your implementation.
"""

from helper import *
from util import *

import numpy as np
import warnings

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Prevent any surprises:
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)

    # Normalize coordinates (just scale as told in Piazza @453):
    pts1 = pts1/M
    pts2 = pts2/M
    
    # Convenience Naming:
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    A = np.asarray([
        x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, np.ones(x1.shape)
    ]).T
    
    """
    SVD Approach (works fine, just slower than eig. Keeping in case of issue with eig.)
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    if s.shape[0] > 8 and abs(s[8]) > 1e-6:
        warnings.warn("Min singular value should be approximately zero.")
    V = Vh.T
    F = V[:,-1] # Solution is eigenvector for smallest eigenvalue of A.T@A (last col of V)
    """
    D,V = np.linalg.eig(A.T@A)
    D = np.absolute(D) # only care about distance of e-values from 0
    i_min = np.argmin(D)
    # Don't worry about throwing this warning since we singularize F anyway:
    #if D[i_min] > 1e-3:
        #warnings.warn("Min singular value should be approximately zero; it was {}.".format(D[i_min]))
    F = V[:,i_min].reshape((3,3))
    
    # Singularize F:
    U, s, Vh = np.linalg.svd(F)
    s[-1] = 0
    F = U @ np.diag(s) @ Vh
    
    # Refine F (minimize geometric error):
    F = refineF(F,pts1,pts2)
    
    # Unscale:
    # Corrective Matrix
    T = np.asarray([
        [1/M,0,0],
        [0,1/M,0],
        [0,0,1],
    ])
    F = T.T @ F @ T
    
    # Normalize F:
    F = F / F[2,2]
    
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = K1.T @ F @ K2
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Prevent any surprises:
    pts1 = pts1.astype(float)
    pts2 = pts2.astype(float)
    
    ## Build A:
        
    # Convenience Notation:
    u1 = pts1[:,0]
    v1 = pts1[:,1]
    u2 = pts2[:,0]
    v2 = pts2[:,1]
    N = pts1.shape[0]
    
    ones = np.ones(v1.shape)
    zeros = np.zeros(ones.shape)
    
    # 3D Stack of 2D x_cross matrices (skew symmetric matrix for x_1i)
    X1 = np.asarray([
        [zeros, -ones, v1],
        [ones, zeros, -u1]#,
        #[-v1, u1, zeros]
    ])

    # Multiply each matrix in the (2x3xN) X1 stack by the (3x4) C1 matrix (*very* fast):
    A1 = np.einsum('mnr,nd->mdr', X1, C1)
    
    # 3D Stack of 2D x_cross matrices (skew symmetric matrix for x_2i)
    X2 = np.asarray([
        [zeros, -ones, v2],
        [ones, zeros, -u2]#,
        #[-v1, u1, zeros]
    ])

    # Multiply each matrix in the (2x3xN) X2 stack by the (3x4) C2 matrix
    A2 = np.einsum('mnr,nd->mdr', X2, C2)
    
    # Stack to get resulting stack of Ai matrices:
    A = np.vstack((A1,A2))
    
    ## Find "Homogeneous Least Squares Solution":
    err = 0
    P = np.zeros((N,4)) # Preallocate for speed
    for i in range(0,N):
        Ai = A[:,:,i]
        D,V = np.linalg.eig(Ai.T@Ai)
        D = np.absolute(D) # only care about distance of e-values from 0
        i_min = np.argmin(D)
        #if D[i_min] > 1e-3:
            #warnings.warn("Min singular value should be approximately zero; it was {}.".format(D[i_min]))
        P[i,:] = V[:,i_min]
        
    # Just this might work?:...
    #_,_,Vh = np.linalg.svd(A)
    #P = Vh[:,-1,:].T
    
    # Normalize:
    P = P / P[:,-1].reshape((-1,1))
    
    # Compute Error:
    err = 0
    for i in range(0,N):
        x1i = pts1[i,:]
        x2i = pts2[i,:]
        err = err + np.linalg.norm(x1i - (C1 @ P[i,:])[0:2])**2 + np.linalg.norm(x2i - (C2 @ P[i,:])[0:2])**2
    
    #print(P)
    #print(err)
    return P[:,0:-1], err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    window_size = 10
    x2 = np.zeros(x1.shape)
    y2 = np.zeros(y1.shape)
    
    # Convert Images to Grey Scale:
        
    # Gauss Blur Images:
        
    
    # Get Ref Window:
    ref_window = im[y1-window_size/2:y1+window_size/2,x1-window_size/2:x1+window_size/2,0]
    
    # Get Epipolar Line:
    line = F @ np.asarray([x1,y1,1]).reshape((-1,1))
    
    # Slide along line and compare windows to find best match:
    
    return x2,y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
