'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
from helper import *
from util import *
from submission import *

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Load Images:
im1 = cv2.cvtColor(cv2.imread('../data/im1.png'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('../data/im2.png'), cv2.COLOR_BGR2RGB)

# Load Point Correspondences:
corresp = np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# Compute Scale Size:
M = max(max(im1.shape),max(im2.shape))

# Compute Fundamental Matrix:
F = eightpoint(pts1, pts2, M)

# Compute Essential Matrix:
Ks = np.load('../data/intrinsics.npz')
K1 = Ks['K1']
K2 = Ks['K2']

E = essentialMatrix(F, K1, K2)

# Find M2:
M1 = np.hstack((np.eye(3), np.zeros((3,1))))
C1 = K1 @ M1

M2s = camera2(E)
C21 = K2 @ M2s[:,:,0]
C22 = K2 @ M2s[:,:,1]
C23 = K2 @ M2s[:,:,2]
C24 = K2 @ M2s[:,:,3]

N = pts1.shape[0]
Ps = np.zeros((N,3,4))
Ps[:,:,0], err1 = triangulate(C1, pts1, C21, pts2)
Ps[:,:,1], err2 = triangulate(C1, pts1, C22, pts2)
Ps[:,:,2], err3 = triangulate(C1, pts1, C23, pts2)
Ps[:,:,3], err4 = triangulate(C1, pts1, C24, pts2)

# Find the M2 Matrix which causes all P to be have z >= 0
n_behind_camera = np.zeros(4)
for i in range(0,4):
    n_behind_camera[i] = np.count_nonzero(Ps[:,2,i] < 0)

best_idx = np.argmin(n_behind_camera)
if n_behind_camera[best_idx] > 0:
    print("Best M should have no P.z behind camera (something went wrong).")
    
# Select that M2 matrix:
M2 = M2s[:,:,best_idx]
C2 = K2 @ M2
P = Ps[:,:,best_idx]

# Save Result:
np.savez('q3_3.npz', M2=M2, C2=C2, P=P)


