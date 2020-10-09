import numpy as np
import cv2
import matplotlib.pyplot as plt
#Import necessary functions
from cwcolombHelperFunctions import get_opts, matchPics, computeH_ransac, plotMatches
#Write script for Q4.2x

opts = get_opts()
opts.sigma = 0.1
opts.ratio = 0.6

# Load Images:
left_img = cv2.imread('../data/pano_left.jpg')
right_img = cv2.imread('../data/pano_right.jpg')

width = left_img.shape[1] + right_img.shape[1]
height = left_img.shape[0] + right_img.shape[0]

# Perform Feature Matching:
matches, locs1, locs2 = matchPics(left_img, right_img, opts)   
bestH2to1, inliers = computeH_ransac(locs1[matches[:,0],:], locs2[matches[:,1],:], opts)

# Plot Correspondences (for debugging)
plotMatches(left_img, right_img, matches, locs1, locs2)

# Make Panorama:
panorama = cv2.warpPerspective(left_img, np.linalg.inv(bestH2to1), (width, height))
panorama[0:right_img.shape[0], 0:right_img.shape[1]] = right_img

# Display Result:
plt.figure()
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.show()