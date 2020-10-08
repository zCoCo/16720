import numpy as np
import cv2
import skimage.io 
import skimage.color
import matplotlib.pyplot as plt
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

opts = get_opts()

# Write script for Q2.2.4
def q224():
    # Load images:
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    
    # Resize imposed template (hp_cover) to match size of matched template (cv_cover)
    # (Fix from Q2.2.4.4):
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    
    # Compute Homography:
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    
    bestH2to1, inliers = computeH_ransac(locs1[matches[:,0],:], locs2[matches[:,1],:], opts)
    
    # Composite Images:
    composite = compositeH(bestH2to1, hp_cover, cv_desk)
    
    # Display Result:
    plt.figure()
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.show()
    
if opts.c2_testing:
    # Profile if testing
    import timeit
    print(timeit.timeit(q224))
else:
    q224()