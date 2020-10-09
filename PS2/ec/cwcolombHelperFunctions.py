import numpy as np
import cv2
import warnings
import argparse
import numpy as np
import cv2
import skimage.color
import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature

PATCHWIDTH = 9

def briefMatch(desc1,desc2,ratio):

	matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)
	return matches


def plotMatches(im1,im2,matches,locs1,locs2):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	plt.axis('off')
	skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
	plt.show()
	return


def makeTestPattern(patchWidth, nbits):
	np.random.seed(0)
	compareX = patchWidth*patchWidth * np.random.random((nbits,1))
	compareX = np.floor(compareX).astype(int)
	np.random.seed(1)
	compareY = patchWidth*patchWidth * np.random.random((nbits,1))
	compareY = np.floor(compareY).astype(int)

	return (compareX, compareY)

def computePixel(img, idx1, idx2, width, center):
	halfWidth = width // 2
	col1 = idx1 % width - halfWidth
	row1 = idx1 // width - halfWidth
	col2 = idx2 % width - halfWidth
	row2 = idx2 // width - halfWidth
	return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computeBrief(img, locs):
	
	
	patchWidth = 9
	nbits = 256
	compareX, compareY = makeTestPattern(patchWidth,nbits)
	m, n = img.shape

	halfWidth = patchWidth//2
	
	locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
	desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])
	
	return desc, locs
	


def corner_detection(img, sigma):
	# fast method
	result_img = skimage.feature.corner_fast(img, n=PATCHWIDTH, threshold=sigma)
	locs = skimage.feature.corner_peaks(result_img, min_distance=1)
	return locs


def matchPics(I1, I2, opts):
    """
    Find feature correspondences between the two given images.
    
    I1, I2 : Images to match
    opts: input opts
    """
    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	

    #Convert Images to GrayScale
    gray1 = skimage.color.rgb2gray(I1)
    gray2 = skimage.color.rgb2gray(I2)
	
    #Detect Features in Both Images
    locs1 = corner_detection(gray1, sigma)
    locs2 = corner_detection(gray2, sigma)
	
    #Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(gray1, locs1)
    desc2, locs2 = computeBrief(gray2, locs2)

    #Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW2: Homography')

    ## Feature detection (requires tuning)
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='threshold for corner detection using FAST feature detector')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='ratio for BRIEF feature descriptor')

    ## Ransac (requires tuning)
    parser.add_argument('--max_iters', type=int, default=500,
                        help='the number of iterations to run RANSAC for')
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='the tolerance value for considering a point to be an inlier')

    ## Additional options (add your own hyperparameters here)
    parser.add_argument('--c2_testing', type=bool, default=False,
                        help='Whether this run is for testing.')
    parser.add_argument('--save_addr', type=str, default=None,
                        help='Address to export content to.')
    ##
    opts = parser.parse_args()

    return opts


def computeH(x1, x2):
    """
    Compute the homography between two sets of points.

    Q2.2.1
    """
    Npts = x1.shape[0]
    Npts2 = x2.shape[0]
    assert Npts==Npts2, "Number of points in each list must be equal"
    
    A = np.zeros((Npts*2,9))
    
    for i in range(Npts):
        A[i*2:(i+1)*2, :] = np.array([
            [-x2[i,0], -x2[i,1], -1, 0, 0, 0, x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0]],
            [0, 0, 0, -x2[i,0], -x2[i,1], -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1]],
        ])
        
        
    """
    SVD Approach (works fine, just slower than eig. Keeping in case of issue with eig.)
    _, s, Vh = np.linalg.svd(A, full_matrices=True)
    if s.shape[0] > 8 and abs(s[8]) > 1e-6:
        warnings.warn("Min singular value should be approximately zero.")
    V = Vh.T
    H2to1 = V[:,8] # Solution is eigenvector for smallest eigenvalue of A.T@A (last col of V)
    """
    D,V = np.linalg.eig(A.T@A)
    D = np.absolute(D) # only care about distance of e-values from 0
    i_min = np.argmin(D)
    if D[i_min] > 1e-3:
        warnings.warn("Min singular value should be approximately zero; it was {}.".format(D[i_min]))
    H2to1 = V[:,i_min]
    
    return H2to1.reshape((3,3))

def point_mean(x):
    """Compute mean of columns of given matrix x (faster than np.mean)."""
    Npts = x.shape[0]
    return np.sum(x[:, 0])/Npts, np.sum(x[:, 1])/Npts

def computeH_norm(x1, x2):
    """
    Compute the centroid of the points.
    
    Q2.2.2
    """
    #Shift the origin of the points to the centroid
    mean1 = point_mean(x1) # mean for each column (xvalues,yvalues) in x1
    mean2 = point_mean(x2) # mean for each column (xvalues,yvalues) in x2
    x1_centered = x1 - mean1
    x2_centered = x2 - mean2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max1 = np.max(np.absolute(x1_centered), axis=0)
    max2 = np.max(np.absolute(x2_centered), axis=0)
    if np.count_nonzero(max1==0) == 0:
        scale1 = 1.0 / max1 # scaling for each column (xvalues,yvalues) in x1
    else:
        warnings.warn("Zeros in maximum value for a column in x1. Ignoring scaling for that point collection.")
        scale1 = np.ones(max1.shape) # can't scale so don't. this likely won't generate the solution given this anomaly so this is fine.
    if np.count_nonzero(max2==0) == 0:
        scale2 = 1.0 / max2 # scaling for each column (xvalues,yvalues) in x2
    else:
        warnings.warn("Zeros in maximum value for a column in x2. Ignoring scaling for that point collection")
        scale2 = np.ones(max2.shape) # can't scale so don't. this likely won't generate the solution given this anomaly so this is fine.
    x1_normalized = x1_centered * scale1
    x2_normalized = x2_centered * scale2

	#Similarity transform 1
    T1 = np.array([
        [scale1[0], 0, -mean1[0]*scale1[0]],
        [0, scale1[1], -mean1[1]*scale1[1]],
        [0, 0, 1]
    ])

	#Similarity transform 2
    T2 = np.array([
        [scale2[0], 0, -mean2[0]*scale2[0]],
        [0, scale2[1], -mean2[1]*scale2[1]],
        [0, 0, 1]
    ])

	#Compute homography
    H2to1_norm = computeH(x1_normalized, x2_normalized)

	#Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1_norm @ T2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    """
    Compute the best fitting homography given a list of matching points.
    
    Q2.2.3
    """
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    # Copy and swap columns (from (y,x) to (x,y)) per Piazza
    pts1 = np.copy(locs1)
    pts1[:,[1,0]] = pts1[:,[0,1]]
    pts2 = np.copy(locs2)
    pts2[:,[1,0]] = pts2[:,[0,1]]

    Npts = pts1.shape[0]
    Npts2 = pts2.shape[0]
    assert Npts == Npts2, "Both locs should contain same number of points."

    # Make homogeneous ahead of time for use in RANSAC inlier testing:
    pts_h1 = pts1.T
    pts_h1 = np.append(pts_h1, np.ones((1,Npts)), axis=0)
    pts_h2 = pts2.T
    pts_h2 = np.append(pts_h2, np.ones((1,Npts)), axis=0)

    # Count inliers in model given by homography H:
    def count_inliers(H):
        estimate = H@pts_h2
        # Rescale third element to rehomogenize estimate (account for "up to scale" equivalence):
        if np.count_nonzero(estimate[2,:]) == estimate.shape[1]:
            estimate = estimate / estimate[2,:]
        inliers = np.linalg.norm(pts_h1 - estimate, axis=0) < inlier_tol # True if inlier
        num_inliers = np.count_nonzero(inliers)
        return num_inliers, inliers

    count = 1
    bestH2to1 = None
    best_inliers = None # Boolean vector with True for inliers during best homography fit
    best_inlier_count = -1
    while count < max_iters:
        # Select randomly the minimum number of points to fix model (8 points = 4 correspondences):
        sample_idx = np.random.choice(Npts, 4)
        sample1 = pts1[sample_idx,:]
        sample2 = pts2[sample_idx,:]
        
        # Solve for model (H):
        H2to1 = computeH_norm(sample1, sample2)
        
        # Count inliers:
        num_inliers, inliers = count_inliers(H2to1)
        
        # If best homography, save the model and the inliers (to try to build a better model later):
        if num_inliers > best_inlier_count:
            bestH2to1 = H2to1
            best_inliers = inliers
            best_inlier_count = num_inliers
        
        # If all points are (somehow) inliers, woohoo! Terminate early.
        if num_inliers == Npts:
            #terminate early
            break

        count = count + 1
        
    # Rebuild model using all inliers from best fit:
    H2to1 = computeH_norm(pts1[best_inliers,:], pts2[best_inliers,:])
    num_inliers, inliers = count_inliers(H2to1)
    # Make sure it's actually an improvement; if so, keep:
    if num_inliers > best_inlier_count:
        bestH2to1 = H2to1
        best_inliers = inliers

    inliers = (best_inliers*1) # convert from Bool to Int vector (per spec)
    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    """
    Create a composite image after warping the template image on top of the image using the homography
    """
    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    H1to2 = np.linalg.inv(H2to1)
	
    #Create mask of same size as template
    mask = np.ones(template.shape)

    #Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H1to2, (img.shape[1],img.shape[0]))
    #Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H1to2, (img.shape[1],img.shape[0]))
    
    #Use mask to combine the warped template and the image
    composite_img = img * (np.logical_not(warped_mask)) + warped_template
    
    return composite_img



