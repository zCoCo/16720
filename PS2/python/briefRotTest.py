import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

opts = get_opts()

# Q2.1.6
# Read the image and convert to grayscale, if necessary
image = cv2.imread('../data/cv_cover.jpg') # Note: matchPics takes care of grayscaling

hist_angles = np.arange(36)*10
hist_matches = np.zeros((1,36)) # preallocate for speed

for i in range(36):
	# Rotate Image
    angle = hist_angles[i]
    rotated = scipy.ndimage.rotate(image, angle)
	
	# Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(image, rotated, opts)
    
	# Update histogram
    hist_matches[i] = matches.shape[0]
    
    # Display matching at three orientations:
    if i < 36 and not i % (36/3): 
        print(angle)
        plotMatches(image, rotated, matches, locs1, locs2)

# Display histogram
plt.figure()
plt.bar(hist_angles, hist_matches)
plt.show()

pass
