import numpy as np
import cv2
import skimage.color
import scipy.ndimage
from matchPics import matchPics
from opts import get_opts
from helper import plotMatches

opts = get_opts()

# Q2.1.6
# Read the image and convert to grayscale, if necessary
image = cv2.imread('../data/cv_cover.jpg') # Note: matchPics takes care of grayscaling

for i in range(36):
	# Rotate Image
    angle = i*10
    rotated = scipy.ndimage.rotate(image, angle)
	
	# Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(image, rotated, opts)
    
    if i < 36 and i % (36/3): # Display matching at three orientations
        print(angle)
        plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

    i = i
	# Update histogram


# Display histogram

