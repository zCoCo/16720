import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # Preprocess Image:
    # (if this fails later, consider using total variation filter instead)
    # Bilateral is nice b/c it's ideally edge preserving
    sigma = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    denoised = skimage.restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=100*sigma, multichannel=True)

    # Get BW Image:
    gray = skimage.color.rgb2gray(denoised)
    thesholds = skimage.filters.threshold_multiotsu(image)
    bw = gray > 0.9*np.min(thesholds)
    
    # Invert
    # (letters need to be 1, bg needs to be 0 for subsequent processing):
    inverted = ~bw
    
    # Refine Characters:
    refined = skimage.morphology.closing(inverted) # get morphological closing of image
    refined = skimage.segmentation.clear_border(refined)
    
    # Label Image Segments:
    labelled = skimage.measure.label(refined)
    
    # Extract bounding boxes:
    min_area = (32)**2 # (min area: square with half the size of NN input's square)
    for region in skimage.measure.regionprops(labelled):
        # Ensure region isn't trivial in size:
        if region.area < min_area: 
            continue;
            
        # Ensure region has roughly the right aspect ratio (i.e. that it won't 
        # be unrecognizably stretched when mapped to 32x32):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if width > 10*height or height > 10*width:
            continue;
            
        # If still here, all conditions passed and this can be added:
        bboxes.append(region.bbox)

    bw = skimage.img_as_float(bw)

    return bboxes, bw