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
    bboxes = np.asarray([], dtype=int)
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    # Preprocess Image:
    # (if this fails later, consider using total variation filter instead)
    # Bilateral is nice b/c it's ideally edge preserving
    sigma = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    denoised = skimage.restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=100*sigma, multichannel=True)
    # boosted = skimage.exposure.adjust_log(denoised, 1) # boost contrast w/ log correction
    # boosted2 = skimage.exposure.adjust_gamma(denoised, 2) # boost contrast w/ gamma correction

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
    
    dilated = skimage.morphology.dilation(refined)
    dilated = skimage.morphology.dilation(dilated)
    dilated = skimage.morphology.dilation(dilated)
    
    # Label Image Segments:
    pre_labelling = dilated
    labelled = skimage.measure.label(pre_labelling)
    
    # Extract bounding boxes:
    min_area = (32/2)**2 # (min area: square with half the size of NN input's square)
    for region in skimage.measure.regionprops(labelled):
        strikes = 0 # a N-strikes and you're out system to disqualify letter candidates
            
        # Ensure region has roughly the right aspect ratio (i.e. that it won't 
        # be unrecognizably stretched when mapped to 32x32):
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if width > 10*height or height > 10*width:
            strikes+=1
            
        # Ensure pixel count isn't trivial in size:
        if region.area < min_area:
            continue
        # Ensure region count isn't trivial in size:
        if width*height < 2*min_area:
            strikes+=1
        
        # Ensure that when the letter is remapped to a square it won't be 
        # mostly a solid block (in which case it's likely not a letter):
        if np.count_nonzero(pre_labelling[minr:maxr, minc:maxc]) > 0.5*width*height:
            strikes+=1
            
        # If still here (critical conditions passed), and there are less than 2 strikes, this can be added:
        if strikes < 2:
            bboxes = np.append(bboxes, region.bbox).reshape(-1,4)

    bw = skimage.img_as_float(bw)

    return bboxes, bw