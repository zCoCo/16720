import numpy as np
import numpy.linalg
from cv2 import warpAffine
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    height, width = image1.shape
    
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    
    # image 1 registered onto image 2:
    image1_onto_2 = warpAffine(image1, M[:2,:], (width,height)) # allowed to use cv2 here according to Piazza @327_f1
    
    mask = np.abs(image2 - image1_onto_2) > tolerance
    
    # Join up bodies:
    mask = binary_dilation(mask, iterations=2)
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)
    mask = binary_fill_holes(mask) # allowed according to Piazza @397
    
    # Remove Noise:
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=2)
    
    return mask
