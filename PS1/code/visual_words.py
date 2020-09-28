"""Function definitions for extracting visual words."""

import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn

import util

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    
    # Ensure input is numpy array of dtype float with values in [0,1]:
    img = np.array(img).astype(np.float32)
    if not np.isclose(np.amax(img), 1.0) or not np.isclose(np.amin(img), 0.0):
        # Normalize if not normalized:
        img_min = np.amin(img)
        img_max = np.amax(img)
        img = (img - img_min) / (img_max - img_min)
        
        if util.c2_debug():
            util.dbg_assert(np.isclose(np.amax(img), 1.0))
            util.dbg_assert(np.isclose(np.amin(img), 0.0))
            
    # Ensure input has three channels:
    if img.ndim != 3 or img.shape[2] == 1:
        if img.ndim == 3:
            img = img[:,:,0]
        elif img.ndim != 2:
            util.dbg_warn("Improperly shaped Image array. Needs to be 2 or 3 dimensional. Given is {} dimensional.".format(img.ndim))
        
        original_shape = img.shape[:2]
        img = np.stack((img,img,img), axis=2)
        util.dbg_assert(img.shape == (*original_shape, 3))
        
    # Convert to L*a*b Color Space:
    img = skimage.color.rgb2lab(img)
    
    # Apply Filters:
    num_filters = 4 * len(filter_scales)
    filter_responses = np.zeros((*img.shape[:2], 3*num_filters))
    
    for scale_idx, scale in enumerate(filter_scales):
        for filter_idx in range(4):
            for channel in range(3):
                response_idx = scale_idx*4*3 + filter_idx*3 + channel
                
                response = None
                if filter_idx == 0: # Gaussian
                    response = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=scale) # use sigma as scale
                elif filter_idx == 1: # Laplacian of Gaussian
                    response = scipy.ndimage.gaussian_laplace(img[:,:,channel], sigma=scale)
                elif filter_idx == 2: # X-Deriv of Gaussian
                    response = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=scale, order=(0,1))
                elif filter_idx == 3: # Y-Deriv of Gaussian
                    response = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma=scale, order=(1,0))
                    
                filter_responses[:, :, response_idx] = response
                
    return filter_responses
    

def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    
    file_idx, file_name = args
    
    sample_pixel_generator = compute_dictionary__worker_cache.sample_pixel_generator

    opts = compute_dictionary__worker_cache
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    
    print("Working...")
    image = Image.open(join(data_dir, file_name))
    image = np.asarray(image)
    filter_responses = extract_filter_responses(opts, image)
    
    num_responses = filter_responses.shape[2] # 3*F
    
    sampled_responses = np.zeros((alpha, num_responses)) # preallocate for speed
    for response_idx in range(num_responses):
        flattened = filter_responses[:,:,response_idx].reshape(-1)
        shape = image.shape
        pixels = (sample_pixel_generator*shape[0]).astype(np.int)
        sampled_responses[:,response_idx] = flattened[pixels]

    # just going to return as list using pool map and np.vstack results instead of saving to temp file
    return sampled_responses

def compute_dictionary__initialize_workers(opts):
    global compute_dictionary__worker_cache
    compute_dictionary__worker_cache = opts
    # pixels to sample (consistently "random" across all images, will just be scaled by image size):
    compute_dictionary__worker_cache.sample_pixel_generator = np.random.rand(opts.alpha)

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    # #### TAKE SUBSET FOR DEV TESTING PRUPOSES:
    train_files = train_files[:5]
    
    # Process filter responses in parallel:
    pool = multiprocessing.Pool(n_worker, initializer=compute_dictionary__initialize_workers, initargs=(opts,))
    pool_data = zip(range(len(train_files)), train_files)
    result = pool.map(compute_dictionary_one_image, pool_data)
    pool.close()
    pool.join()
    
    # Stack results:
    sampled_filter_responses = np.vstack(result)
    
    # Cluster results:
    kmeans = sklearn.cluster.KMeans(nclusters=K, n_jobs=n_worker).fit(sampled_filter_responses)
    dictionary = kmeans.cluster_centers_
    
    # Save and output results:
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    return dictionary

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    pass

