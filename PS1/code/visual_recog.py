"""Function definitions for building a system of visual words."""

import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import visual_words

import util


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    
    # Compute histogram:
    hist, _ = np.histogram(wordmap, bins = np.arange(0,K+1) - 0.5);
    
    # L1 Normalize so sum is 1:
    hist = hist / np.sum(hist)
    
    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    """
    K = opts.K
    L = opts.L-1 # convert to 0-indexing to be compatible with writeup
    
    assert L >= 0, "Number of layers must be at least 1 (L=0). Given L={}.".format(L)
    
    hist_all = None;
    if L == 0:
        # default case for only 1 layer
        hist_all = get_feature_from_wordmap(opts, wordmap)
    else:
        hist_all = np.zeros( int(K*(4**(L+1) - 1)/3) ) # preallocate for speed
        
        def chunk_op(l0, index0, chunk):
            """Operations to perform for each pyramid chunk / layer quadrant"""
            nonlocal L
            nonlocal K
            weight = 0
            if l0 > 1:
                weight = 2**(l0-L-1)
            else:
                weight = 2**(-L)
            # Record weighted responses for this chunk:
            # (not composing out of child chunk histograms b/c, even though 
            # there would be a performance boost, such compostion would cause 
            # features bigger than the smallest chunk to be sporadically 
            # detected (this was tested and observed for high L)... this could 
            # be fixed if an algorithm could be developed to determine whether 
            # such details could be lost given filter_scales and L):
            hist_all[index0*K:(index0+1)*K] = weight * get_feature_from_wordmap(opts, chunk)
            
            if l0 < L:
                # If not on final layer, chunk and repeat:
                shape = chunk.shape
                quads = [None]*4
                quads[0] = chunk[:int(shape[0]/2), :int(shape[1]/2)]
                quads[1] = chunk[:int(shape[0]/2), int(shape[1]/2):]
                quads[2] = chunk[int(shape[0]/2):, :int(shape[1]/2)]
                quads[3] = chunk[int(shape[0]/2):, int(shape[1]/2):]

                l0 = l0+1
                for i, quad in enumerate(quads):
                    chunk_op(l0, 4*index0 + i + 1, quad)
                    
        chunk_op(0, 0, wordmap)
        
        # Ensure Renormalized (probably unnecessary):
        hist_all = hist_all / np.sum(hist_all)
        
    return hist_all
    
def get_image_feature(opts, img_path, dictionary):
    """
    Extract the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """
    img = Image.open(join(opts.data_dir,img_path))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)

def recognition_system__get_image_feature(args):
    """Worker thread to extract image features as part of a Pool."""
    img_idx, img_path = args
    
    util.dbg_print('\t Processing Image {}'.format(img_idx))
    
    opts = recognition_system__worker_cache['opts']
    dictionary = recognition_system__worker_cache['dictionary']
    
    return get_image_feature(opts, img_path, dictionary)
    

def recognition_system__initialize_workers(opts, dictionary):
    """
    Initialize pool of the workers.
    
    Populates a global cache with values that need to be used by all of them 
    and don't vary by worker.
    """
    global recognition_system__worker_cache
    recognition_system__worker_cache = {"opts": opts, "dictionary": dictionary}

def build_recognition_system(opts, n_worker=1):
    """
    Create a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = util.c2_load_dictionary(opts)

    util.dbg_print('Pooling...')
    # Process filter responses in parallel:
    pool = multiprocessing.Pool(n_worker, initializer=recognition_system__initialize_workers, initargs=(opts,dictionary))
    pool_data = zip(range(len(train_files)), train_files)
    features_array = pool.map(recognition_system__get_image_feature, pool_data)
    pool.close()
    pool.join()
    
    # Stack results:
    features = np.stack(features_array, axis=0)
    util.dbg_assert(features.shape == (len(train_files), opts.K*(4**opts.L-1)/3))
    
    util.dbg_print('Exporting Trained System...')

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """
    shape = histograms.shape
    sim = shape[0]
    for t in range(shape[0]):
        sim[t] = 1 - np.amax(np.minimum(word_hist, histograms[t,:])) # actually distance was requested
    return sim
    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    pass

# DEV TESTING:
if __name__ == '__main__':
    from opts import get_opts
    opts = get_opts()
    
    dictionary = util.c2_load_dictionary(opts)
    
    img = Image.open(join('../data/','kitchen/sun_aasmevtpkslccptd.jpg'))
    img = np.array(img).astype(np.float32)/255
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    #opts.L = 3
    #get_feature_from_wordmap_SPM(opts, wordmap)
    
    build_recognition_system(opts, n_worker=util.get_num_CPU())
