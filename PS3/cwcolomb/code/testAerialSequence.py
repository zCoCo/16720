import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from SubtractDominantMotion import SubtractDominantMotion

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.175, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
num_frames = seq.shape[2]

# Frames to run on (use np.arange(1,num_frames) if you want to run on all frames):
frames = np.asarray([30,60,90,120]) # np.arange(1,num_frames) # for all frames

masks = np.zeros(seq.shape) # collection of all masks

def show_frame(f, save=False):
    plt.figure()
    plt.imshow(seq[:,:,f], cmap='gray')
    plt.imshow(masks[:,:,f], cmap='Blues', alpha=0.75)
    if save:
        plt.axis('off')
        plt.savefig("../outputs/q2_3_aerial_{}.png".format(f), bbox_inches = 'tight')
    else:
        plt.show()

pn = np.zeros(2) # keep p_n around and use it as the p0 seed for finding p_n+1 (new version with template correction)
time_total = 0
for f in frames:# range(num_frames-1):
    tic = time.time()
    masks[:,:,f] = SubtractDominantMotion(seq[:,:,f-1],seq[:,:,f], threshold, num_iters, tolerance)
    toc = time.time()
    
    show_frame(f)
    #show_frame(f, save=True)
    
    time_total = time_total + (toc-tic)
    print("Frame {:d} done after {:3f}s, {:3f}s total".format(f, toc-tic, time_total))