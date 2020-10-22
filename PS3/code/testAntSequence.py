import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-3, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.25, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
num_frames = seq.shape[2]

masks = np.zeros(seq.shape) # collection of all masks

def show_frame(f, save=False):
    plt.figure()
    #plt.imshow(seq[:,:,f], cmap='gray')
    plt.imshow(masks[:,:,f], cmap='Blues')#, alpha=0.7)
    if save:
        plt.axis('off')
        plt.savefig("../outputs/q1_4_car_{}.png".format(f), bbox_inches = 'tight')
    else:
        plt.show()

pn = np.zeros(2) # keep p_n around and use it as the p0 seed for finding p_n+1 (new version with template correction)
for f in range(num_frames-1):
    masks[:,:,f+1] = SubtractDominantMotion(seq[:,:,f],seq[:,:,f+1], threshold, num_iters, tolerance)
    show_frame(f+1)
    print("{}/{}".format(f+2,num_frames))