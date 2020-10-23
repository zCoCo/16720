import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-4, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
num_frames = seq.shape[2];

rect = [59, 116, 145, 151] # initial rectangle value

rects = np.zeros((num_frames,4))
rects[0,:] = rect

def show_frame(f, save=False):
    plt.figure()
    plt.imshow(seq[:,:,f], cmap='gray')
    width, height = rects[f,2:] - rects[f,0:2]
    patch = patches.Rectangle(rects[f,0:2], width, height, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(patch)
    if save:
        plt.axis('off')
        plt.savefig("../outputs/q1_3_car_{}.png".format(f), bbox_inches = 'tight')
    else:
        plt.show()

for f in range(num_frames-1):
    # NB: argmin is init w/zeros (better performance than p_n-1/minimizes affect of drift - prevents inertia)
    p = LucasKanade(seq[:,:,f], seq[:,:,f+1], rects[f,:], threshold, num_iters, p0=np.zeros(2))
    rects[f+1,:] = rects[f,:] + [p[0],p[1], p[0],p[1]]
    #print("{}/{}".format(f+2,num_frames))

np.save('../result/carseqrects.npy', rects) # Note: rect is allowed to be float according to Piazza @323

for f in [1,100,200,300,400]:
    show_frame(f)