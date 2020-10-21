import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
num_frames = seq.shape[2];

rect = [59, 116, 145, 151] # initial rectangle value

rects = np.zeros((num_frames,4))
rects[0,:] = rect

rects_wcrt = np.zeros((num_frames,4)) # rects with template correction
rects_wcrt[0,:] = rect

def show_frame(f, save=False):
    plt.figure()
    plt.imshow(seq[:,:,f], cmap='gray')
    width, height = rects[f,2:] - rects[f,0:2]
    patch = patches.Rectangle(rects[f,0:2], width, height, linewidth=1, edgecolor='b', facecolor='none')
    plt.gca().add_patch(patch)
    patch = patches.Rectangle(rects_wcrt[f,0:2], width, height, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(patch)
    if save:
        plt.axis('off')
        plt.savefig("../outputs/q1_4_car_{}.png".format(f), bbox_inches = 'tight')
    else:
        plt.show()

pn = np.zeros(2) # keep p_n around and use it as the p0 seed for finding p_n+1 (new version with template correction)
for f in range(num_frames-1):
    # Original (so we can plot it on each frame):
    p = LucasKanade(seq[:,:,f], seq[:,:,f+1], rects[f,:], threshold, num_iters, p0=np.zeros(2))
    rects[f+1,:] = rects[f,:] + [p[0],p[1], p[0],p[1]]
    
    # New version with template correction:
    pn = LucasKanade(seq[:,:,f], seq[:,:,f+1], rects_wcrt[f,:], threshold, num_iters, p0=pn)
    pn_seed = (rects_wcrt[f,:2] + pn - rects_wcrt[0,:2]) # convert change from n to n+1 from to 0 to n+1 frame
    pn_star = LucasKanade(seq[:,:,0], seq[:,:,f+1], rects_wcrt[0,:], threshold, num_iters, p0=pn_seed) # find p_n+1^star, start at p_n+1
    if np.linalg.norm(pn_star - pn_seed) <= template_threshold:
        # make sure p_star gradient descent didn't diverge
        rects_wcrt[f+1,:] = rects_wcrt[0,:] + [pn_star[0],pn_star[1], pn_star[0],pn_star[1]]
        pn = (pn_star - pn_seed) + pn # apply template warp correction to pn (so next iter starts correctly). See Figure 2 in paper which says pn should be seeded from previous p_star
    else:
        # otherwise, just keep the old one around (act conservatively)
        rects_wcrt[f+1,:] = rects_wcrt[f,:]
    #print("{}/{}".format(f+2,num_frames))

np.save('../result/carseqrects-wrct.npy', rects_wcrt) # Note: rect is allowed to be float according to Piazza @323

for f in [1,100,200,300,400]:
    show_frame(f)