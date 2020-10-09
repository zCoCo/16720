import numpy as np
import cv2
#Import necessary functions
import skimage.io 
import skimage.color
import matplotlib.pyplot as plt
from opts import get_opts
import multiprocessing

# Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

opts = get_opts()

#Write script for Q3.1

# Setup:
n_cpu = multiprocessing.cpu_count()
    
# Load reference image:
cv_cover = cv2.imread('../data/cv_cover.jpg')
# Load videos:
vid_source = loadVid('../data/ar_source.mov')
vid_target = loadVid('../data/book.mov')

# Define Render Pipeline:
def process_frame(args):
    """Worker thread to extract image features as part of a Pool."""
    img_idx, source_frame, target_frame = args
    
    print('\t Processing Frame {}'.format(img_idx))
    
    opts = c2__frame_worker_cache['opts']
    cv_cover = c2__frame_worker_cache['cv_cover']
    source_xmin = c2__frame_worker_cache['source_xmin']
    source_xmax = c2__frame_worker_cache['source_xmax']
    
    # Crop center to appropriate aspect ratio then scale to meet target size:
    source_frame = source_frame[:, source_xmin:source_xmax]
    source_frame = cv2.resize(source_frame, (cv_cover.shape[1], cv_cover.shape[0]))
    
    # Compute Homography:
    matches, locs1, locs2 = matchPics(cv_cover, target_frame, opts)
    bestH2to1, inliers = computeH_ransac(locs1[matches[:,0],:], locs2[matches[:,1],:], opts)
    
    # Composite Images:
    composite = compositeH(bestH2to1, source_frame, target_frame)
    
    # Display Result:
    """\
    plt.figure()
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    """
    
    return composite

def initialize_frame_workers(opts, cv_cover, source_xmin,source_xmax):
    """
    Initialize pool of the workers.
    
    Populates a global cache with values that need to be used by all of them 
    and don't vary by worker.
    """
    global c2__frame_worker_cache
    c2__frame_worker_cache = {"opts": opts, "cv_cover": cv_cover, "source_xmin": source_xmin, "source_xmax": source_xmax}


# Precompute cropping parameters:
ideal_source_width = cv_cover.shape[1]/cv_cover.shape[0] * vid_source.shape[1]
mid = vid_source.shape[2]/2
source_xmin = np.round(mid - ideal_source_width/2)
source_xmax = np.round(mid + ideal_source_width/2)

# Execute Render Pipeline:
print('Pooling...')
# Process frames in parallel, stack them at the end (since there are no inter-frame dependencies, this application is great for parallelization):
pool = multiprocessing.Pool(n_cpu, initializer=initialize_frame_workers, initargs=(opts,cv_cover,source_xmin,source_xmax))
pool_data = zip(range(len(vid_target)), vid_source, vid_target)
vid_rendered = pool.map(process_frame, pool_data)
pool.close()
pool.join()

# Save video:
writer = cv2.VideoWriter('../result/ar.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         (vid_target.shape[0]/20+1), (vid_rendered.shape[2], vid_rendered.shape[1]))
    
for frame in vid_rendered:
    writer.write(frame)
    cv2.imshow('Frame', frame)
    
writer.release()


