import numpy as np
import cv2
import time
# Import necessary functions
from loadVid import loadVid
from cwcolombHelperFunctions import get_opts, matchPics, computeH_ransac, compositeH

#Write script for Q4.1x

def main():
    opts = get_opts()
    
    # Adjust settings to best tuned parameters (allowed per Piazza @288):
    opts.sigma = 0.15
    opts.ratio = 0.65
    opts.max_iters = 15
    opts.inlier_tol = 1.0
        
    # Load reference image:
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    # Load videos:
    print('Loading Videos...')
    vid_source = loadVid('../data/ar_source.mov')
    vid_target = loadVid('../data/book.mov')
    
    print('Booting...')
    # Precrop entire source video cropping parameters:
    # also eliminate the horizontal black bars:
    # determine the bar size as 2* the centroid position of the dark (10%) values 
    # in the upper half of the first frame:
    source_frame1 = vid_source[0,:,:,1]
    mid = source_frame1.shape[0]//2
    bar_centroid = 0
    for c in range(source_frame1.shape[1]):
        bar_centroid = bar_centroid + np.mean(np.where(source_frame1[0:mid,c] < 0.1*255))
    bar_centroid = bar_centroid // source_frame1.shape[1]
    bar_height = 2*bar_centroid
    
    source_ymin = int(bar_height)
    source_ymax = int(vid_source.shape[1]-bar_height)
    
    ideal_source_width = cv_cover.shape[1]/cv_cover.shape[0] * (source_ymax-source_ymin)
    mid = vid_source.shape[2]/2
    source_xmin = int(np.round(mid - ideal_source_width/2))
    source_xmax = int(np.round(mid + ideal_source_width/2))
    
    vid_source = vid_source[:, source_ymin:source_ymax, source_xmin:source_xmax, :]
    

    font = cv2.FONT_HERSHEY_SIMPLEX # FPS Counter font
        
    # Execute Render Pipeline:
    print('Running...')
    start_time = time.time()
    homography_update_period = 2
    bestH2to1 = None
    for frame_idx, source_frame, target_frame in zip(range(len(vid_target)), vid_source, vid_target):
        # Crop center to appropriate aspect ratio then scale to meet target size:
        source_frame = cv2.resize(source_frame, (cv_cover.shape[1], cv_cover.shape[0]))
        
        # Compute Homography (only update every nth frame):
        if frame_idx % homography_update_period or bestH2to1 is None:
            matches, locs1, locs2 = matchPics(cv_cover, target_frame, opts)
            bestH2to1, inliers = computeH_ransac(locs1[matches[:,0],:], locs2[matches[:,1],:], opts)
        
        # Composite Images:
        composite = compositeH(bestH2to1, source_frame, target_frame)
        
        # Compute fps:
        fps = int( (frame_idx+1) // (time.time() - start_time) ) 
        # Layer on fps:
        cv2.putText(composite, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        # Display feed:
        cv2.imshow('frame', composite)#cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()

