import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    # plt.imshow(im1); plt.show();

    plt.imshow(bw, cmap=plt.cm.gray)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # Uses Least Squares to find the coefficients that best fit a line to the given
    # x,y data.
    # Returns a vector of line coefficients, C, and fitting error.
    def fit_line(x,y):
        # Ensure x and y are column vectors:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        m = np.Inf
        b = np.Inf
        if x.size > 0 and y.size > 0:
            meanx = np.mean(x)
            meany = np.mean(y)
            
            # Least squares:
            SDx2 = np.sum( (x-meanx)**2 )
            if SDx2 > 1e-9:
                m = np.sum( (x-meanx)*(y-meany) ) / SDx2
                b = meany - m*meanx
        
        # y = mx+b -> -y + mx + b = 0 -> C = [m; b]
        C = np.asarray([m,b]).reshape(-1,1)
        
        A = np.hstack((x, np.ones(x.shape)))
        
        y_line = A @ C;
        error = np.sqrt(np.sum((y - y_line)**2))
        return C, error
    
    # Returns a vector of the point-to-line distance for every point given by 
    #  the (x,y) vectors to the line defined by the given coefficients vector. 
    def point_to_line_dists(x,y,C):
        # Ensure x and y are column vectors:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        A = np.hstack((x, np.ones(x.shape)))
        
        y_line = A @ C
        
        scaling = np.sqrt(C[0]**2 + 1)
        if np.count_nonzero(scaling == np.Inf) > 0 or np.count_nonzero(y_line == np.Inf) > 0:
            dists = np.Inf * y
        else:
            dists = np.abs(y - y_line) / scaling
            
        return dists
    
    # Uses RANSAC to find the line that intersects the most (x,y) datapoints where 
    # x and y are column vectors
    def RANSAC_line(x,y, inlier_tol, stop_thresh, max_iter):
        # Ensure x and y are column vectors:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        # Populate an initial estimate:
        best_C, _ = fit_line(x,y)
        best_inliers = np.nonzero(point_to_line_dists(x,y,best_C).flatten() < inlier_tol)[0]
        best_inlier_count = best_inliers.size
        
        # RANSAC the line
        count = 0
        sample_size = 3
        while count < max_iter and x.size > 0 and y.size > 0:
            # Randomly select points:
            sample = np.random.randint(0, x.size, size=(1,sample_size)) # indices to sample
            # Perform fit:
            C, _ = fit_line(x[sample],y[sample])
            inliers = np.nonzero(point_to_line_dists(x,y,C).flatten() < inlier_tol)[0] # indices of inliers
            inlier_count = np.size(np.setdiff1d(inliers,sample)) # counts # of inliers (excl. sample)
            
            if inlier_count > best_inlier_count:
                best_C = C
                best_inliers = inliers
                best_inlier_count = inlier_count
            
            if inlier_count/x.size > stop_thresh:
                break
            count = count + 1
        
        # Try to build a better model out of all the best inliers:
        C, _ = fit_line(x[best_inliers],y[best_inliers])
        inliers = np.nonzero(point_to_line_dists(x,y,C).flatten() < inlier_tol)[0] # indices of inliers
        inlier_count = inliers.size # counts # of inliers (excl. sample)
        if (inlier_count-sample_size) > best_inlier_count:
            best_C = C # Accept this new model
            best_inliers = inliers
        
        A = np.hstack((x[best_inliers], np.ones((best_inliers.size,1))))
        y_line = A @ best_C
        error_inlier = np.sqrt(np.sum((y[best_inliers] - y_line)**2))
        
        avg_dist_inlier = np.Inf
        if np.count_nonzero(best_inliers) > 0:
            avg_dist_inlier = np.mean(point_to_line_dists(x[best_inliers],y[best_inliers],best_C))
        return best_C, error_inlier, avg_dist_inlier, best_inliers
    
    # Performs the NIRDE algorithm (as described in writeup) to find N Dominant
    # Examples in the given data set using RANSAC with the given parameters.
    # Returns a matrix where each column contains the line coefficients vector 
    # for the ith line. Also returns a matrix, inliers, where the ith column is a
    # logical index vector for x indiciating which points are inliers to the
    # line described by the ith column of Cs.
    def NIRDE_lines(N, x,y, inlier_tol, stop_thresh, max_iter):
        # Ensure x and y are column vectors:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        
        Cs = np.zeros((2,N));
        inliers = np.zeros((x.size,N), dtype=bool)
        considered = np.ones(x.size, dtype=bool) # Points still being considered
        i = 0
        while i < N:
            # Fit line:
            csi, _,_, considered_inliers = RANSAC_line(x[considered],y[considered], inlier_tol, stop_thresh, max_iter);
            Cs[:,i] = csi.flatten()
            # considered_inliers is a vector of indices **for x[considered]** (not x). Convert to boolean index vector for x.
            considered_idx = np.nonzero(considered)[0];
            inliers[considered_idx[considered_inliers],i] = True;
            # Remove inliers from consideration:
            considered = considered & ~inliers[:,i];
            
            i = i + 1
            # Stop early if all points are already inliers to indentified lines:
            if np.count_nonzero(considered) == 0:
                break
            
        # Trim any columns that were not considered (ran out of entities):
        Cs = np.delete(Cs, range(i,Cs.shape[1]), axis=1)
        inliers = np.delete(inliers, range(i,inliers.shape[1]), axis=1)
            
        # Trim any columns with no inliers (despite best efforts):
        no_inliers = np.nonzero(np.logical_not(np.sum(inliers, axis=0)))[0]
        Cs = np.delete(Cs, no_inliers, axis=1)
        inliers = np.delete(inliers, no_inliers, axis=1)
        
        return Cs, inliers
    
    # Plots the lines with the coefficients vectors that are the columns of Cs 
    # on top of the given image.
    # Also plots boxes around inliers for each line.
    # Note: x is the image column coordinate and y is the image row coordinate.
    def show_lines_on_image(im, Cs, inliers, bboxes):
        xmin = 0
        xmax = im.shape[1]
        xs = np.asarray([xmin,xmax]).reshape(-1,1)
        A = np.hstack((xs, np.ones(xs.shape)))
        ys = A @ Cs
        xs = np.repeat(xs, ys.shape[1], axis=1) # match shape of ys
        plt.figure()
        plt.imshow(im) 
        plt.plot(xs,ys)
        
        colors = ['red','orange','green','blue','black','violet']
        
        for i in range(inliers.shape[1]):
            for bbox in bboxes[inliers[:,i]]:
                color = colors[i % len(colors)]
                minr, minc, maxr, maxc = bbox
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor=color, linewidth=2)
                plt.gca().add_patch(rect)
        plt.show()
        
        
    
    centroids = np.zeros((len(bboxes), 2))
    sizes = np.zeros((len(bboxes), 2))
    for i in range(len(bboxes)):
        minr, minc, maxr, maxc = bboxes[i]
        centroids[i,:] = np.asarray([(maxc+minc), (maxr+minr)])/2 # (x,y)
        sizes[i,:] = np.asarray([(maxc-minc), (maxr-minr)]) # (width,height)
    
    Cs, inliers = NIRDE_lines(10, centroids[:,0],centroids[:,1], np.mean(sizes)/2, 1.0, 400)
    # show_lines_on_image(bw, Cs, inliers, bboxes) # UNCOMMENT TO PLOT NIRDE RANSAC results
    
    # Sort bboxes by increasing x-centroid first:
    x_sorted = np.argsort(centroids[:,0])
    bboxes = bboxes[x_sorted,:]
    centroids = centroids[x_sorted,:]
    sizes = sizes[x_sorted,:]
    inliers = inliers[x_sorted,:]
    
    # Sort rows by increasing y-intercept (decreasing height on page = top-to-bottom):
    line_sort = np.argsort(Cs, axis=1)
    sort_by_intercept = line_sort[1,:] # indices to sort entries by increasing intercept
    
    Cs = Cs[:,sort_by_intercept]
    inliers = inliers[:,sort_by_intercept]

    # Produce the dataset:       
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    dataset = np.zeros((bboxes.shape[0],1024))
    for i in range(len(bboxes)):
        # Grab letter:
        minr, minc, maxr, maxc = bboxes[i]
        letter = bw[(minr):(maxr+1), (minc):(maxc+1)]
        
        
        # Crop / pad out box:
            
        width = maxc - minc + 1
        height = maxr - minr + 1
        buffer = int(max(width,height)/3/2)
        side_padding = int((height-width)/2.0)
        side_padding_1, side_padding_2 = side_padding, side_padding
        if (height-width) % 2:
            side_padding_2 += 1 * np.sign(side_padding_2)  # deal with odds
        if height > width:
            padding = ((buffer,buffer), (buffer+side_padding_1,buffer+side_padding_2))
        else:
            padding = ((buffer+abs(side_padding_1),buffer+abs(side_padding_2)), (buffer,buffer))
        letter = np.pad(letter, padding, mode='constant', constant_values=1.)
        
        # Dilate letter (to better match dataset's thick lines):
        # b/c image is inverted, erosion dilates.
        letter = skimage.morphology.erosion(letter)
        letter = skimage.morphology.erosion(letter)
        letter = skimage.morphology.erosion(letter)
        letter = skimage.morphology.erosion(letter)
        letter = skimage.morphology.erosion(letter)
        # Boost letters up to the same "fullness" (similar to dataset):
        passes = 0
        while passes < 10 and np.count_nonzero(letter) / letter.size > 0.8:
            letter = skimage.morphology.erosion(letter)
            passes += 1
        
        # Rescale box:
        #letter = skimage.transform.rescale(letter, 32.0/letter.shape[0], anti_aliasing = True)
        letter = skimage.transform.resize(letter, (32,32), anti_aliasing=True) # *ensure* letter is appropriate size (in case of unforeseen error)
        
        dataset[i,:] = letter.T.flatten()
        
    # Visualize the letter dataset:
    #if False: # True to plot
     #   from mpl_toolkits.axes_grid1 import ImageGrid
     #   fig = plt.figure()
     #   size = int(np.sqrt(len(bboxes))) + 1
     #   grid = ImageGrid(fig, 111, nrows_ncols = (size,size), axes_pad=0.04)
     #   for i in range(dataset.shape[0]):
     #       grid[i].imshow(dataset[i,:].reshape(32,32).T)
     #   plt.show()
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    h1 = forward(dataset,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    prediction = np.argmax(probs, axis=1)
    
    text = ""
    mean_char_width = np.mean(sizes[:,0])
    for line in range(inliers.shape[1]):
        on_line = np.nonzero(inliers[:,line])[0]
        for i in range(len(on_line)):
            letter_idx = on_line[i] # get letter index
            
            if i > 0: # not the first character
                letter_prev = on_line[i-1]
                if (centroids[letter_idx,0]-0.85*sizes[letter_prev,0]) > (centroids[letter_prev,0]+0.85*sizes[letter_prev,0]):
                #if (centroids[letter_idx,0] - centroids[letter_prev,0]) > 1.75 * mean_char_width:
                    # if letters are separated, add a space first:
                    text += " "
                    
            text += letters[prediction[letter_idx]]
        text += "\n" # add linefeed at end of line
        
    print("==== {} Text: ====".format(img))
    print(text)
    print("========")
    