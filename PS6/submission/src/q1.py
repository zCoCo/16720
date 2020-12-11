# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import lsqr as splsqr

from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    
    res = res.flatten() # just in case it's given as a (2,1) instead of (2,)
    light = light.reshape(-1,1) # just in case given as (3,) instead of (3,1)

    # Initialize Image:
    # Note: assuming `res` is the dimensions of the camera frame in px (since 
    # the assignment refers to the 3840x2160 size as resolution.)
    # Note: assuming `res` is given as (width, height) in px
    # Note: assuming pxSize is in the same units (eg. cm) as everything else
    image = np.zeros(np.flip(res))
    width, height = res * pxSize # physical width and height of camera
    
    # Grab center location of sphere (in a case not 0,0,0 as specified in problem):
    # Note: z center doesn't matter because camera is orthographic.
    xc, yc = center[0:2]
    
    # Operate over a meshgrid of the image:
    c,r = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    # Compute pixel location:
    x = c * pxSize - width/2.0
    y = height/2.0 - r * pxSize # note y axis is flipped in image (i.e. row=0 is at top of image, y=-height/2 is at bottom of camera)
    
    # Check if point is inside sphere:
    in_sphere = (x-xc)**2 + (y-yc)**2 < rad**2
    
    # Keep only the points in the sphere:
    x = x[in_sphere]
    y = y[in_sphere]
    
    # Get corresponding location on sphere:
    z = np.sqrt( rad**2 - (x-xc)**2 - (y-yc)**2 )
    
    # Compute surface normal at that location:
    # Skipping over this step and just baking it into intensity calculation
    # n = np.asarray([x-xc,y-yc,z]) / np.sqrt( (x-xc)**2 + (y-yc)**2 + z**2 )
    # n = n.reshape(-1,1)
    
    # Compute intensity (NDotL):
    # Note: normal is just [x-xc,y-yc,z] / sqrt( (x-xc)**2 + (y-yc)**2 + z**2 )
    I = ((x-xc) * light[0] + (y-yc) * light[1] + z * light[2]) / np.sqrt( (x-xc)**2 + (y-yc)**2 + z**2 )
    I[I<0.0] = 0.0 # toss out negative values (per lecture), equiv. to max(n dot l, 0)
    
    # Store output anywhere inside sphere:
    image[in_sphere] = I
    
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    
    # Load the nth image, process it, and return the processed vector:
    def load_image_row(n):
        # Load rgb image:
        rgb = imread(path + 'input_{}.tif'.format(n))
        # Convert to xyz:
        xyz = rgb2xyz(rgb)
        # Extract the luminance channel (Y), ravel, and build row for I:
        return xyz[:,:,1].ravel().reshape(1,-1)
    
    # Load first image to build first row of I:
    I = load_image_row(1)
        
    # Repeat for rest of images:
    for i in np.arange(2,8):
        I = np.append(I, load_image_row(i), axis=0)

    # Load lighting data:
    L = np.load(path + 'sources.npy').T # Transpose to force into correct shape (comes as 7x3)

    # Grab original image shape:
    s = imread(path + 'input_1.tif').shape[0:2]

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # Setup:
    P = I.shape[1]

    # Setup Sparse Linear System:
    y = I.flatten().reshape(-1,1)
    
    # Create the 7P*3P A matrix in accordance with writeup:
    """
    Format (where Li is the ith row of L^T):
    L1 0 ... 0
    0 L1 ... 0
    0 0 ... L1
    L2 0 ... 0
    0 L2 ... 0
    0 0 ... L2
    ... ... ...
    L7 0 ... 0
    0 L7 ... 0
    0 0 ... L7
    """
    LT = L.T
    LP = np.repeat(LT, repeats=P, axis=0).reshape(7*P,1,3)
    idx = np.tile(np.arange(P), 7) # range out to num cells (P), tile out to num rows (7)
    indptr = np.arange(7*P+1)
    A = bsr_matrix((LP, idx,indptr), shape=(7*P,3*P))

    # Solve the sparse system in a least squares sense:
    x = splsqr(A,y)[0] # can use lsqr to solve per @628 on Piazza

    # Devectorize and build the B matrix:
    B = x.reshape(-1,3).T
    
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    
    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(s[0],s[1],3)

    # Albedos Image:
    plt.figure()
    plt.imshow(albedoIm, cmap=cm.gray)
    plt.show()
    
    # Normals Image:
    plt.figure()
    # Normalize for display purposes (avoid "clipping" error):
    normalImDisplay = (normalIm - np.min(normalIm)) / (np.max(normalIm) - np.min(normalIm))
    # Display:
    plt.imshow(normalImDisplay, cmap=cm.rainbow)
    plt.show()

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    
    # Estimate Derivatives:
    fx = -normals[0,:] / normals[2,:]
    fy = -normals[1,:] / normals[2,:]
    
    # Reshape Derivatives to Image:
    fx = fx.reshape(s)
    fy = fy.reshape(s)
    
    surface = integrateFrankot(fx, fy, pad = 512)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    # Setup:
    X, Y = np.meshgrid(np.arange(surface.shape[1]), np.arange(surface.shape[0]))
    
    # Plot:
    for elev, azim in [(30,-60), (-90, -90), (-112,-110), (-142,-130)]:
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=elev, azim=azim)
        plt.show()


if __name__ == '__main__':

    # Put your main code here
    
    # Q1.b:
    run_q1b = True
    if run_q1b:
        lighting = np.asarray([
        [1,1,1],
        [1,-1,1],
        [-1,-1,1]
        ]).T / np.sqrt(3)
        center = np.asarray([0,0,-10]) #  w.r.t. camera
        rad = 0.75
        pxSize = 7e-6 * 100 # um to cm
        res = np.asarray([3840,2160])
        for i in range(3):
            image = renderNDotLSphere(center, rad, lighting[:,i], pxSize, res)
            plt.figure()
            plt.imshow(image, cmap=cm.gray)
            plt.title(r'Q1.b - For lighting vector {}/$\sqrt{{3}}$'.format(np.array2string((lighting[:,i]*np.sqrt(3.0)).astype(int))))
            plt.show()

    # Q1.d:
    run_q1d = True
    if run_q1d:
        I, _, _ = loadData()
        _, s, _ = np.linalg.svd(I, full_matrices=False)
        print(s)
        
    # Q1.e:
    run_q1e = True
    if run_q1e:
        I, L, s = loadData()
        B = estimatePseudonormalsCalibrated(I, L)
        albedos, normals = estimateAlbedosNormals(B)

    # Q1.f:
    run_q1f = True
    if run_q1f:
        I, L, s = loadData()
        B = estimatePseudonormalsCalibrated(I, L)
        albedos, normals = estimateAlbedosNormals(B)
        albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
        
    # Q1.i:
    run_q1i = True
    if run_q1i:
        I, L, s = loadData()
        B = estimatePseudonormalsCalibrated(I, L)
        albedos, normals = estimateAlbedosNormals(B)
        surface = estimateShape(normals, s)
        plotSurface(surface)
