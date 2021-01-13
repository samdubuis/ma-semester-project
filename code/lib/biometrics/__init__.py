"""
" All the biometric routines used in 
" the various applications
"""

import cv2
import logging
import os
import subprocess

from .biocore import preprocess, maximum_curvature, miurascore, np, si, sp


log = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING) #Silencing those enormous debug messages from matplotlib


def extract_features(image):

    """ Runs complete feature extraction (Preprocessing + Maximum Curvature Vein Extraction)
    on a given image.

    @param image (numpy.ndarray) : The images to extract veins from
    #TODO: one could optimise by having an array of booleans as output !
    @return (numpy.ndarray of float64 0's and 1's) : The extracted image
    """

    return maximum_curvature(*preprocess(image), sigma = 3)


def backelcpp(img, filename, camera, quiet = False):

    """ Wrapper for the C++ background elimination script
    Takes an image, a name, a camera number and removes the background of the scanner, 
    leaving only the finger part

    @param img (numpy.ndarray) : The image to be extracted
    @param filename (str) : The input path. The output will be saved as <filename>_mod_mod.png
    @param camera (str) : The number of the camera which took the image ('0' for left, '1' for center, '2' for right)
    @param quiet : Whether to silence the c++ prints or not

    @return s (int) : The return code of the c++ script
    @return (numpy.ndarray) : The resulting image
    """

    os.chdir(os.path.realpath(os.path.dirname(__file__)))
    cv2.imwrite(filename, img)
    s = subprocess.call(["./background_elimination", filename, camera], stdout = subprocess.DEVNULL if quiet else None)
    img = cv2.imread(filename[:-4] + "_mod_mod.png", cv2.IMREAD_GRAYSCALE)

    return s, img


def fingerfocus(impath, camnum, keepratio = .5, verbose = False):

    BORDERS = [(40, 270, 40, 140), (40, 270, 50, 190), (40, 270, 100, 200)] #[(Left), (Center), (Right)]

    """ Attempt at an alternative background elimination method

    @date : 13 Aug 2019
    @author : Julien Corsin SC-MA1 @ EPFL

    Uses Sobel filters to construct a gradient of the image, 
    and connected components algorithm to find a suitable finger 
    mask approximation.

    @param : impath (str) path of the image to extract
    @param : camnum (int) index of the camera which took the picture (0 = LEFT, 1 = RIGHT, 2 = CENTER)

    @return img (numpy.ndarray) : The resulting image
    """

    if camnum < 0 or camnum > 2:
        raise ValueError("The camera index must be in [0, 1, 2]")

    ymin, ymax, xmin, xmax = BORDERS[camnum]

    img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)

    if verbose: cv2.imshow("Original Image", img)

    cropped = img[xmin : xmax, ymin : ymax]

    if verbose: cv2.imshow("Cropped", cropped)

    # Sobel filters
    gx = sp.convolve2d(cropped, np.array([[-1, -2, -1],
                                          [ 0,  0,  0],
                                          [ 1,  2,  1]]), mode = "same")

    gy = sp.convolve2d(cropped, np.array([[-1,  0,  1],
                                          [-2,  0,  2],
                                          [-1,  0,  1]]), mode = "same")

    gradient = np.hypot(gx, gy)

    if verbose: cv2.imshow("Gradient", gradient / np.max(gradient))

    mask = gradient > gradient.mean()
    gradient[mask] = 0
    gradient[~mask] = 255

    if verbose: cv2.imshow("Thresholded+Inverted gradient", gradient / np.max(gradient))

    gradient = si.binary_erosion(gradient, iterations = 2, structure = np.array([[0, 1, 0],
                                                                                 [1, 1, 1],
                                                                                 [0, 1, 0]]))

    if verbose: cv2.imshow("Eroded gradient", gradient / np.max(gradient))

    connected, n = si.label(gradient, structure = np.array([[0, 1, 0],
                                                            [1, 1, 1],
                                                            [0, 1, 0]]))

    sorted_labels = np.bincount(connected.ravel()).argsort()
    largest = sorted_labels[-1] if sorted_labels[-1] != 0 else sorted_labels[-2]
    mask = connected == largest

    for i in range(n):
        connected[connected == i] = i * 255 / n

    if verbose: cv2.imshow("Connected Components", connected / 255)

    if verbose: cv2.imshow("Largest component", mask / mask.max())

    while mask[mask].size / (mask.shape[0] * mask.shape[1]) < keepratio:
        mask = si.binary_dilation(mask, iterations = 1, structure = np.array([[0, 1, 0],
                                                                              [1, 1, 1],
                                                                              [0, 1, 0]]))

    mask = si.binary_fill_holes(mask)

    if verbose: cv2.imshow("Dilated until > " + keepratio*100 + "% + filled", mask / mask.max())

    img[:xmin] = img[xmax:] = 0
    img[:, :ymin] = img[:, ymax:] = 0
    img[xmin : xmax, ymin : ymax] *= mask

    if verbose:
        cv2.imshow("End result : ", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    cv2.imwrite(impath[:-4] + "_focus.png", img)

    return img