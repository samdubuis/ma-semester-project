""" 
" Backbone of the biometric features, extracted from IDIAP's bob library https://www.idiap.ch/software/bob/
" The paper referred to in the various method documentations is located at the root of the doc/ folder
"""

import logging
import numpy as np
import scipy.ndimage as si
import scipy.signal as sp
import time


log = logging.getLogger(__name__)


def leemask(image, filter_height = 4, filter_width = 40):

    """ Estimates the finger region given an input NIR image using Lee et al.

    This method is based on the work of Finger vein recognition using
    minutia-based alignment and local binary pattern-based feature extraction,
    E.C. Lee, H.C. Lee and K.R. Park, International Journal of Imaging Systems
    and Technology, Volume 19, Issue 3, September 2009, Pages 175--178, doi:
    10.1002/ima.20193

    This code is based on the Matlab implementation by Bram Ton, available at:

    https://nl.mathworks.com/matlabcentral/fileexchange/35752-finger-region-localisation/content/lee_region.m

    In this method, we calculate the mask of the finger independently for each
    column of the input image. Firstly, the image is convolved with a [1,-1]
    filter of size ``(self.filter_height, self.filter_width)``. Then, the upper and
    lower parts of the resulting filtered image are separated. The location of
    the maxima in the upper part is located. The same goes for the location of
    the minima in the lower part. The mask is then calculated, per column, by
    considering it starts in the point where the maxima is in the upper part and
    goes up to the point where the minima is detected on the lower part.

    @param filter_height (:py:obj:`int`, optional): Height of contour mask in pixels, must be an even number

    @param filter_width (:py:obj:`int`, optional): Width of the contour mask in pixels
    """

    padding_width = 5
    image = np.pad(image, padding_width, 'constant',
                      constant_values = 51) # We were using default padder
    if image.dtype == np.uint8:
        image = image.astype('float64') / 255.

    img_h, img_w = image.shape # @UnusedVariable

    # Determine lower half starting point
    half_img_h = int(img_h / 2)

    # Construct mask for filtering
    mask = np.ones((filter_height, filter_width), dtype = 'float64')
    mask[int(filter_height / 2.):, :] = -1.0

    img_filt = si.convolve(image, mask, mode = 'nearest')

    # Upper part of filtered image
    img_filt_up = img_filt[:half_img_h, :]
    y_up = img_filt_up.argmax(axis = 0)

    # Lower part of filtered image
    img_filt_lo = img_filt[half_img_h:, :]
    y_lo = img_filt_lo.argmin(axis = 0)

    # Translation: for all columns of the input image, set to True all pixels
    # of the mask from index where the maxima occurred in the upper part until
    # the index where the minima occurred in the lower part.
    finger_mask = np.zeros(image.shape, dtype = 'bool')
    for i in range(img_filt.shape[1]):
        finger_mask[y_up[i]:(y_lo[i] + img_filt_lo.shape[0] + 1), i] = True

    w = padding_width
    return finger_mask[w:-w, w:-w]


def histogram_equalization(image, mask):

    """
    Applies histogram equalization on the input image, returns filtered

    @param image (numpy.ndarray) : raw image to filter as 2D array of unsigned 8-bit integers
    @param mask (numpy.ndarray) : mask to normalize as 2D array of booleans

    @return (numpy.ndarray) : A 2D boolean array with the same shape and data type of
        the input image representing the filtered image.
    """

    from skimage.exposure import equalize_hist
    from skimage.exposure import rescale_intensity

    retval = rescale_intensity(equalize_hist(
        image, mask = mask), out_range = (0, 255))

    # make the parts outside the mask totally black
    retval[~mask] = 0

    return retval


def detect_valleys(image, mask, sigma):

    """ Detects valleys on the image respecting the mask

    This step corresponds to Step 1-1 in the original paper. The objective is,
    for all 4 cross-sections (z) of the image (horizontal, vertical, 45 and -45
    diagonals), to compute the following proposed valley detector as defined in
    Equation 1, page 348:

    .. math::

       \kappa(z) = \\frac{d^2P_f(z)/dz^2}{(1 + (dP_f(z)/dz)^2)^\\frac{3}{2}}


    We start the algorithm by smoothing the image with a 2-dimensional gaussian
    filter. The equation that defines the kernel for the filter is:

    .. math::

       \mathcal{N}(x,y)=\\frac{1}{2\pi\sigma^2}e^\\frac{-(x^2+y^2)}{2\sigma^2}


    This is done to avoid noise from the raw data (from the sensor). The
    maximum curvature method then requires we compute the first and second
    derivative of the image for all cross-sections, as per the equation above.

    We instead take the following equivalent approach:

    1. construct a gaussian filter
    2. take the first (dh/dx) and second (d^2/dh^2) deritivatives of the filter
    3. calculate the first and second derivatives of the smoothed signal using
       the results from 3. This is done for all directions we're interested in:
       horizontal, vertical and 2 diagonals. First and second derivatives of a
       convolved signal

    .. note::

       Item 3 above is only possible thanks to the steerable filter property of
       the gaussian kernel. See "The Design and Use of Steerable Filters" from
       Freeman and Adelson, IEEE Transactions on Pattern Analysis and Machine
       Intelligence, Vol. 13, No. 9, September 1991.


    @param image (numpy.ndarray) : an array of 64-bit floats containing the input image
    @param mask (numpy.ndarray) : an array, of the same size as ``image``, containing a mask (booleans) indicating where the finger is on ``image``.
    @param sigma (float) : Variance of the gaussian filter

    @return (numpy.ndarray) : a 3-dimensional array of 64-bits containing $\kappa$ for
        all considered directions. $\kappa$ has the same shape as ``image``,
        except for the 3rd. dimension, which provides planes for the
        cross-section valley detections for each of the contemplated directions,
        in this order: horizontal, vertical, +45 degrees, -45 degrees.

    """

    # 1. constructs the 2D gaussian filter "h" given the window size,
    # extrapolated from the "sigma" parameter (4x)
    # N.B.: This is a text-book gaussian filter definition
    winsize = np.ceil(4 * sigma) # enough space for the filter
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = 1.0 / (2 * np.pi * sigma ** 2)
    G *= np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 2. calculates first and second derivatives of "G" with respect to "X"
    # (0), "Y" (90 degrees) and 45 degrees (?)
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90 = G1_0.T
    G2_90 = G2_0.T
    hxy = ((X * Y) / (sigma ** 4)) * G

    # 3. calculates derivatives w.r.t. to all directions of interest
    #    stores results in the variable "k". The entries (last dimension) in k
    #    correspond to curvature detectors in the following directions:
    #
    #    [0] horizontal
    #    [1] vertical
    #    [2] diagonal \ (45 degrees rotation)
    #    [3] diagonal / (-45 degrees rotation)
    image_g1_0 = si.convolve(image, G1_0, mode = 'nearest')
    image_g2_0 = si.convolve(image, G2_0, mode = 'nearest')
    image_g1_90 = si.convolve(image, G1_90, mode = 'nearest')
    image_g2_90 = si.convolve(image, G2_90, mode = 'nearest')
    fxy = si.convolve(image, hxy, mode = 'nearest')

    # support calculation for diagonals, given the gaussian kernel is
    # steerable. To calculate the derivatives for the "\" diagonal, we first
    # **would** have to rotate the image 45 degrees counter-clockwise (so the
    # diagonal lies on the horizontal axis). Using the steerable property, we
    # can evaluate the first derivative like this:
    #
    # image_g1_45 = cos(45)*image_g1_0 + sin(45)*image_g1_90
    #             = sqrt(2)/2*fx + sqrt(2)/2*fx
    #
    # to calculate the first derivative for the "/" diagonal, we first
    # **would** have to rotate the image -45 degrees "counter"-clockwise.
    # Therefore, we can calculate it like this:
    #
    # image_g1_m45 = cos(-45)*image_g1_0 + sin(-45)*image_g1_90
    #              = sqrt(2)/2*image_g1_0 - sqrt(2)/2*image_g1_90
    #

    image_g1_45 = 0.5 * np.sqrt(2) * (image_g1_0 + image_g1_90)
    image_g1_m45 = 0.5 * np.sqrt(2) * (image_g1_0 - image_g1_90)

    # NOTE: You can't really get image_g2_45 and image_g2_m45 from the theory
    # of steerable filters. In contact with B.Ton, he suggested the following
    # material, where that is explained: Chapter 5.2.3 of van der Heijden, F.
    # (1994) Image based measurement systems: object recognition and parameter
    # estimation. John Wiley & Sons Ltd, Chichester. ISBN 978-0-471-95062-2

    # This also shows the same result:
    # http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Derivati.html (look for
    # SDGD)

    # He also suggested to look at slide 75 of the following presentation
    # indicating it is self-explanatory: http://slideplayer.com/slide/5084635/

    image_g2_45 = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90
    image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90

    # ######################################################################
    # [Step 1-1] Calculation of curvature profiles
    # ######################################################################

    # Peak detection (k or kappa) calculation as per equation (1) page 348 on
    # Miura's paper
    finger_mask = mask.astype('float64')

    return np.dstack([
        (image_g2_0 / ((1 + image_g1_0 ** 2) ** (1.5))) * finger_mask,
        (image_g2_90 / ((1 + image_g1_90 ** 2) ** (1.5))) * finger_mask,
        (image_g2_45 / ((1 + image_g1_45 ** 2) ** (1.5))) * finger_mask,
        (image_g2_m45 / ((1 + image_g1_m45 ** 2) ** (1.5))) * finger_mask,
    ])


def eval_vein_probabilities(k):

    """ Evaluates joint vein centre probabilities from cross-sections

    This function will take $\kappa$ and will calculate the vein centre
    probabilities taking into consideration valley widths and depths. It
    aggregates the following steps from the paper:

    * [Step 1-2] Detection of the centres of veins
    * [Step 1-3] Assignment of scores to the centre positions
    * [Step 1-4] Calculation of all the profiles

    Once the arrays of curvatures (concavities) are calculated, here is how
    detection works: The code scans the image in a precise direction (vertical,
    horizontal, diagonal, etc). It tries to find a concavity on that direction
    and measure its width (see Wr on Figure 3 on the original paper). It then
    identifies the centers of the concavity and assign a value to it, which
    depends on its width (Wr) and maximum depth (where the peak of darkness
    occurs) in such a concavity. This value is accumulated on a variable (Vt),
    which is re-used for all directions. Vt represents the vein probabilites
    from the paper.


    @param k (numpy.ndarray): a 3-dimensional array of 64-bits containing $\kappa$
        for all considered directions. $\kappa$ has the same shape as
        ``image``, except for the 3rd. dimension, which provides planes for the
        cross-section valley detections for each of the contemplated
        directions, in this order: horizontal, vertical, +45 degrees, -45
        degrees.

    @return (numpy.ndarray): The un-accumulated vein centre probabilities ``V``. This
        is a 3D array with 64-bit floats with the same dimensions of the input
        array ``k``. You must accumulate (sum) over the last dimension to
        retrieve the variable ``V`` from the paper.
    """

    V = np.zeros(k.shape[:2], dtype = 'float64')


    def _prob_1d(a):

        """ Finds "vein probabilities" in a 1-D signal

        This function efficiently counts the width and height of concavities in
        the cross-section (1-D) curvature signal ``s``.

        It works like this:

        1. We create a 1-shift difference between the thresholded signal and itself
        2. We compensate for starting and ending regions
        3. For each sequence of start/ends, we compute the maximum in the original signal

        Example (mixed with pseudo-code):

        a = 0 1 2 3 2 1 0 -1 0 0 1 2 5 2 2 2 1
        b = a > 0 (as type int)
        b = 0 1 1 1 1 1 0  0 0 0 1 1 1 1 1 1 1

        0 1 1 1 1 1  0 0 0 0 1 1 1 1 1 1 1
          0 1 1 1 1  1 0 0 0 0 1 1 1 1 1 1 1 (-)
        -------------------------------------------
        X 1 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 X (length is smaller than orig.)

        starts = numpy.where(diff > 0)
        ends   = numpy.where(diff < 0)

        -> now the number of starts and ends should match, otherwise, we must compensate

            -> case 1: b starts with 1: add one start in begin of "starts"
            -> case 2: b ends with 1: add one end in the end of "ends"

        -> iterate over the sequence of starts/ends and find maximums

        @param a (numpy.ndarray): 1D signal with curvature to explore

        @return (numpy.ndarray): 1D container with the vein centre probabilities
        """

        b = (a > 0).astype(int)
        diff = b[1:] - b[:-1]
        starts = np.argwhere(diff > 0)
        starts += 1 # compensates for shifted different
        ends = np.argwhere(diff < 0)
        ends += 1 # compensates for shifted different
        if b[0]:
            starts = np.insert(starts, 0, 0)
        if b[-1]:
            ends = np.append(ends, len(a))

        z = np.zeros_like(a)

        if starts.size == 0 and ends.size == 0:
            return z

        for start, end in zip(starts, ends):
            maximum = np.argmax(a[int(start):int(end)])
            z[start + maximum] = a[start + maximum] * (end - start)

        return z


    # Horizontal direction
    for index in range(k.shape[0]):
        V[index, :] += _prob_1d(k[index, :, 0])

    # Vertical direction
    for index in range(k.shape[1]):
        V[:, index] += _prob_1d(k[:, index, 1])

    # Direction: 45 degrees (\)
    curv = k[:, :, 2]
    i, j = np.indices(curv.shape)
    for index in range(-curv.shape[0] + 1, curv.shape[1]):
        V[i == (j - index)] += _prob_1d(curv.diagonal(index))

    # Direction: -45 degrees (/)
    # NOTE: due to the way the access to the diagonals are implemented, in this
    # loop, we operate bottom-up. To match this behaviour, we also address V
    # through Vud.
    # required so we get "/" diagonals correctly
    curv = np.flipud(k[:, :, 3])
    Vud = np.flipud(V) # match above inversion
    for index in reversed(range(curv.shape[1] - 1, -curv.shape[0], -1)):
        Vud[i == (j - index)] += _prob_1d(curv.diagonal(index))

    return V


def connect_centres(V):

    """ Connects vein centres by filtering vein probabilities ``V``

    This function does the equivalent of Step 2 / Equation 4 at Miura's paper.

    The operation is applied on a row from the ``V`` matrix, which may be
    acquired horizontally, vertically or on a diagonal direction. The pixel
    value is then reset in the center of a windowing operation (width = 5) with
    the following value:

    .. math::

        b[w] = min(max(a[w+1], a[w+2]) + max(a[w-1], a[w-2]))


    @param V (numpy.ndarray): The accumulated vein centre probabilities ``V``. This
        is a 2D array with 64-bit floats and is defined by Equation (3) on the
        paper.

    @return (numpy.ndarray): A 3-dimensional 64-bit array ``Cd`` containing the result
        of the filtering operation for each of the directions. ``Cd`` has the
        dimensions of $\kappa$ and $V_i$. Each of the planes correspond to the
        horizontal, vertical, +45 and -45 directions.
    """


    def _connect_1d(a):

        """ Connects centres in the given vector

        The strategy we use to vectorize this is to shift a twice to the left and
        twice to the right and apply a vectorized operation to compute the above.

        @param a (numpy.ndarray): Input 1D array which will be window scanned

        @return numpy.ndarray: Output 1D array (must be writable), in which we will
            set the corrected pixel values after the filtering above. Notice that,
            given the windowing operation, the returned array size would be 4 short
            of the input array.

        """

        return np.amin([np.amax([a[3:-1], a[4:]], axis = 0), np.amax([a[1:-3], a[:-4]], axis = 0)], axis = 0)


    Cd = np.zeros(V.shape + (4,), dtype = 'float64')

    # Horizontal direction
    for index in range(V.shape[0]):
        Cd[index, 2:-2, 0] = _connect_1d(V[index, :])

    # Vertical direction
    for index in range(V.shape[1]):
        Cd[2:-2, index, 1] = _connect_1d(V[:, index])

    # Direction: 45 degrees (\)
    i, j = np.indices(V.shape)
    border = np.zeros((2,), dtype = 'float64')
    for index in range(-V.shape[0] + 5, V.shape[1] - 4):
        # NOTE: hstack **absolutely** necessary here as double indexing after
        # array indexing is **not** possible with np (it returns a copy)
        Cd[:, :, 2][i == (j - index)] = np.hstack([border, _connect_1d(V.diagonal(index)), border])

    # Direction: -45 degrees (/)
    Vud = np.flipud(V)
    Cdud = np.flipud(Cd[:, :, 3])
    for index in reversed(range(V.shape[1] - 5, -V.shape[0] + 4, -1)):
        # NOTE: hstack **absolutately** necessary here as double indexing after
        # array indexing is **not** possible with np (it returns a copy)
        Cdud[:, :][i == (j - index)] = np.hstack([border, _connect_1d(Vud.diagonal(index)), border])

    return Cd


def binarise(G):

    """ Binarise vein images using a threshold assuming distribution is diphasic

    This function implements Step 3 of the paper. It binarises the 2-D array
    ``G`` assuming its histogram is mostly diphasic and using a median value.

    @param G (numpy.ndarray): A 2-dimensional 64-bit array ``G`` containing the
        result of the filtering operation. ``G`` has the dimensions of the original image.

    @return (numpy.ndarray): A 2-dimensional 64-bit float array with the same 
        dimensions of the input image, but containing its vein-binarised version.
        The output of this function corresponds to the output of the method.

    """

    median = np.median(G[G > 0])
    Gbool = G > median
    return Gbool.astype(np.float64)


def preprocess(data):

    """ Preprocesses an image given in the form of a numpy array
    The preprocessing steps were chosen from bob's api to mimic the original
    extract_features script which used bob's library.

    @param data (numpy.ndarray) : the numpy array to be preprocessed

    @return data (numpy.ndarray) : the preprocessed image
    @return mask (numpy.ndarray of bool) : the mask representing the finger
    """

    # We were using NoCrop, so omitted
    # We were using LeeMask, height 40 width 4
    mask = leemask(data, filter_height = 40, filter_width = 4)
    # We were not using any normalization, so omitted
    # We were using histogram equalization
    data = histogram_equalization(data, mask)

    return data, mask


def maximum_curvature(image, mask, sigma):

    """ Extracts an image (given in the form of a numpy array)'s fingerveins
    The extracting steps were chosen from bob's api to mimic the original
    extract_features script which used bob's library.

    @param image (numpy.ndarray) : The numpy array to extract veins from
    @param mask (numpy.ndarray) : The mask representing the finger
    @param sigma (float) : The sigma parameter for detect_valleys

    @return (numpy.ndarray of float64 0's and 1's) : The extracted image
    """

    finger_image = image.astype('float64')

    start = time.time()
    kappa = detect_valleys(finger_image, mask, sigma)
    log.info('filtering took %.2f seconds' % (time.time() - start))

    start = time.time()
    V = eval_vein_probabilities(kappa)
    log.info('probabilities took %.2f seconds' % (time.time() - start))

    start = time.time()
    Cd = connect_centres(V)
    log.info('connections took %.2f seconds' % (time.time() - start))

    start = time.time()
    retval = binarise(np.amax(Cd, axis = 2))
    log.info('binarization took %.2f seconds' % (time.time() - start))

    return retval


def miurascore(model, probe, ch = 30, cw = 90):

    """ Computes the score between the probe and the model.

    @param model (numpy.ndarray): The model of the user to test the probe against
    @param probe (numpy.ndarray): The probe to test
    @param ch (int) : Maximum search displacement in y-direction.
    @param cw (int) : Maximum search displacement in x-direction.

    @return (float): Value between 0 and 0.5, larger value means a better match
    """

    I = probe.astype(np.float64)

    if len(model.shape) == 2:
        model = np.array([model])

    scores = []

    # iterate over all models for a given individual
    for md in model:
        # erode model by (ch, cw)
        R = md.astype(np.float64)
        h, w = R.shape # same as I
        crop_R = R[ch:h - ch, cw:w - cw]

        # correlates using scipy - fastest option available iff the self.ch and
        # self.cw are height (>30). In this case, the number of components
        # returned by the convolution is high and using an FFT-based method
        # yields best results. Otherwise, you may try  the other options bellow
        # -> check our test_correlation() method on the test units for more
        # details and benchmarks.
        Nm = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')
        # 2nd best: use convolve2d or correlate2d directly;
        # Nm = sp.convolve2d(I, np.rot90(crop_R, k=2), 'valid')
        # 3rd best: use correlate2d
        # Nm = sp.correlate2d(I, crop_R, 'valid')

        # figures out where the maximum is on the resulting matrix
        t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

        # this is our output
        Nmm = Nm[t0, s0]

        # normalizes the output by the number of pixels lit on the input
        # matrices, taking into consideration the surface that produced the
        # result (i.e., the eroded model and part of the probe)
        scores.append(Nmm / (crop_R.sum() + I[t0:t0 + h - 2 * ch, s0:s0 + w - 2 * cw].sum()))

    return np.mean(scores)