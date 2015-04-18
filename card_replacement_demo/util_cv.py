#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This is a utility class with useful utility functions related to Computer Vision

    TODO: histogram visualisation
"""





# built-in packages
import csv
import cv2
import logging
import os
import string
import numpy
# 3rd party packages
import cv2
import numpy
import skimage.feature



__author__ = "Eduard Feicho"
__copyright__ = "Copyright 2015"



logger = logging.getLogger(__name__)






# TODO: histogram visualisation
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np

# from skimage import data, img_as_float
# from skimage import exposure
# matplotlib.rcParams['font.size'] = 8

# from http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html#histogram-equalization
# def plot_img_and_hist(self, img, axes, bins=256, xlabel='Pixel intensity'):
#     """
#         Plot an image along with its histogram and cumulative histogram.
#     """
#     img = img_as_float(img)
#     ax_img, ax_hist = axes
#     ax_cdf = ax_hist.twinx()

#     # Display image
#     ax_img.imshow(img, cmap=plt.cm.gray)
#     ax_img.set_axis_off()

#     # Display histogram
#     (n, bins, patches) = ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
#     ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#     ax_hist.set_xlabel(xlabel)
#     ax_hist.set_xlim(0, 1)
#     ax_hist.set_yticks([])

#     # Display cumulative distribution
#     img_cdf, bins = exposure.cumulative_distribution(img, bins)
#     ax_cdf.plot(bins, img_cdf, 'r')
#     ax_cdf.set_yticks([])

#     return ax_img, ax_hist, ax_cdf


# OpenCV2 is lacking this function in the python interface...
# Source: http://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python
# OpenCV3 will support it again, follow this tutorial:
# https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = numpy.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    if len(img1.shape) == 2 or img1.shape[2] == 1:
        out[:rows1,:cols1,:] = numpy.dstack([img1, img1, img1])
    else:
        out[:rows1,:cols1,:] = img1

    # Place the next image to the right of it
    if len(img2.shape) == 2 or img2.shape[2] == 1:
        out[:rows2,cols1:cols1+cols2,:] = numpy.dstack([img2, img2, img2])
    else:
        out[:rows2,cols1:cols1+cols2,:] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out



def draw_homography(img1, M, img2):
    h,w = img1.shape[:2]
    pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[numpy.int32(dst)],True,(0, 255, 0),3, cv2.CV_AA)



def load_image(path, options=cv2.CV_LOAD_IMAGE_COLOR):
    """
        Load image located at given path
    """
    if not os.path.exists(path):
        logger.error('Trying to read image, but path does not exist at {}'.format(path))
    mat = cv2.imread(path, options)
    if mat.size == 0:
        logger.error('Image empty for file at {}'.format(path))
    return mat



def display_images(images, title='Debug', wait=True, windowScale=1.0, close=False, createWindow=True):
    """
        Scales an image then displays it using OpenCV's inbuilt methods.

        :param img: input image
        :param title: window title
        :param wait: waitKey?
        :param windowScale: scale image
    """
    # TODO support non-equally-sized images by padding zeros

    im_stacked = numpy.hstack( tuple(images) )
    display_image(im_stacked, title, wait, windowScale, close, createWindow)



def display_image(img, title='Debug', wait=True, windowScale=1.0, close=False, createWindow=True):
    """
        Scales an image then displays it using OpenCV's inbuilt methods.

        :param img: input image
        :param title: window title
        :param wait: waitKey?
        :param windowScale: scale image
    """

    if windowScale != 1.0:
        height = int(round(img.shape[0] * windowScale))
        width = int(round(img.shape[1] * windowScale))
        img = cv2.resize(img, (width, height))
    
    if createWindow:
        cv2.namedWindow(title)
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey()
    if close:
        cv2.destroyWindow(title)



def compute_edges(image, sigma=1.2, low_threshold=24):

    assert (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[2] == 1)

    return skimage.feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=None, mask=None)
    #return cv2.Canny(image, threshold1, threshold1*3) # alternatively use opencv




def compute_color_hist(image, bins=4):
    """
        Compute a color histogram from given image

        :param image: The image
        :param bins:  The number of bins
    """

    assert len(image.shape) >= 2
    
    
    # image dimensions
    h,w = image.shape[:2]

    # assess number of channels and set colors accordingly
    channels = ['black']
    if len(image.shape) == 3:
        channels = ['b','g','r'][0:image.shape[2]]

    # compute histogram for each channel
    histogram = []
    for i, color in enumerate(channels):
        hist_channel = cv2.calcHist(images=[image], channels=[i], mask=None, histSize=[bins], ranges=[0, 256])
        # normalise
        hist_channel = hist_channel.astype('float') / float(h*w)

        # append each channel histogram to the final histogram
        histogram = numpy.append(histogram, hist_channel)

        ## TODO Visualisation for debugging purposes
        # if self.args.debug:
        #     plot.hist(histogram, color=color)
        #     plot.xlim([0,bins])

        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
        # ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(image, axes[:, 0], bins=bins)
        # ax_img.set_title('Low contrast image')
        # # y_min, y_max = ax_hist.get_ylim()
        # #ax_hist.set_ylabel('Number of pixels')
        # #ax_hist.set_yticks(np.linspace(0, y_max, 5))
        # ax_cdf.set_ylabel('Fraction of total intensity')
        # ax_cdf.set_yticks(np.linspace(0, 1, 5))
    

    # logger.debug(' color histogram : {}'.format(histogram))

    return numpy.array(histogram).astype('float32')



def binary_dilation(binary_image, element='rectangle', size=1):
    """
        Compute a binary dilation of given binary image

        :param binary_image: The image as a binary numpy array
        :param element: type of element. Supported: diamond, rectangle. As a fallback and default, rectangle will be used.
        :param size: the size of the structuring element. The interpretation depends on the element. For example could be radius or width and height.
    """

    if element == 'diamond':
        selem = skimage.morphology.diamond(radius=size)
    else:
        selem = skimage.morphology.rectangle(width=size, height=size)

    return skimage.morphology.binary_dilation(binary_image, selem=selem)



