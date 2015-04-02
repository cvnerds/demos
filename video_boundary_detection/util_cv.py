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



def load_image(path):
    """
        Load image located at given path
    """
    if not os.path.exists(path):
        logger.error('Trying to read image, but path does not exist at {}'.format(path))
    mat = cv2.imread(path)
    if mat.size == 0:
        logger.error('Image empty for file at {}'.format(path))
    return mat




def display_image(img, title='Debug', wait=True, windowScale=1.0, close=False, createWindow=True):
    """
    Scales an image then displayes it using OpenCV's inbuilt methods.

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



