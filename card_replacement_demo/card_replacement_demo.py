#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Homography image replacement demo.
"""

# built-in packages
import argparse
import logging
import os
import string
import sys
import time
import timeit
from functools import partial
# 3rd party packages
import csv
import cv2
import numpy
# custom packages
import util_csv
import util_cv





__author__ = "Eduard Feicho"
__copyright__ = "Copyright 2015"



logger = logging.getLogger(__name__)





class Frame():
    """
        Models a video frame
    """
    def __init__(self, index = 0, name = 'N/A'):
        self.index = index
        self.name  = name




class DemoApp():
    """
        Homography image replacement demo.

        This demo expects as input a folder with an image sequence.
        A reference image and a replacement image of the same size need to be provided (default: reference.jpg, replacement.jpg).
        This is a feature-based image matching demo that works for a planar object.
        The object (reference card) in the video is replaced by the replacement image.
        Think augmented reality.
    """



    def __init__(self):

        # Setup an empty args object, which contains the same variables as the command line argument parser would produce.
        # This way we can either run the python script from command line or alternatively instantiate the class as a module
        # and adjust settings using the args object
        self.args                  = lambda: None
        self.args.logfile          = None
        self.args.verbose          = 0
        self.args.debug            = False

        self.args.input            = './sequence'
        self.args.input_format     = 'seq_%08d.jpg'
        self.args.output           = None
        # self.args.output           = './output'

        self.args.reference        = 'reference.jpg'
        self.args.replacement      = 'replacement.jpg'

        self.args.matcher          = 'flann'


    
    def run(self):
        """
            Main entry point of the app
        """

        ## some initial setup
        
        # parse the input
        self.args.input = self.expandPath(self.args.input)
        self.args.output = self.expandPath(self.args.output)

        # list all frames from directory
        self.names = self.listSequence(path=self.args.input)
        self.paths = self.expandPaths(base=self.args.input, files=self.names)

        # load reference images
        self.im_reference = util_cv.load_image(self.args.reference)
        self.im_replacemt = util_cv.load_image(self.args.replacement)
        self.im_reference_gray = cv2.cvtColor(self.im_reference, cv2.COLOR_BGR2GRAY)
        self.im_replacemt_gray = cv2.cvtColor(self.im_replacemt, cv2.COLOR_BGR2GRAY)
        
        # initialise modules
        self.detector = self.initFeatureDetector()
        self.matcher  = self.initFeatureMatcher()
        
        # compute features for reference image
        self.keypoints_ref, self.descriptors_ref = self.computeFeatures(image=self.im_reference_gray)

        ## main loop (sequential algorithm)
        self.runLoop(self.names, self.paths)



    def runLoop(self, files, paths):
        """
            Process input frame-by-frame
        """

        logger.info(' * Analysing image sequence')

        if len(paths) == 0:
            logger.warn(' Empty sequence, nothing to be done')
            return
        
        # sequence of frames
        frames = []
        for (i, path) in enumerate(paths):

            logger.info('')
            logger.info(' * frame {} : {}'.format(i, path) )

            # set up frame object
            frame = Frame(index=i, name=files[i])
            frames.append(frame)

            # load frame image
            frame.image = util_cv.load_image(path)
            frame.image_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            # detect and compute features
            frame.keypoints, frame.descriptors = self.computeFeatures(image=frame.image_gray)

            # match feature descriptors
            frame.matches = self.matchFeatures(frame.descriptors)

            # debug features
            if self.args.debug:
                debugImage = util_cv.drawMatches(self.im_reference, self.keypoints_ref, frame.image, frame.keypoints, frame.matches)
                util_cv.display_image(debugImage, title='DEBUG')

            ## compute homography from the reference object to the scene
            frame.homography = self.computeHomography(self.keypoints_ref, frame.keypoints, frame.matches)

            # debug homography 
            if self.args.debug:
                debugImage = frame.image.copy()
                util_cv.draw_homography(self.im_reference, frame.homography.astype('float32'), debugImage)
                debugImage = util_cv.drawMatches(self.im_reference, self.keypoints_ref, debugImage, frame.keypoints, frame.matches)
                util_cv.display_image(debugImage, title='DEBUG')


            ## check if homography is bad
            bad_homo = False
            # If the det of the upper 2x2 matrix of the homography is smaller than 0, it is not orientation-preserving.
            # src: http://answers.opencv.org/question/2588/check-if-homography-is-good/
            # This filters out some bad frames, although it's not a perfect criteria (tested with the bad results using BruteForceMatcher)
            det2x2 = cv2.determinant(frame.homography[0:2,0:2])
            if det2x2 < 0:
                bad_homo = True

            # This criteria can be found in the image stitching pipeline of opencv...
            det3x3 = abs(cv2.determinant(frame.homography))
            if det3x3 < 0.00001:#sys.float_info.epsilon:
                bad_homo = True

            if bad_homo:
                
                # homography is useless, just show the original picture
                frame.im_replacemt = frame.image.copy()

            else:

                # warp replacement image and merge with the frame
                h,w = frame.image.shape[:2]
                im_warped = cv2.warpPerspective(self.im_replacemt, frame.homography, dsize=(w,h))
                im_replacemt = numpy.ma.array(frame.image, mask=im_warped > 0, fill_value=0).filled()
                im_replacemt = cv2.bitwise_or(im_replacemt, im_warped)
                frame.im_replacemt = im_replacemt


            # debug result
            if self.args.debug:
                color = (255,0,0)
                if det2x2 < 0 or det3x3 < 0.00001:
                    color = (0,0,255)
                cv2.putText(frame.im_replacemt, 'Frame {}: {}'.format(i, os.path.basename(path)), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
                cv2.putText(frame.im_replacemt, 'det2x2 = {}'.format(det2x2), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                cv2.putText(frame.im_replacemt, 'det3x3 = {}'.format(det3x3), (0,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                util_cv.display_images([frame.image, frame.im_replacemt], title='Processed Frame')

            # write image
            if self.args.output:
                output = path.replace(self.args.input, self.args.output)
                logger.info( ' * Write {}'.format(output) )
                cv2.imwrite( output, frame.im_replacemt )

            # clear memory
            frames.pop(0)



    def listSequence(self, path):
        """
            Given a path, assume it's a folder and return a sorted list of the folder contents
        """

        logger.info(' * Read input sequence from directory: {}'.format(path))

        if not os.path.isdir(path):
            logger.error('path is not a directory: {}'.format(path))
            return []

        files = os.listdir(path)
        # TODO filter only valid entries using self.args.input_format
        files = filter(lambda file: ".jpg" in file, files)
        files = sorted(files)

        logger.info(' * Found {} files'.format(len(files)))

        return files



    def expandPath(self, path):
        if path is None:
            return None
        return os.path.abspath(os.path.expanduser(path))



    def expandPaths(self, base, files):
        """
            expand user home folder and convert to absolute paths
        """
        files = map(lambda p: self.expandPath( os.path.join(base, p) ), files)
        return files



    def initFeatureDetector(self):
        return cv2.SURF(hessianThreshold=400, nOctaves=4, nOctaveLayers=2, extended=True, upright=False)



    def initFeatureMatcher(self):
        if self.args.matcher == 'bruteforce':
            return cv2.BFMatcher(normType=cv2.NORM_L2)
        else:
            
            # bug: flann enums are missing
            # workaround based on src: http://stackoverflow.com/questions/8301962/opencv-pythons-api-flannbasedmatcher

            FLANN_INDEX_KDTREE = 1
            indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
            return cv2.FlannBasedMatcher(indexParams=indexParams, searchParams={})



    def computeFeatures(self, image):
        """
            Detect interesting keypoints and compute a set of features
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, mask=None)
        return keypoints, descriptors



    def matchFeatures(self, features_query):
        """
            Match feature vectors against reference features
        """
        if self.args.matcher == 'flann':
            # bug: flann enums are missing
            matches = self.matcher.knnMatch(self.descriptors_ref, features_query, k=2)
            # store all the good matches as per Lowe's ratio test.
            matches = self.applyRatioTest(matches)
        else:
            matches = self.matcher.match(self.descriptors_ref, features_query)

        return matches





    def applyRatioTest(self, matches, ratio=0.75):
        """
            Ratio test from D.Lowe's SIFT paper
        """

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < ratio*n.distance:
                good.append(m)
        return good



    def computeHomography(self, keypoints_src, keypoints_dst, good_matches):
        """
            Compute homography given keypoints and matches
        """
        # Localize the object

        src_pts = numpy.float32([ keypoints_src[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = numpy.float32([ keypoints_dst[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        homography, mask = cv2.findHomography( src_pts, dst_pts, cv2.cv.CV_RANSAC, 5.0)

        logger.debug('H={}'.format(homography))

        return homography





def main():

    # initialize the app (contains default values for some command line arguments)
    app = DemoApp()

    # setup command line argument parser
    parser = argparse.ArgumentParser(description="Homography image replacement demo.")
    parser.add_argument("input",                metavar="INPUT_FOLDER",                                                              help="Path to input sequence folder")
    parser.add_argument("-o", "--output",       dest="output",          default=app.args.output,                                     help="Path to output sequence folder")
    parser.add_argument("--reference",          dest="reference",       default=app.args.reference,                                  help="Path to reference image")
    parser.add_argument("--replacement",        dest="replacement",     default=app.args.replacement,                                help="Path to replacement image")
    parser.add_argument("--format",             dest="input_format",    default=app.args.input_format,                               help="Format of input sequence")
    parser.add_argument("--matcher",            dest="matcher",         default=app.args.matcher,                                    help="Matcher (flann|bruteforce|...)")
    
    # various arguments related to logging/debugging
    parser.add_argument("-l", "--logfile",      dest="logfile",         default=None,                                                help="Path to logfile")
    parser.add_argument("-v", "--verbose",      dest="verbose",         default=1,      action="count",                              help="verbosity level (use -vv to increase more)")

    parser.add_argument("-q", "--quiet",        dest="quiet",           default=False,  action="store_true",                         help="don't print status messages to stdout")
    #parser.add_argument("--test",               dest="test",            default=False,  action="store_true",                         help="run unit tests (TODO)")
    parser.add_argument("--debug",              dest="debug",           default=False,  action="store_true",                         help="enable debugging mode")
    parser.add_argument("--evaluate",           dest="evaluate",        default=False,  action="store_true",                         help="enable evaluation mode")

    # parse command line arguments
    args = parser.parse_args()
    if args.quiet: args.verbose = 0
    if args.debug: args.verbose = 2

    # setup logging from command line arguments
    setupLogging(args, logger)

    # use command line arguments to specify the configuration of the app
    app.args = args

    # the main entry point of the app
    p = partial(app.run)
    time_seconds = timeit.timeit( p, number=1)
    logger.info(' * Algorithm took {:.2f} seconds for {:5d} frames ({:.2f} FPS)'.format(time_seconds, len(app.names), float(len(app.names))/time_seconds))



def setupLogging(args, logger, basicFormat='%(levelname)s:\t%(message)s', fileFormat='%(asctime)s\t%(levelname)s:\t%(message)s', loglevel=None):
    """
        a basic logging setup based on some assumptions of the given command line arguments
    """

    # if logging level is not explcitly specified, chose based on verbosity level specified on command line
    if loglevel is None:
        loglevel = logging.WARNING                            # default to log level WARNING
        if args.verbose >= 1: loglevel = logging.INFO         # specify for example -v or --verbose
        if args.verbose >= 2: loglevel = logging.DEBUG        # specify for example -vv

    logging.basicConfig(format=basicFormat, level=loglevel)

    # write to file, if -l/--logfile specified on command line
    if args.logfile is not None:
        handler = logging.FileHandler(args.logfile)
        formatter = logging.Formatter(fileFormat)
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)



if __name__ == "__main__":
    main()
