#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Video shout boundary detection demo

    TODO: move color histogram computation code from ShotBoundaryDetectorEdgeRatio into SceneMatcher
    TODO: plot histograms for debugging purposes
    TODO: ShotBoundaryDetectorEdgeRatio: Detect fades by looking at previous frames
    TODO: SceneMatcher: treat fades differently by looking at start and end frame and potentially merging/replacing previously detected hard keyframes
    TODO: evaluation framework for csv files created with --evaluate switch
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
    def __init__(self, index = 0, name = 'N/A'):
        self.index = index
        self.name  = name



class Keyframe():
    def __init__(self, index = 0, name = 'N/A', scene_start = '-1', scene_end = '-1',  scene_id = -1, frame_cut_type = 'hard'):
        
        self.index          = index
        self.name           = name
        self.scene_start    = scene_start
        self.scene_end      = scene_end
        self.scene_id       = scene_id
        self.frame_cut_type = frame_cut_type


    def __init__(self, frame, scene_end = '-1', scene_id = 0, frame_cut_type = 'hard'):
        self.index          = frame.index
        self.name           = frame.name
        self.scene_start    = frame.name
        self.scene_end      = scene_end
        self.scene_id       = scene_id
        self.frame_cut_type = frame_cut_type
        self.frame          = frame



class ShotBoundaryDetectorCHD():
    """
        Detect scene cuts using color histogram difference measure [1].

        [1] Rainer Lienhart - Comparison of Automatic Shot Boundary Detection Algorithms, Microcomputer Research Labs, Intel Corporation, Santa Clara, CA 95052-8819, 1999
    """

    def __init__(self):
        self.args                  = lambda: None
        # threshold of classifier
        self.args.chd_thresh       = 0.03               # manually chosen for dataset "frames"

    def process_frame(self, frame):
        """
            Process a frame
            :param frame: The frame
            :type frame:  Frame
        """
        assert isinstance(frame, Frame)

        frame.colorhist = util_cv.compute_color_hist(frame.image)

    def detect_cut(self, index, frames):
        """
            Detect a cut and return a Keyframe object
        """

        assert type(index) is int
        assert type(frames) is list
        assert isinstance(index, int)
        assert isinstance(frames, list)

        frame = frames[index]
        keyframe = []

        if index == 0:
            # initial frame is always key frame of first scene
            frame.CHD = 0             # kind of irrelevant
            return [Keyframe(frame)]

        else:

            # compute color histogram difference
            frame.CHD = self.computeCHD(histogramA=frames[index-1].colorhist, histogramB=frame.colorhist)

            if frame.CHD > self.args.chd_thresh:
                # new hard cut
                return [Keyframe(frame)]

        return []

    def computeCHD(self, histogramA, histogramB):
        """
            Analyse color histogram differences to compute shot boundaries.
        """

        ## compute color histogram differences between all frame pairs

        assert (len(histogramA) == len(histogramB))
        assert (len(histogramA) > 0)

        # compute color histogram difference
        CHD = 0
        # N = 0

        CHD = numpy.abs(histogramA - histogramB).sum() / float(len(histogramA))

        # # note, each histogram can be a list of histograms for each channel
        # for bin in range( len(histA) ):
        #     # loop through each histogram bin
        #     CHD += abs(histA[bin] - histB[bin])
        #     N += 1

        # CHD = float(CHD) / float(N)

        logger.debug(' * Color Histogram Difference {}'.format(CHD))
        
        return CHD




class ShotBoundaryDetectorEdgeRatio():
    """
        Detect boundaries using edge ratio measure [1].
        Details and constants found in [2].

        [1] Rainer Lienhart - Comparison of Automatic Shot Boundary Detection Algorithms, Microcomputer Research Labs, Intel Corporation, Santa Clara, CA 95052-8819, 1999
        [2] Zabih, Miller, Mai - A FeatureBased Algorithm for Detecting and Classifying Scene Breaks, 1995
    """

    def __init__(self):
        self.args                   = lambda: None
        # Canny algorithm Gaussian sigma 
        self.args.canny_sigma       = 1.2               # see [2]
        # Canny algorithm lower threshold
        self.args.canny_thresh      = 24                # see [2]
        # Radius of diamond shaped dilation kernel used to make edge ratio robust against small motion.
        self.args.dilation_radius   = 6                 # see [1]
        # Threshold of classifier
        self.args.edge_ratio_thresh = 10                 # TODO
        # number of previous frames to include when detecting fades
        self.args.fade_window_size  = 20                # manually chosen from looking at groundturth of dataset

        # toggle debugging
        self.args.debug             = False



    def process_frame(self, frame):
        """
            Process a frame.
            :param frame: The frame
            :type frame:  Frame
        """
        assert isinstance(frame, Frame)


        # TODO: this is actually only necessary for the scene recognition part,
        # so I'd rather have it insde the SceneMatcher like: if has_attribute(o, 'colorhist') ... something like that
        frame.colorhist = util_cv.compute_color_hist(frame.image)

        frame.imageGray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        # edge image needed to compate edge ratio
        frame.edges     = util_cv.compute_edges(frame.imageGray, low_threshold=self.args.canny_thresh)

        # dilation needed for comparing leaving and entering pixels
        frame.dilated   = util_cv.binary_dilation(frame.edges / 255, element='diamond', size=self.args.dilation_radius) * 255


        if self.args.debug:
            util_cv.display_image(frame.imageGray, title='Image in grayscale')
            util_cv.display_image(frame.edges, title='Edge Image')
            util_cv.display_image(frame.dilated, title='Dilated Image')



    def detect_cut(self, index, frames):
        """
            Detect scene cut or fade in the current frame

            :param index:  index of current frame
            :param frames: list of frames
            :type frames: list of Frame

            :return: list containing a Keyframe if a cut is detected
        """

        assert type(index) is int
        assert type(frames) is list

        frame = frames[index]

        if index == 0:
            frame.edgeRatio = 0

            # setup keyframe object
            return [Keyframe(frame)]

        # compute edge ratio
        frame.edgeRatio = self.computeEdgeRatio(frame0=frames[index-1], frame1=frame)

        if frame.edgeRatio > self.args.edge_ratio_thresh:
            # new shot boundary
            
            # setup keyframe object
            return [Keyframe(frame)]

        return []



    def computeEdgeRatio(self, frame0, frame1):
        """
            Compute the edge ratio, which is defined as the max ratio of leaving vs. entering edge pixels.

            In the original paper the camera motion of consecutive frames is compensated using a Hausdorff distance measure (TODO).
            Without motion compensation the method should still give reasonable results with videos where the camera doesn't move (much)
            within a scene.

            Note, edge pixels are not considered leaving/entering if there is another edge pixels in the other frame within a fixed distance.
            This is implemented by comparing with a dilated image of the other frame where the structuring element is a diamond with the wanted radius.

            :param frame0: previous frame
            :type frame0: Frame
            :param frame1: current frame
            :type frame1: Frame
        """

        # TODO motion compensation frame0.edges = frame0.edges + motion

        sigma0 = (frame0.edges > 0 ).sum()
        sigma1 = (frame1.edges > 0 ).sum()

        # X0out = ( (frame0.edges - (frame0.edges & frame1.edges)) > 0 ).sum()
        # X1in  = ( (frame1.edges - (frame1.edges & frame0.edges)) > 0 ).sum()
        X0out = ( (frame0.edges - (frame0.edges & frame1.dilated)) > 0 ).sum()
        X1in  = ( (frame1.edges - (frame1.edges & frame0.dilated)) > 0 ).sum()

        ECR = max( float(X1in) / float(sigma1), float(X0out) / float(sigma0) )
        
        return ECR



class SceneMatcher():
    """
        Compare keyframes and classify if they match using color histograms.
    """

    def __init__(self):
        self.args = lambda: None
        # toggle debugging
        self.args.debug = False
        # matching distance for classifier (chi-square distance of color histograms)
        self.args.match_dist = 0.13                 # manually chosen



    def analyse(self, keyframes, new_keyframe):
        """
            Analyse if new keyframes match existing ones.
            Matching is based on color histogram.

            :param keyframes: list of existing keyframes
            :param new_keyframe: the newly detected keyframe
        """


        if len(keyframes) == 0:
            # TODO quick hack
            if self.args.evaluate:
                new_keyframe.frame.chisqr_dist = 0
            keyframes.append(new_keyframe)
            return

        if new_keyframe.frame_cut_type == 'hard':
            self.analyseHardCut(keyframes=keyframes, new_keyframe=new_keyframe)
        else:
            # TODO: treat fades differently by looking at start and end frame and potentially merging/replacing previously detected hard keyframes
            pass


    def analyseHardCut(self, keyframes, new_keyframe):
        """
            Analyse if new keyframes match existing ones.
            Matching is based on color histogram.

            :param keyframes: list of existing keyframes
            :param new_keyframe: the newly detected keyframe
        """

        # store maximum known scene while iterating over keyframes
        scene_max_id = -1

        ## compute chi square distance of color histograms for all previous keyframes
        hist_b = new_keyframe.frame.colorhist
        dist_hist = []

        
        for keyframe in keyframes:

            scene_max_id = max(scene_max_id, keyframe.scene_id)

            hist_a = keyframe.frame.colorhist
            dist_hist.append( cv2.compareHist(hist_a, hist_b, cv2.cv.CV_COMP_CHISQR) )

        idx_match = numpy.argmin(dist_hist)

        ## classify match
        if dist_hist[idx_match] < self.args.match_dist:
            # found match
            new_keyframe.scene_id = keyframes[idx_match].scene_id
        else:
            # no match, new scene
            new_keyframe.scene_id = scene_max_id + 1

        keyframes.append(new_keyframe)

        # store distance for evaluation purposes
        if self.args.evaluate:
            new_keyframe.frame.chisqr_dist = dist_hist[idx_match]


    def analyse_hard_cut(self, keyframes, new_keyframe):
        """
            Analyse if new keyframes match existing ones.
            Matching is based on color histogram.

            :param keyframes: list of existing keyframes
            :param new_keyframe: the newly detected keyframe
        """


        




class DemoApp():
    """
        Demo application
    """

    def __init__(self):

        # Setup an empty args object, which contains the same variables as the command line argument parser would produce.
        # This way we can either run the python script from command line or alternatively instantiate the class as a module
        # and adjust settings using the args object
        self.args                  = lambda: None
        self.args.logfile          = None
        self.args.verbose          = 0
        self.args.debug            = False
        self.args.evaluate         = False

        self.args.input            = './data/frames'
        self.args.input_format     = '%06d.jpg'
        #self.args.output           = None
        self.args.use_chd          = False
        self.args.chd_thresh       = 0.03
        self.args.use_edge_ratios  = False
        self.args.canny_sigma      = 1.2
        self.args.canny_thresh     = 24

        # matching distance for scene matcher
        self.args.match_dist = 0.13                 # manually chosen
        


    def run(self):
        """
            Main entry point of the app
        """

        # some initial setup
        self.shotBoundaryDetector = self.initShotBoundaryDetector()
        self.sceneMatcher         = self.initSceneMatcher()

        # parse the input
        self.names = self.listSequence(path=self.args.input)
        self.paths = self.expandPaths(base=self.args.input, files=self.names)

        # call the main run loop (sequential algorithm)
        self.runLoop(self.names, self.paths)



    def initShotBoundaryDetector(self):
        """
            initialise the boundary cut detectuion
        """
        if self.args.use_edge_ratios:
            shotBoundaryDetector = ShotBoundaryDetectorEdgeRatio()
            shotBoundaryDetector.canny_sigma  = self.args.canny_sigma
            shotBoundaryDetector.canny_thresh = self.args.canny_thresh
        else:
            shotBoundaryDetector = ShotBoundaryDetectorCHD()
            shotBoundaryDetector.chd_thresh = self.args.chd_thresh
        shotBoundaryDetector.debug = self.args.debug
        return shotBoundaryDetector


    def initSceneMatcher(self):
        sceneMatcher = SceneMatcher()
        sceneMatcher.args = self.args # lazy copy of arguments
        return sceneMatcher


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
        files = sorted(files)

        logger.info(' * Found {} files'.format(len(files)))

        return files



    def expandPaths(self, base, files):
        files = map(lambda p: os.path.abspath(os.path.expanduser(os.path.join(base, p))), files)
        return files



    def runLoop(self, files, paths):
        """
            Process input frame-by-frame
        """

        logger.info(' * Analysing image sequence')

        if len(paths) == 0:
            logger.warn(' Empty sequence, nothing to be done')
            return

        ## assume at least one image present


        # sequence of frames
        frames = []
        # store information about a scene in keyframes
        keyframes = []

        for (i, path) in enumerate(paths):

            # setup frame object
            frame = Frame(index=i, name=files[i])
            frames.append(frame)

            frame.image = util_cv.load_image(path)

            self.shotBoundaryDetector.process_frame(frame)

            new_keyframes = self.shotBoundaryDetector.detect_cut(i, frames)

            for new_keyframe in new_keyframes:
                self.sceneMatcher.analyse(keyframes, new_keyframe)
                logger.info(' Shot boundary {:5d} starts at keyframe {}'.format(new_keyframe.scene_id, new_keyframe.name))

            # if i > 0 and not self.args.evaluate and not self.args.use_edge_ratios:
            #     # no need for all previous frames
            #     #  - unless we want to evaluate scores
            #     #  - unless we identify fades using edge ratios
            #     frames.pop(0)

        if self.args.evaluate:
            if not self.args.use_edge_ratios:
                util_csv.write_csv(path='evaluation.chd.plot.csv', rows=map(lambda keyframe:[keyframe.name, '{:.5f}'.format(keyframe.frame.CHD)], keyframes), header=['#frame, CHD'])
            else:
                util_csv.write_csv(path='evaluation.edgeRatio.plot.csv', rows=map(lambda keyframe:[keyframe.name, '{:.5f}'.format(keyframe.frame.edgeRatio)], keyframes), header=['#frame, edgeRatio'])
            util_csv.write_csv(path='evaluation.chisqr_dist.plot.csv', rows=map(lambda keyframe:[keyframe.name, '{:.5f}'.format(keyframe.frame.chisqr_dist)], keyframes), header=['#frame, chisquare_dist'])

        logger.info(' * Found {}/{} unique shots'.format(len(keyframes), max(map(lambda keyframe: keyframe.scene_id, keyframes))))


        # write results
        self.writeBoundariesCSV(path=self.args.input + '.boundaries.csv', keyframes=keyframes)
        pass
        self.writeScenesCSV(path=self.args.input + '.scenes.csv', keyframes=keyframes)


    


    def writeBoundariesCSV(self, path, keyframes):
        """
            Write shot boundaries into CSV file
        """

        if path is None:
            return

        header = ['# frame start, frame end, frame cut type']
        csvdata = map(lambda keyframe:[keyframe.name, keyframe.scene_end, keyframe.frame_cut_type], keyframes)
        util_csv.write_csv( path=path, rows=csvdata, header=header)



    def writeScenesCSV(self, path, keyframes):
        """
            Write final scene analysis into CSV.
        """
        if path is None:
            return

        header = ['# keyframe, scene id']
        csvdata = map(lambda keyframe:[keyframe.name, unichr(ord('A') + keyframe.scene_id)], keyframes)
        util_csv.write_csv( path=path, rows=csvdata, header=header)



def main():

    # initialize the app (contains default values for some command line arguments)
    app = DemoApp()

    # setup command line argument parser
    parser = argparse.ArgumentParser(description="Demo application")
    parser.add_argument("input",                metavar="VIDEO",                                                                     help="Path to input sequence folder")
    #parser.add_argument("-o", "--output",       dest="output",          default=app.args.output,                                     help="Path to output video")
    parser.add_argument("--format",             dest="input_format",    default=app.args.input_format,                               help="Format of input sequence")
    parser.add_argument("--chd",                dest="use_chd",         default=app.args.use_chd,               action="store_true", help="Use color histogram difference (CHD) boundary detection")
    parser.add_argument("--chdT",               dest="chd_thresh",      default=app.args.chd_thresh,                                 help="Threshold of CHD boundary detection")
    parser.add_argument("--edgeRatios",         dest="use_edge_ratios", default=app.args.use_edge_ratios,       action="store_true", help="Use edge ratios for boundary detection")
    parser.add_argument("--cannyT",             dest="canny_thresh",    default=app.args.canny_thresh,                               help="Lower Threshold of Canny edge detector")
    parser.add_argument("--cannySigma",         dest="canny_sigma",     default=app.args.canny_sigma,                                help="Sigma of Canny edge detector")
    parser.add_argument("--matchDist",          dest="match_dist",      default=app.args.match_dist,                                 help="Maximum distance of scene matcher for a keyframe to be considered matching")
    
    
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
