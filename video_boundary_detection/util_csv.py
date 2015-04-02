#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This is a utility class with useful utility functions related to CSV files
"""


import csv
import cv2
import logging
import string

__author__ = "Eduard Feicho"
__copyright__ = "Copyright 2015"



logger = logging.getLogger(__name__)


def write_csv(path, rows, header = None):

    if path is None:
        return

    logger.info(' * Write {}'.format(path))

    with open(path, 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        # write header
        if header is not None:
            csvwriter.writerow( map(str, header) )
        # write data
        for row in rows:
            row = map(str, row)
            csvwriter.writerow( row )
            logger.debug( string.join(row, ',') )
