import _init_paths
import argparse
import glob
import json
import os
import shutil
import sys
from typing import List

import numpy as np
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat

from pascalvoc import getBoundingBoxes
from pascalvoc import ValidateCoordinatesTypes
from pascalvoc import ValidateImageSize


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.
    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero
    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def match_and_calculate_iou(gtBoundingBoxes, detBoundingBoxes, matchingMode, isPerFrame, minIou=0.0):
    """
    Args:
        `mode`: one of "gt", "det", or "common". gt/det means calculating IoU for each gt/det
        box. common means calculating the IoU for pairs of matched gt/det boxes.
    """
    assert matchingMode in ['gt', 'det', 'gt-common', 'det-common']
    if 'det' in matchingMode:
        gtBoundingBoxes, detBoundingBoxes = detBoundingBoxes, gtBoundingBoxes

    imageName2IouArr = dict()

    imageName2DetBboxes = dict()
    for det in detBoundingBoxes.getBoundingBoxes():
        imageName = det.getImageName()
        if imageName not in imageName2DetBboxes:
            imageName2DetBboxes[imageName] = [det]
        else:
            imageName2DetBboxes[imageName].append(det)

    # For each gt bbox, look for a matching det bbox
    for gt in gtBoundingBoxes.getBoundingBoxes():
        # Keep only detections in the same frame
        imageName = gt.getImageName()
        # detInThisFrameArr = [b for b in detBoundingBoxes.getBoundingBoxes() if b.getImageName() == imageName]
        detInThisFrameArr = imageName2DetBboxes[imageName] if imageName in imageName2DetBboxes else []
        # Iterate over detections, find the det with highest IoU
        gtBbox = gt.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
        max_iou = 0
        hasMatch = False

        for det in detInThisFrameArr:
            # Boxes of different classes are discarded
            if det.getClassId() != gt.getClassId():
                continue

            detBbox = det.getAbsoluteBoundingBox(format=BBFormat.XYX2Y2)
            iou = get_iou(gtBbox, detBbox)
            max_iou = max(iou, max_iou)

            if iou > 1e-5:
                hasMatch = True

        if hasMatch or 'common' not in matchingMode:
            if max_iou >= minIou:
                if imageName in imageName2IouArr:
                    imageName2IouArr[imageName].append(max_iou)
                else:
                    imageName2IouArr[imageName] = [max_iou]

    if isPerFrame:
        return imageName2IouArr
    else:
        iouArr = []
        for imageName, iouArrOfThisImage in imageName2IouArr.items():
            iouArr.extend(iouArrOfThisImage)
        return iouArr


def filterBoundingBoxes(allBoundingBoxes, filterFn):
    tmp = BoundingBoxes()
    for bbox in allBoundingBoxes.getBoundingBoxes():
        if filterFn(bbox):
            tmp.addBoundingBox(bbox)
    return tmp


if __name__ == '__main__':
    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
        'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
        'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    # formatter_class=RawTextHelpFormatter)
    # Positional arguments
    # Mandatory
    parser.add_argument(
        '-gt',
        '--gtfolder',
        dest='gtFolder',
        default=os.path.join(currentPath, 'groundtruths'),
        metavar='',
        help='folder containing your ground truth bounding boxes')
    parser.add_argument(
        '-det',
        '--detfolder',
        dest='detFolder',
        default=os.path.join(currentPath, 'detections'),
        metavar='',
        help='folder containing your detected bounding boxes')
    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the ground truth bounding boxes: '
        '(\'xywh\': <left> <top> <width> <height>)'
        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the detected bounding boxes '
        '(\'xywh\': <left> <top> <width> <height>) '
        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument(
        '-imgsize',
        dest='imgSize',
        metavar='',
        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument(
        '-sp', '--savepath', dest='savePath', metavar='', help='folder where the plots are saved')
    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')
    parser.add_argument(
        '--start-cutoff',
        dest='startCutoff',
        type=int,
        help='number of frames to remove from the start')
    parser.add_argument(
        '--max-frame-id',
        dest='maxFrameId',
        type=int,
        help='frames after this will not be considered')
    parser.add_argument(
        '--filter-for-common-images',
        dest='filterForCommonImages',
        action='store_true',
        help='keep bboxes from common images only')
    parser.add_argument(
        '--min-size',
        dest='minSize',
        type=int,
        help='min width or height of tracking boxes to use')

    args = parser.parse_args()

    gtFolder = args.gtFolder
    detFolder = args.detFolder

    gtFormat = BBFormat.XYWH if args.gtFormat == 'xywh' else BBFormat.XYX2Y2
    detFormat = BBFormat.XYWH if args.detFormat == 'xywh' else BBFormat.XYX2Y2

    # Coordinates types
    errors = []
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)

    savePath = args.savePath

    # WARNING: Do not filter gt by frame count, if it is not continuous
    gtBoundingBoxes, gtClasses = getBoundingBoxes(
        gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize,
        num_cutoff_frames_start=args.startCutoff)
    detBoundingBoxes, detClasses = getBoundingBoxes(
        detFolder, False, detFormat, detCoordType, imgSize=imgSize,
        num_cutoff_frames_start=args.startCutoff)
    if args.filterForCommonImages:
        gtImgArr = [os.path.splitext(os.path.basename(f))[0]
                    for f in glob.glob(os.path.join(args.gtFolder, '*.txt'))]
        detImgArr = [os.path.splitext(os.path.basename(f))[0]
                     for f in glob.glob(os.path.join(args.detFolder, '*.txt'))]
        commonImgSet = set(gtImgArr) & set(detImgArr)

        tmp = BoundingBoxes()
        for bbox in gtBoundingBoxes.getBoundingBoxes():
            if bbox.getImageName() in commonImgSet:
                tmp.addBoundingBox(bbox)
        gtBoundingBoxes = tmp
        tmp = BoundingBoxes()
        for bbox in detBoundingBoxes.getBoundingBoxes():
            if bbox.getImageName() in commonImgSet:
                tmp.addBoundingBox(bbox)
        detBoundingBoxes = tmp

    # Filter
    if args.maxFrameId is not None:
        filterFnByFrameId = lambda bbox: (int(bbox.getImageName()) < args.maxFrameId)
        gtBoundingBoxes = filterBoundingBoxes(gtBoundingBoxes, filterFnByFrameId)
        detBoundingBoxes = filterBoundingBoxes(detBoundingBoxes, filterFnByFrameId)
    if args.minSize is not None:
        def filterFnByMinSize(bbox):
            _, _, w, h = bbox.getAbsoluteBoundingBox()
            return w >= args.minSize and h >= args.minSize
        gtBoundingBoxes = filterBoundingBoxes(gtBoundingBoxes, filterFnByMinSize)
        detBoundingBoxes = filterBoundingBoxes(detBoundingBoxes, filterFnByMinSize)

    imageName2IouArr = match_and_calculate_iou(gtBoundingBoxes, detBoundingBoxes,
                                               matchingMode='gt', isPerFrame=True)
    print(json.dumps(imageName2IouArr))
