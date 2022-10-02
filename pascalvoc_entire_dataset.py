import argparse
import copy
import glob
import os
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from pascalvoc import ValidateCoordinatesTypes
from pascalvoc import ValidateFormats
from pascalvoc import ValidateImageSize
from pascalvoc import ValidateMandatoryArgs

from utils import BBFormat


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxesEntireDataset(root_directory,
                                  isGT,
                                  bbFormat,
                                  coordType,
                                  includeFilter,
                                  vidnameIdxInPath,
                                  tagIdxInPath,
                                  vid_name_2_tags_in,
                                  allBoundingBoxes=None,
                                  allClasses=None,
                                  imgSize=(0, 0),
                                  classesToConsider=None,
                                  num_cutoff_frames_start=0,
                                  num_cutoff_frames_stop=0):
    """
    Based on getBoundingBoxes(), but read bbox of all videos of a dataset.
    """
    if isGT:
        assert tagIdxInPath is None
    else:
        assert vid_name_2_tags_in is None

    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []

    # Collect unique image names
    imgNameArr = []

    dir_arr = sorted(glob.glob(os.path.join(root_directory, includeFilter)))

    # For each video, collect its list of tags in the glob
    if (tagIdxInPath is not None) and (not isGT):
        vid_name_2_tags_out = dict()
        for d in dir_arr:
            vid_name = d.split(os.sep)[vidnameIdxInPath]
            tag = d.split(os.sep)[tagIdxInPath]
            if vid_name not in vid_name_2_tags_out:
                vid_name_2_tags_out[vid_name] = [tag]
            else:
                vid_name_2_tags_out[vid_name].append(tag)
    else:
        vid_name_2_tags_out = None

    for directory in dir_arr:
        vidName = ''
        if (tagIdxInPath is not None) and (not isGT):
            vidName += directory.split(os.sep)[tagIdxInPath] + '--'
        # elif (tagIdxInPath is not None) and isGT:
        #     # Replicate
        #     pass
        vidName += directory.split(os.sep)[vidnameIdxInPath]
        # Read ground truths
        os.chdir(directory)
        files = glob.glob("*.txt")
        files.sort()

        # Cutoff frames
        files = files[num_cutoff_frames_start:]
        if num_cutoff_frames_stop > 0:
            files = files[:(-1 * num_cutoff_frames_stop)]

        # Read GT detections from txt file
        # Each line of the files in the groundtruths folder represents a ground truth bounding box
        # (bounding boxes that a detector should detect)
        # Each value of each line is  "class_id, x, y, width, height" respectively
        # Class_id represents the class of the bounding box
        # x, y represents the most top-left coordinates of the bounding box
        # x2, y2 represents the most bottom-right coordinates of the bounding box
        for f in files:
            nameOfImage = vidName + '--' + f.replace(".txt", "")
            fh1 = open(f, "r")
            for line in fh1:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")
                if isGT:
                    # idClass = int(splitLine[0]) #class
                    idClass = (splitLine[0])  # class
                    if classesToConsider is not None:
                        if idClass not in classesToConsider:
                            continue
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                    # If relative, convert x, y to center of bbox
                    if coordType == CoordinatesType.Relative:
                        x += w / 2
                        y += h / 2
                    bb = BoundingBox(nameOfImage,
                                     idClass,
                                     x,
                                     y,
                                     w,
                                     h,
                                     coordType,
                                     imgSize,
                                     BBType.GroundTruth,
                                     format=bbFormat)
                    # Repeat the same bbox multiple times
                    if vid_name_2_tags_in is None:
                        allBoundingBoxes.addBoundingBox(bb)
                        imgNameArr.append(nameOfImage)
                    else:
                        for tag in vid_name_2_tags_in[vidName]:
                            bb_cp = copy.copy(bb)
                            bb_cp._imageName = tag + '--' + nameOfImage
                            allBoundingBoxes.addBoundingBox(bb_cp)
                            imgNameArr.append(bb_cp.getImageName())
                else:
                    imgNameArr.append(nameOfImage)
                    # idClass = int(splitLine[0]) #class
                    idClass = (splitLine[0])  # class
                    if classesToConsider is not None:
                        if idClass not in classesToConsider:
                            continue
                    confidence = float(splitLine[1])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
                    # If relative, convert x, y to center of bbox
                    if coordType == CoordinatesType.Relative:
                        x += w / 2
                        y += h / 2
                    bb = BoundingBox(nameOfImage,
                                     idClass,
                                     x,
                                     y,
                                     w,
                                     h,
                                     coordType,
                                     imgSize,
                                     BBType.Detected,
                                     confidence,
                                     format=bbFormat)
                    allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
            fh1.close()
    return allBoundingBoxes, allClasses, imgNameArr, vid_name_2_tags_out


def main():
    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description=f'{message}\nThis project applies the most popular metrics used to evaluate object detection '
        'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
        'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument('-gt',
                        '--gtfolder',
                        dest='gtFolder',
                        default=os.path.join(currentPath, 'groundtruths'),
                        metavar='',
                        help='folder containing your ground truth bounding boxes')
    parser.add_argument('-det',
                        '--detfolder',
                        dest='detFolder',
                        default=os.path.join(currentPath, 'detections'),
                        metavar='',
                        help='folder containing your detected bounding boxes')
    # Optional
    parser.add_argument('-t',
                        '--threshold',
                        dest='iouThreshold',
                        type=float,
                        default=0.5,
                        metavar='',
                        help='IOU threshold. Default 0.5')
    parser.add_argument('-gtformat',
                        dest='gtFormat',
                        metavar='',
                        default='xywh',
                        help='format of the coordinates of the ground truth bounding boxes: '
                        '(\'xywh\': <left> <top> <width> <height>)'
                        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument('-detformat',
                        dest='detFormat',
                        metavar='',
                        default='xywh',
                        help='format of the coordinates of the detected bounding boxes '
                        '(\'xywh\': <left> <top> <width> <height>) '
                        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument('-gtcoords',
                        dest='gtCoordinates',
                        default='abs',
                        metavar='',
                        help='reference of the ground truth bounding box coordinates: absolute '
                        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument('-detcoords',
                        default='abs',
                        dest='detCoordinates',
                        metavar='',
                        help='reference of the ground truth bounding box coordinates: '
                        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument('-imgsize',
                        dest='imgSize',
                        metavar='',
                        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument('-sp',
                        '--savepath',
                        dest='savePath',
                        metavar='',
                        help='folder where the plots are saved')
    parser.add_argument('-np',
                        '--noplot',
                        dest='showPlot',
                        action='store_false',
                        help='no plot is shown during execution')
    parser.add_argument('--start-cutoff',
                        dest='startCutoff',
                        type=int,
                        help='number of frames to remove from the start')
    parser.add_argument('--max-frame-id',
                        dest='maxFrameId',
                        type=int,
                        help='frames after this will not be considered')
    parser.add_argument('--filter-for-common-images',
                        dest='filterForCommonImages',
                        action='store_true',
                        help='keep bboxes from common images only')
    parser.add_argument('--min-size',
                        dest='minSize',
                        type=int,
                        help='min width or height of tracking boxes to use')
    parser.add_argument('--include-filter-gt',
                        dest='includeFilterGt',
                        help='pattern of video names to include in this dataset for gt folder',
                        default='*')
    parser.add_argument('--vidname-idx-in-path-gt',
                        dest='vidnameIdxInPathGt',
                        type=int,
                        help='index of path component corresponding to video name')
    parser.add_argument('--include-filter-det',
                        dest='includeFilterDet',
                        help='pattern of video names to include in this dataset for det folder',
                        default='*')
    parser.add_argument('--vidname-idx-in-path-det',
                        dest='vidnameIdxInPathDet',
                        type=int,
                        help='index of path component corresponding to video name')
    parser.add_argument('--tag-idx-in-path-det',
                        dest='tagIdxInPathDet',
                        type=int,
                        default=None,
                        help="""index of path component corresponding to tag, used to distinguish
                                multiple instances of the same video. This is necessary because
                                mAP matching is without replacement. Ground truth bboxes will be
                                replicated number of unique tag times.
                        """)
    parser.add_argument('--classes-to-consider',
                        dest='classesToConsider',
                        nargs='+',
                        type=str,
                        help='labels of classes to include in mAP calculation')
    args = parser.parse_args()

    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []
    # Validate formats
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
    # Groundtruth folder
    if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
        gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
    else:
        # errors.pop()
        gtFolder = os.path.join(currentPath, 'groundtruths')
        if os.path.isdir(gtFolder) is False:
            errors.append('folder %s not found' % gtFolder)
    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    imgSize = (0, 0)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
    # Detection folder
    if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
        detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
    else:
        # errors.pop()
        detFolder = os.path.join(currentPath, 'detections')
        if os.path.isdir(detFolder) is False:
            errors.append('folder %s not found' % detFolder)
    if args.savePath is not None:
        try:
            os.makedirs(args.savePath)
        except FileExistsError:
            pass
        savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
    else:
        savePath = os.path.join(currentPath, 'results')
    # Validate savePath
    # If error, show error messages
    if len(errors) != 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    showPlot = args.showPlot

    # Get detected boxes
    detBoundingBoxes, detClasses, detImgNameArr, vid_name_2_tags = getBoundingBoxesEntireDataset(
        detFolder,
        False,
        detFormat,
        detCoordType,
        args.includeFilterDet,
        args.vidnameIdxInPathDet,
        args.tagIdxInPathDet,
        None,
        imgSize=imgSize,
        classesToConsider=args.classesToConsider)
    print('Num det bbox: ', len(detBoundingBoxes._boundingBoxes))
    # Get groundtruth boxes
    gtBoundingBoxes, gtClasses, gtImgNameArr, _ = getBoundingBoxesEntireDataset(
        gtFolder,
        True,
        gtFormat,
        gtCoordType,
        args.includeFilterGt,
        args.vidnameIdxInPathGt,
        None,
        vid_name_2_tags,
        imgSize=imgSize,
        classesToConsider=args.classesToConsider)
    print('Num gt bbox: ', len(gtBoundingBoxes._boundingBoxes))

    # Merge det and gt bounding boxes
    allBoundingBoxes = BoundingBoxes()
    for bbox in gtBoundingBoxes.getBoundingBoxes():
        allBoundingBoxes.addBoundingBox(bbox)
    for bbox in detBoundingBoxes.getBoundingBoxes():
        allBoundingBoxes.addBoundingBox(bbox)
    allClasses = list(set(gtClasses + detClasses))

    # Filter
    if args.startCutoff is not None:
        tmp = BoundingBoxes()
        for bbox in allBoundingBoxes.getBoundingBoxes():
            # Assuming last 6 chars are frame id
            if int(bbox.getImageName()[-6:]) >= args.startCutoff:
                tmp.addBoundingBox(bbox)
        allBoundingBoxes = tmp
    if args.maxFrameId is not None:
        tmp = BoundingBoxes()
        for bbox in allBoundingBoxes.getBoundingBoxes():
            if int(bbox.getImageName()) < args.maxFrameId:
                tmp.addBoundingBox(bbox)
        allBoundingBoxes = tmp
    if args.filterForCommonImages:
        commonImgSet = set(gtImgNameArr) & set(detImgNameArr)
        tmp = BoundingBoxes()
        for bbox in allBoundingBoxes.getBoundingBoxes():
            if bbox.getImageName() in commonImgSet:
                tmp.addBoundingBox(bbox)
        allBoundingBoxes = tmp
    if args.minSize is not None:
        tmp = BoundingBoxes()
        for bbox in allBoundingBoxes.getBoundingBoxes():
            _, _, w, h = bbox.getAbsoluteBoundingBox()
            if w >= args.minSize and h >= args.minSize:
                tmp.addBoundingBox(bbox)
        allBoundingBoxes = tmp

    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    if validClasses > 0:
        mAP = acc_AP / validClasses
    else:
        mAP = 0.0
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)
    f.close()


if __name__ == '__main__':
    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))

    VERSION = '0.2 (beta)'

    with open('message.txt', 'r') as f:
        message = f'\n\n{f.read()}\n\n'

    print(message)

    main()
