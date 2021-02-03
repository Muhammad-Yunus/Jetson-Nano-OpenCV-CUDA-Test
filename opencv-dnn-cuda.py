import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import Utils
util = Utils()

parser = argparse.ArgumentParser()
parser.add_argument('--target', help='DNN Target')
parser.add_argument('--backend', help='DNN Backend')
args = parser.parse_args()

print("OpenCV version : %s \n\n" % cv2.__version__)

print('[INFO] load coco names :')
classesFile = "yolo/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')    
print("number of classes : %s \n\n" % len(classes))

print("[INFO] load tiny yolo model :")
modelConfiguration = "yolo/coco_yolov3-tiny.cfg"
modelWeights = "yolo/coco_yolov3-tiny.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.getLayerNames()
layerOutput = net.getUnconnectedOutLayersNames()
print("unconnected layer :", layerOutput, "\n\n")

if args.backend == 'CUDA' :
    print("[INFO] Set CUDA as BACKEND and TARGET DNN...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
elif args.backend == 'OPENCV' :
    print("[INFO] Set OPENCV CPU as BACKEND and TARGET DNN...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
else :
    assert("DNN Backend not found!")

if args.target == 'CUDA' :
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
elif args.target == 'CPU' :
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
else :
    assert("DNN Backend not found!")
print("\n")

print("[INFO] inference for all image in `images/` :\n")
inpWidth = 416
inpHeight = 416
for filename in os.listdir("images/") :
    print("processing %s ..." % filename)
    frame = cv2.imread("images/%s" % filename)
    blob = cv2.dnn.blobFromImage(
                            frame, 
                            1/255, 
                            (inpWidth, inpHeight), 
                            [0, 0, 0],
                            1, 
                            crop=False)

    net.setInput(blob)
    outs = net.forward(layerOutput)

    t, _ = net.getPerfProfile()
    print('inference time: %.2f s' % (t / cv2.getTickFrequency()))

    print("postprocess %s ...\n" % filename)
    frame = util.postprocess(outs, frame, classes)

    cv2.imwrite("outputs/%s" % filename, frame)