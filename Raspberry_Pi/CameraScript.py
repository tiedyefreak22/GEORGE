import sys
sys.path.append("/home/kevinhardin/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games")
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
from time import sleep, time
from datetime import datetime, date
from PIL import Image
import os
os.environ["LIBCAMERA_LOG_LEVELS"] = "2"
import matplotlib.pyplot as plt
import cv2
import argparse
from math import sqrt
import glob
from tqdm import tqdm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import warnings
warnings.filterwarnings('ignore')

tuning = Picamera2.load_tuning_file("/usr/share/libcamera/ipa/rpi/vc4/imx708_noir.json")
camera = Picamera2(tuning=tuning)
os.system("v4l2-ctl --set-ctrl wide_dynamic_range=1 -d /dev/v4l-subdev0")
capture_config = camera.create_still_configuration(main = {"size": (1920,1080)})
reference = cv2.imread('focus_target/focus_target.png') # reference image

def get_centroid(vertices):
    x = [vertex[0] for vertex in vertices]
    y = [vertex[1] for vertex in vertices]
    return (sum(x)/len(vertices), sum(y)/len(vertices))

def return_focus_target_window(reference, image):
    img1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # training image
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    thresh = 0.4
    for m,n in matches:
        if m.distance < thresh*n.distance:
            good.append([m])
            
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    points = [i[0] for i in np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)]
    
    while True:
        if len(points) <= 5:
            good = []
            thresh = thresh + 0.01
            for m,n in matches:
                if m.distance < thresh*n.distance:
                    good.append([m])
                    
            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            points = [i[0] for i in np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)]
        else:
            break
    #plt.figure(figsize = (20,20))
    #plt.imshow(img3),plt.show()
    hull = ConvexHull([(i[0], i[1]) for i in points])
    hull_vertices = []
    for i in range(len(hull.vertices)):
        hull_vertices.append((points[hull.vertices[i]][0], points[hull.vertices[i]][1]))

    centroid = get_centroid(points)

    return [int(centroid[0] - np.shape(img2)[1]/24), int(centroid[1] - np.shape(img2)[0]/24), int(np.shape(img2)[1]/12), int(np.shape(img2)[0]/12)]
    
def takePic():
    x = 527
    y = 876
    w = 128
    h = 122
    FocusVal = 3
    camera.stop()
    camera.configure(capture_config)
    camera.start()
    
    # Give time to settle before changing settings
    sleep(1)
    camera.controls.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0, "FrameRate": 1.0})
    # Wait for settings to take effect
    sleep(1)

    source_img = camera.capture_array()
    source_im = Image.fromarray(source_img)
    window = return_focus_target_window(np.array(reference), np.array(source_im))
    startmetadata = camera.capture_metadata()
    StartFocusVal = startmetadata['LensPosition']
    # Give time to settle before changing settings
    sleep(1)
    camera.controls.set_controls({"AfMode": controls.AfModeEnum.Continuous, "AfMetering": controls.AfMeteringEnum.Windows, "AfWindows": [ (window[0], window[1], window[2], window[3]) ], "FrameRate": 1.0})
    # Wait for settings to take effect
    sleep(1)
    
    image = camera.capture_array()
    im = Image.fromarray(image)
    finalmetadata = camera.capture_metadata()
    FinalFocusVal = finalmetadata['LensPosition']
    camera.stop()
    
    return im, StartFocusVal, FinalFocusVal
