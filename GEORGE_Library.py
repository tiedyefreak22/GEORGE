'''
=========================================================================
Gradient-Effected Object Recognition Gauge for hive Entrances (GEORGE)
Neural-net-powered honeybee hive-mounted pollen, varroa, and wasp counter
#========================================================================
'''

import PIL
import random
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import ndimage as ndi
import os
import pathlib
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import io
import imageio
import glob
import cv2
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
from tqdm import tqdm
import math
from math import pi, ceil, floor
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import pandas as pd
import sys
import skimage
import skimage.io
import skimage.transform
from skimage import io, transform
from statistics import mean
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
from collections import defaultdict
import json
from datetime import datetime

IMAGE_PATH = 'Yang Model Training/bee_imgs/bee_imgs/'
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 40
IMAGE_CHANNELS = 3
category_index = {1: {'id': 1, 'name': 'regular'}, 2: {'id': 2, 'name': 'pollen'}, 3: {'id': 3, 'name': 'varroa'}, 4: {'id': 4, 'name': 'wasps'}}
batch_size = 64

def plot_detections(image_np, # credit: deeplearning.ai (https://github.com/https-deeplearning-ai)
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
        image_np: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
        figsize: size for the figure.
        image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)

def get_label(label):
    trans_label = []
    for j in range(len(label) + 1):
        trans_label.append(label[j-1]*j)
    trans_label = np.array(int(np.sum(trans_label)))
    return list(category_index.values())[trans_label], tf.one_hot(trans_label, 4), [trans_label + 1]

def zoom_image (new_image):
    DataGen = tf.keras.Sequential([layers.RandomZoom(0.1, 0.1)])
    new_image = tf.cast(np.array(new_image), tf.float32)
    new_image = DataGen(new_image)
    return np.array(new_image).astype('uint8')

def extract_sub(test_image, params=[110, 254, 1.001, -0.099]):
    t_upper, t_lower, alpha, beta = params
    h, w, d = tf.shape(test_image)
    res = (w, h);
    
    new_image = cv2.convertScaleAbs(test_image, alpha=alpha, beta=beta)
    
    # sharpen
    sharp_image = cv2.GaussianBlur(new_image, [0, 0], 3)
    new_image = cv2.addWeighted(new_image, 2, sharp_image, -1, 0)

    aperture_size = 3 # Aperture size (default = 3)
    L2Gradient = False # Boolean (default = False)
    edges = cv2.Canny(np.array(new_image), params[0], params[1], apertureSize = aperture_size, L2gradient = L2Gradient)
    ymin = h
    xmin = w
    ymax = 0
    xmax = 0
    coords = [ymin, xmin, ymax, xmax]
    
    # Find left bound/xmin
    left_bound = np.zeros([h, w])
    row_track = 0
    while row_track <= h - 1:
        column_track = 0
        while column_track <= w - 1:
            value = edges[row_track, column_track]
            if (value > 0):
                left_bound[row_track][column_track:(w - 1)] = 1
                if (column_track < coords[1]):
                    coords[1] = column_track
                column_track = w
            column_track = column_track + 1
        row_track = row_track + 1
    
    # Find right bound/xmax
    right_bound = np.zeros([h, w])
    row_track = 0
    while row_track <= h - 1:
        column_track = w - 1
        while column_track >= 0:
            value = edges[row_track, column_track]
            if (value > 0):
                right_bound[row_track][0:column_track] = 1
                if (column_track > coords[3]):
                    coords[3] = column_track
                column_track = 0
            column_track = column_track - 1
        row_track = row_track + 1

    # Find top bound/ymin
    top_bound = np.zeros([w, h])
    column_track = 0
    while column_track <= w - 1:
        row_track = 0
        while row_track <= h - 1:
            value = edges[row_track, column_track]
            if (value > 0):
                top_bound[column_track][row_track:(h - 1)] = 1
                if (row_track < coords[0]):
                    coords[0] = row_track
                row_track = h
            row_track = row_track + 1
        column_track = column_track + 1
    top_bound = np.transpose(top_bound)
    
    # Find bottom bound/ymax
    bottom_bound = np.zeros([w, h])
    column_track = 0
    while column_track <= w - 1:
        row_track = h - 1
        while row_track >= 0:
            value = edges[row_track, column_track]
            if (value > 0):
                bottom_bound[column_track][0:row_track] = 1
                if (row_track > coords[2]):
                    coords[2] = row_track
                row_track = 0
            row_track = row_track - 1
        column_track = column_track + 1
    bottom_bound = np.transpose(bottom_bound)
    
    mask = (top_bound * bottom_bound * right_bound * left_bound).astype('uint8')

    # smooth out the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.medianBlur(mask, 5)
    mask = np.array([[math.ceil(abs(j) / 255) for j in i] for i in mask]).astype('uint8')
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    new_image = test_image * mask
    coords = np.array(coords)
    new_image = Image.fromarray(new_image.astype(np.uint8))
    new_image = new_image.crop((coords[1], coords[0], coords[3], coords[2]))
    
    return np.array(new_image).astype('uint8')

def process_image(new_image, BG_img):
    new_image = Image.fromarray(new_image)
    rotation_range=random.uniform(0, 360)
    new_image = new_image.rotate(rotation_range, Image.BILINEAR, expand = 1)
    h = np.shape(new_image)[0]
    w = np.shape(new_image)[1]
    if h > 640 or w > 640:
        print(h, w)
    coords = [0, 0, h, w]
    
    x_offset = tf.random.uniform((), 0 , tf.cast(640-w, tf.int32), dtype=tf.int32)
    y_offset = tf.random.uniform((), 0 , tf.cast(640-h, tf.int32), dtype=tf.int32)
    new_image = tf.image.resize(np.array(new_image), (h, w))
    if np.shape(new_image)[2] == 3:
        new_image = tf.reshape(new_image, (list(np.shape(new_image))[0],list(np.shape(new_image))[1],3))
    elif np.shape(new_image)[2] == 4:
        new_image = tf.reshape(new_image, (list(np.shape(new_image))[0],list(np.shape(new_image))[1],4))
    new_image = tf.keras.utils.array_to_img(new_image).convert('RGBA')
    BG_img.paste(new_image, (x_offset, y_offset), new_image)

    coord_adder = [y_offset, x_offset, y_offset, x_offset]
    coords = np.array([sum(i) for i in zip(coords, coord_adder)])
    coords = np.array([(i / 640) for i in coords])

    return BG_img, coords

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image):
    alow = image.min()
    ahigh = image.max()
    amax = 255
    amin = 0

    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (np.array(auto_result).astype('uint8'))#, alpha, beta)

def get_file_and_info(filename):
    temp_label = np.zeros(5)
    temp_label[int(filename.split('\\')[1][0]) - 1] = 1
    label = get_label(temp_label[1:len(temp_label)])
    temp_gt_box = (filename.split('\\')[1]).split('_')
    gt_box = np.array([[float('0.' + temp_gt_box[0][1:len(temp_gt_box[0])]),
                        float('0.' + temp_gt_box[1]),
                        float('0.' + temp_gt_box[2]),
                        float('0.' + temp_gt_box[3].split('.')[0])]]).astype('float32')
    image = np.array(Image.open(filename).convert('RGB')).astype('uint8')
    return image, label, gt_box

def prep_train_imgs(train_images_np, train_labels, gt_boxes):
    # By convention, our non-background classes start counting at 1.
    num_classes = 4
    #num_classes = 5

    # Convert class labels to one-hot; convert everything to tensors.
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    item_num = np.shape(train_images_np)[0]

    for (train_image_np, gt_box_np, train_label) in zip(train_images_np, gt_boxes, train_labels):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        gt_classes_one_hot_tensors.append([train_label[1]])
    gt_classes_one_hot_tensors = np.array(gt_classes_one_hot_tensors)
    return train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune): # credit: deeplearning.ai (https://github.com/https-deeplearning-ai)
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
        """A single training iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                #[detection_model.preprocess(image_tensor)[0]
                [model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            global total_loss
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return total_loss

    return train_step_fn

class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license
    def get_imgIds(self):
        return list(self.im_dict.keys())
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]
    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]

'''
{
"info": info, "images": [image], "annotations": [annotation], "licenses": [license],
}
 
info{
"year": int, "version": str, "description": str, "contributor": str, "url": str, "date_created": datetime,
}
 
image{
"id": int, "width": int, "height": int, "file_name": str, "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime,
}
 
license{
"id": int, "name": str, "url": str,
}
 
annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}
 
categories[{
"id": int, "name": str, "supercategory": str,
}]
'''

class img:
    def __init__(self, img_id: int, width: int, height: int, file_name: str, license: int, flickr_url: str, coco_url: str, date_captured: str):
        self.id = img_id
        self.width = width
        self.height = height
        self.file_name = file_name
        self.license = license
        self.flickr_url = flickr_url
        self.coco_url = coco_url
        self.date_captured = date_captured
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False,
            indent=4)
 
class license:
    def __init__(self, lic_id: int, name: str, url: str):
        self.id = lic_id
        self.name = name
        self.url = url
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False,
            indent=4)
 
class bbox:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
 
class annotation:
    def __init__(self, ann_id: int, image_id: int, category_id: int, segmentation, area: float, bbox: bbox, iscrowd: bool):
        self.id = ann_id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = list(vars(bbox).values())
        self.iscrowd = iscrowd
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False,
            indent=4)
 
class category:
    def __init__(self, cat_id: int, name: str, supercategory: str):
        self.id = cat_id
        self.name = name
        self.supercategory = supercategory
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=False,
            indent=4)
 
info = {
        "year": int,
        "version": str,
        "description": str,
        "contributor": str,
        "url": str,
        "date_created": datetime,
        }

def image_stats(image):
    # compute the mean and standard deviation of each channel
    (first, second, third) = cv2.split(image)
    (firstMean, firstStd) = (first.mean(), first.std())
    (secondMean, secondStd) = (second.mean(), second.std())
    (thirdMean, thirdStd) = (third.mean(), third.std())
    # return the color statistics
    return (firstMean, firstStd, secondMean, secondStd, thirdMean, thirdStd)

def brightness_match(src_img, tgt_img):
    #convert both images to HSV
    alpha = []
    if tgt_img.shape[-1] == 4:
        _, _, _, alpha = cv2.split(tgt_img)

    source = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2HSV).astype("float32")
    target = cv2.cvtColor(np.array(tgt_img), cv2.COLOR_RGB2HSV).astype("float32")
    
    #Compute the mean and standard deviation of only the V channels for both the source and target images.
    (hMeanSrc, hStdSrc, sMeanSrc, sStdSrc, vMeanSrc, vStdSrc) = image_stats(source)
    (hMeanTar, hStdTar, sMeanTar, sStdTar, vMeanTar, vStdTar) = image_stats(target)
    
    #Subtract the mean of the target V channel from itself.
    (h, s, v) = cv2.split(target)
    v -= vMeanTar
    
    #Scale the target V by the ratio of the standard deviation of the target V divided by the standard deviation of the source V, multiplied by the target V.
    v = (vStdTar / vStdSrc) * v * .5
    
    #Add the mean of the source V channel to the target V channel.
    v += vMeanSrc
    
    #Clip any values that fall outside the range [0, 255].
    v = np.clip(v, 0, 255)
    
    #Merge the target channels back together.
    transfer = cv2.merge([h, s, v])
    
    #Convert target back to the RGB color space and return image
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_HSV2RGB)
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_RGB2RGBA)
    transfer[:, :, 3] = alpha
    return transfer

def james_BG_remover(sel, target, thresh_val):
    sel = np.array(sel)
    #target = sel[0,0]
    target = np.array(target)
    m = np.shape(sel)[0]
    n = np.shape(sel)[1]
    
    def make_palette(color,shape):
        out = np.zeros(shape)
        m = shape[0]
        n = shape[1]
        for i in range(m):
            for j in range(n):
                out[i,j,:] = color
        return out.astype('uint8')
    
    def check_pixel(current, target, bound):
        x = current.astype('int8')
        t = target.astype('int8')
        
        if np.max(np.abs(x-t)) < bound:
            return True
        else:
            return False
    
    for i in range(m):
        for j in range(n):
            if check_pixel(sel[i,j,:],target,thresh_val):
                sel[i,j,3] = 0
            else:
                pass
    return sel

# Convert Coco bb to Pascal_Voc bb
def coco_to_pascal_voc(x1, y1, w, h):
    return [x1,y1, x1 + w, y1 + h]
    