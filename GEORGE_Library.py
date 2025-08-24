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
# import tensorflow_datasets as tfds
# from tensorflow.keras import layers
# from object_detection.utils import label_map_util
# from object_detection.utils import config_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
import pandas as pd
import sys
import skimage
import skimage.io
import skimage.transform
from skimage import io, transform
from statistics import mean, variance
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
from keras_cv import bounding_box
import re
from tensorflow import keras
from copy import deepcopy

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
        #gt_classes_one_hot_tensors.append([train_label[1]])
        gt_classes_one_hot_tensors.append([train_label])
    #gt_classes_one_hot_tensors = np.array(gt_classes_one_hot_tensors)
    return train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors

def prep_train_imgs_only(train_images_np):
    # Convert class labels to one-hot; convert everything to tensors.
    train_image_tensors = []

    for train_image_np in train_images_np:
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))
    return train_image_tensors

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune): # credit: deeplearning.ai (https://github.com/https-deeplearning-ai)
    """Get a tf.function for training step."""

    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function(experimental_relax_shapes=True)
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
    def get_imgLicenses(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
    def get_wh(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        img_w = [self.im_dict[im_id]["width"] for im_id in im_ids]
        img_h = [self.im_dict[im_id]["height"] for im_id in im_ids]
        return img_w[0], img_h[0]

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
def coco_to_pascal_voc(x, y, w, h):
    return [x, y, x + w, y + h]

# Convert Coco bb to relative xyxy
def coco_to_rel_xyxy(x, y, w, h, img_w, img_h):
    return [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]

def rel_xyxy_to_coco(x1, y1, x2, y2, img_w, img_h):
    return [x1 * img_w, y1 * img_h, (x2 - x1) * img_w, (y2 - y1) * img_h]

# Convert Coco bb to relative yxyx
def coco_to_rel_yxyx(x, y, w, h, img_w, img_h):
    return [y / img_h, x / img_w, (y + h) / img_h, (x + w) / img_w]

def rel_yxyx_to_coco(x1, y1, x2, y2, img_w, img_h):
    return [y1 * img_h, x1 * img_w, (y2 - y1) * img_h, (x2 - x1) * img_w]

def load_image(filepath):
    image_data = tf.io.read_file(filepath)
    return tf.cast(tf.io.decode_png(image_data, channels=3), tf.float32)

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images.to_tensor(),
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=10,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )

def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

def kalman(z, Q):
    # initial parameters
    n_iter = len(z)
    sz = (n_iter,)            # size of array
    #Q = 0.005                # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)       # a posteri estimate of x
    P = np.zeros(sz)          # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)     # a priori error estimate
    K = np.zeros(sz)          # gain or blending factor

    R = variance(z)           # estimate of measurement variance

    # initial estimates
    xhat[0] = mean(z[0:3])
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

"""
## Implementing utility functions

Bounding boxes can be represented in multiple ways, the most common formats are:

- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`

Since we require both formats, we will be implementing functions for converting
between the formats.
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


"""
## Computing pairwise Intersection Over Union (IOU)

As we will see later in the example, we would be assigning ground truth boxes
to anchor boxes based on the extent of overlapping. This will require us to
calculate the Intersection Over Union (IOU) between all the anchor
boxes and ground truth boxes pairs.
"""


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax


"""
## Implementing Anchor generator

Anchor boxes are fixed sized boxes that the model uses to predict the bounding
box for an object. It does this by regressing the offset between the location
of the object's center and the center of an anchor box, and then uses the width
and height of the anchor box to predict a relative scale of the object. In the
case of RetinaNet, each location on a given feature map has nine anchor boxes
(at three scales and three ratios).
"""


class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2**x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2**i for i in range(3, 8)]
        self._areas = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2**i),
                tf.math.ceil(image_width / 2**i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)


"""
## Preprocessing data

Preprocessing the images involves two steps:

- Resizing the image: Images are resized such that the shortest size is equal
to 800 px, after resizing if the longest side of the image exceeds 1333 px,
the image is resized such that the longest size is now capped at 1333 px.
- Applying augmentation: Random scale jittering  and random horizontal flipping
are the only augmentations applied to the images.

Along with the images, bounding boxes are rescaled and flipped if required.
"""


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image_shape = np.shape(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


"""
## Encoding labels

The raw labels, consisting of bounding boxes and class ids need to be
transformed into targets for training. This transformation consists of
the following steps:

- Generating anchor boxes for the given image dimensions
- Assigning ground truth boxes to the anchor boxes
- The anchor boxes that are not assigned any objects, are either assigned the
background class or ignored depending on the IOU
- Generating the classification and regression targets using anchor boxes
"""


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


"""
## Building the ResNet50 backbone

RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""


def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


"""
## Building Feature Pyramid Network as a custom layer
"""


class FeaturePyramid(keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super().__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


"""
## Building the classification and box regression heads.
The RetinaNet model has separate heads for bounding box regression and
for predicting class probabilities for the objects. These heads are shared
between all the feature maps of the feature pyramid.
"""


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(keras.layers.ReLU())
    head.add(
        keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head


"""
## Building RetinaNet using a subclassed model
"""


class RetinaNet(keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


"""
## Implementing a custom layer to decode predictions
"""


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=4,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=20,
        max_detections=20,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


"""
## Implementing Smooth L1 loss and Focal Loss as keras custom losses
"""


class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super().__init__(reduction="none", name="RetinaNetBoxLoss")
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super().__init__(reduction="none", name="RetinaNetClassificationLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=4, alpha=0.25, gamma=2.0, delta=1.0):
        super().__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss

def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

def magic_wand(image, reference, tolerance, contiguous = True):
    '''
    A function similar to Photoshop's "magic wand" tool, used to select
    pixels with similar values
    
    ***Inputs***
    image:      RGB or RGBA image array
    reference:  either a 2 or 3 number tuple containing a pixel location 
                or a pixel value, respectively
    tolerance:  integer value defining the +/- range for acceptable pixel values
    contiguous: boolean value defining whether to only return contigous pixels

    ***Outputs***
    mask:       Returns a 1-dimensional numpy array uint8 mask of contiguous or non-contiguous
                pixels with R, G, B values that fall within the +/- tolerance range
    '''
    def bound(value):
        return max(0, min(255, value))
    
    def get_adjacent(ref_number):
        adjacent_nums = []
        adjacent_nums.append((ref_number[0] - 1, ref_number[1]))
        adjacent_nums.append((ref_number[0] + 1, ref_number[1]))
        adjacent_nums.append((ref_number[0] - 1, ref_number[1] - 1))
        adjacent_nums.append((ref_number[0], ref_number[1] - 1))
        adjacent_nums.append((ref_number[0] + 1, ref_number[1] - 1))
        adjacent_nums.append((ref_number[0] - 1, ref_number[1] + 1))
        adjacent_nums.append((ref_number[0], ref_number[1] + 1))
        adjacent_nums.append((ref_number[0] + 1, ref_number[1] + 1))
        return adjacent_nums
        
    def mark_adjacent(ref_number):
        adjacent_nums = get_adjacent(ref_number) # get references of adjacent pixels
        for adjacent_num in adjacent_nums:
            if (np.array(adjacent_num) >= 0).all() and (np.array(adjacent_num) < im.shape[:2]).all(): # in-bounds check
                current_ref_pixel = im[adjacent_num[0], adjacent_num[1], :3]
                if (current_ref_pixel > [bound(i - tolerance) for i in ref_pixel]).all() and (current_ref_pixel < [bound(i + tolerance) for i in ref_pixel]).all(): # in-tolerance and between 0 & 255 check
                    mask[adjacent_num] = 0

    ref_number = ()
    ref_pixel = ()
    im = np.array(deepcopy(image).astype('uint8'))
    ref_pixel_tol = 0
    
    while ref_number == ():
        if len(reference) > 2: # pixel value provided
            ref_pixel = reference
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    current_ref_pixel = im[i, j, :3]
                    if (current_ref_pixel > [bound(i - ref_pixel_tol) for i in ref_pixel]).all() and (current_ref_pixel < [bound(i + ref_pixel_tol) for i in ref_pixel]).all():
                        ref_number = (i, j) # get and store pixel location with reference pixel value
                        break
                    if (i == range(im.shape[0])) and (j == range(im.shape[1])):
                        raise Exception("No pixels match provided reference")
                else:
                    continue
                break
                        
        elif len(reference) == 2: # pixel location provided
            ref_number = reference
            ref_pixel = im[ref_number[0], ref_number[1], :3] # get and store reference pixel value
        else:
            raise Exception("Cannot parse provided reference")
        ref_pixel_tol = ref_pixel_tol + 1

    if contiguous:
        mask = np.ones_like(im[:, :, 0]) # make an array of zeros in the shape of one of the color channels of the original im
        mask[ref_number[0], ref_number[1]] = 0 # mark the starting pixel as 1
        # Top
        i = 0
        while ref_number[0] - i > 0:
            j = 0
            while ref_number[1] - j > 0:
                if mask[ref_number[0] - i, ref_number[1] - j] == 0:
                    mark_adjacent((ref_number[0] - i, ref_number[1] - j))
                j = j + 1
            j = 0
            while ref_number[1] + j < im.shape[1]:
                if mask[ref_number[0] - i, ref_number[1] + j] == 0:
                    mark_adjacent((ref_number[0] - i, ref_number[1] + j))
                j = j + 1
            i = i + 1
    
        # Bottom
        i = 0
        while ref_number[0] + i < im.shape[0]:
            j = 0
            while ref_number[1] - j > 0:
                if mask[ref_number[0] + i, ref_number[1] - j] == 0:
                    mark_adjacent((ref_number[0] + i, ref_number[1] - j))
                j = j + 1
            j = 0
            while ref_number[1] + j < im.shape[1]:
                if mask[ref_number[0] + i, ref_number[1] + j] == 0:
                    mark_adjacent((ref_number[0] + i, ref_number[1] + j))
                j = j + 1
            i = i + 1
                
        return mask
    else:
        mask = np.ones_like(im[:, :, 0]) # make an array of zeros in the shape of one of the color channels of the original im
        
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                current_ref_pixel = im[i, j, :3]
                if (current_ref_pixel > [bound(i - tolerance) for i in ref_pixel]).all() and (current_ref_pixel < [bound(i + tolerance) for i in ref_pixel]).all(): # in-tolerance and between 0 & 255 check
                    mask[i, j] = 0
        return mask

def get_row_view(a):
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[-1])))
    a = np.ascontiguousarray(a)
    return a.reshape(-1, a.shape[-1]).view(void_dt).ravel()
    
def get_mode(img):
    unq, idx, count = np.unique(get_row_view(img), return_index=1, return_counts=1)
    return img.reshape(-1,img.shape[-1])[idx[count.argmax()]]

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    bbox = font.getbbox(display_str)
    text_width, text_height = bbox[2], bbox[3]
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)