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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical
import warnings

global IMAGE_PATH
global IMAGE_WIDTH
global IMAGE_HEIGHT
global IMAGE_CHANNELS
global category_index

IMAGE_PATH = 'Yang Model Training/bee_imgs/bee_imgs/'
IMAGE_WIDTH = 75
IMAGE_HEIGHT = 150
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

def process_image(new_image):
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
    new_image = tf.reshape(new_image, (list(np.shape(new_image))[0],list(np.shape(new_image))[1],3))
    new_image = tf.image.pad_to_bounding_box(new_image, y_offset, x_offset, 640, 640)
    new_image = tf.cast(new_image, tf.uint8)
    
    coord_adder = [y_offset, x_offset, y_offset, x_offset]
    coords = np.array([sum(i) for i in zip(coords, coord_adder)])
    coords = np.array([(i / 640) for i in coords])

    return new_image, coords

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

def do_image(original, filename, thresh_lo = 157, thresh_hi = 200, erode_val = 10, gauss_val = 9, exponent = 0.18, gauss_val2 = 3, R = 135, G = 226, B = 192):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    r_channel, g_channel, b_channel = cv2.split(original)
    color_channels = [r_channel, g_channel, b_channel]
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
    default_vals = [R, G, B]
        
    border_int = 3
    img = cv2.copyMakeBorder(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), border_int, border_int, border_int, border_int, cv2.BORDER_CONSTANT, value=255)
    
    ret, thresh = cv2.threshold(img, thresh_lo, thresh_hi, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = cv2.GaussianBlur(dist, (gauss_val, gauss_val), 0)
    dist = dist[border_int:img.shape[0]-border_int, border_int:img.shape[1]-border_int]
    norm_dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    alpha_channel = cv2.bitwise_not((255 * norm_dist).astype(b_channel.dtype))
    
    kernel = np.ones((erode_val, erode_val), np.uint8)
    alpha_channel = cv2.erode(alpha_channel, kernel)
    
    img_RGBA = Image.fromarray(cv2.cvtColor(cv2.merge((r_channel, g_channel, b_channel, alpha_channel)), cv2.COLOR_BGRA2RGBA))
    
    # flattens alpha-channel image
    background = Image.new('RGBA', (np.shape(img_RGBA)[1], np.shape(img_RGBA)[0]), (255,255,255))
    img_RGB = Image.fromarray(cv2.cvtColor(np.array(Image.alpha_composite(background, img_RGBA)), cv2.COLOR_RGBA2RGB))
    img_RGB.save("tmp/" + filename, 'JPEG', quality=100, subsampling=0)
    
    return Image.open("tmp/" + filename), alpha_channel