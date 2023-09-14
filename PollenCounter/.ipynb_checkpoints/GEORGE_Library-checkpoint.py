import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import io
import imageio
import glob
import imutils
import cv2
import scipy
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
from tqdm import tqdm
import time
import math
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
import seaborn as sns
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
from matplotlib import cm

IMAGE_PATH = 'Yang Model Training/bee_imgs/bee_imgs/'
IMAGE_WIDTH = 75
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
category_index = {1: {'id': 1, 'name': 'regular'}, 2: {'id': 2, 'name': 'cooling'}, 3: {'id': 3, 'name': 'pollen'}, 4: {'id': 4, 'name': 'varroa'}, 5: {'id': 5, 'name': 'wasps'}}


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: a file path.

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
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
    for j in range(len(label)):
        trans_label.append(label[j-1]*j)
    trans_label = np.array(int(np.sum(trans_label)))
    return list(category_index.values())[trans_label], tf.one_hot(trans_label, 5), [trans_label + 1]

def zoom_image (new_image):
    # new input is a uint8 numpy array with shape (img_height, img_width, 3)
    # input is uint8 eagerTensor, non-normalized (values between 0 and 255)
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
    edges = cv2.Canny(np.array(new_image), params[0], params[1], apertureSize = aperture_size, L2gradient = L2Gradient) # 150, 75 confirmed
    ymin = h
    xmin = w
    ymax = 0
    xmax = 0
    coords = [ymin, xmin, ymax, xmax]
    
    # Find left bound/xmin
    left_bound = np.zeros([h, w]) # 150, 75 confirmed
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
    top_bound = np.zeros([w, h]) # 75, 150 confirmed
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
    coords = [0, 0, h, w]
    
    x_offset = tf.random.uniform((), 0 , tf.cast(640-w, tf.int32), dtype=tf.int32)
    y_offset = tf.random.uniform((), 0 , tf.cast(640-h, tf.int32), dtype=tf.int32)
    new_image = tf.image.resize(np.array(new_image), (h, w))
    new_image = tf.reshape(new_image, (list(np.shape(new_image))[0],list(np.shape(new_image))[1],3))
    new_image = tf.image.pad_to_bounding_box(new_image, y_offset, x_offset, 640, 640)
    #new_image = tf.cast(new_image, tf.float32)/255.0
    new_image = tf.cast(new_image, tf.uint8)
    
    coord_adder = [y_offset, x_offset, y_offset, x_offset]
    coords = np.array([sum(i) for i in zip(coords, coord_adder)])
    coords = np.array([(i / 640) for i in coords])

    return new_image, coords

def unison_shuffle(arr1, arr2):
    assert len(arr1) == len(arr2)
    p = np.random.permutation(len(arr1))
    return np.array(arr1)[p], np.array(arr2)[p]

# def augment_set(dataset, training = True, num_images = None, set_name = None, ex_sub = True):
#     if num_images == None:
#         num_images = len(dataset)
#     train_images_np = [0] * num_images
#     gt_boxes = [0] * num_images
#     train_labels = [0] * num_images
#     subset = dataset.shuffle(len(dataset)).take(num_images)
#     with tqdm(total=num_images, desc=str(set_name), unit="images") as pbar:
#         for i in range(num_images):
#             new_image, label = next(iter(subset))
#             train_labels[i] = get_label(label)
#             new_image = zoom_image(new_image)
#             if ex_sub:
#                 new_image = extract_sub(np.array(new_image))
#             new_image, new_coords = process_image(new_image)
#             train_images_np[i] = np.array(new_image)
#             if training:
#                 gt_boxes[i] = np.array([new_coords], dtype=np.float32)
#             pbar.update(1)
#     if training:
#         print('Succeeded for ' + str(len(train_images_np)) + ' of ' + str(num_images), flush=True)
#         return train_images_np, gt_boxes, train_labels
#     else:
#         print('Succeeded for ' + str(len(train_images_np)) + ' of ' + str(num_images), flush=True)
#         return train_images_np
        
def augment_npset(dataset, opt_labels, training = True, num_images = None, set_name = None, ex_sub = True):
    assert(len(dataset) == len(opt_labels))
    if num_images == None:
            num_images = len(dataset)
    train_images_np = [0] * num_images
    gt_boxes = [0] * num_images
    train_labels = [0] * num_images
    subset, new_labels = unison_shuffle(np.array(dataset), opt_labels)
    with tqdm(total=num_images, desc=str(set_name), unit="images") as pbar:
        for i in range(num_images):
            new_image = subset[i]
            label = new_labels[i]
            train_labels[i] = get_label(label)
            new_image = zoom_image(new_image)
            if ex_sub:
                new_image = extract_sub(np.array(new_image))
            new_image, new_coords = process_image(new_image)
            train_images_np[i] = np.array(new_image)
            if training:
                gt_boxes[i] = np.array([new_coords], dtype=np.float32)
            pbar.update(1)
    if training:
        print('Succeeded for ' + str(len(train_images_np)) + ' of ' + str(num_images), flush=True)
        return train_images_np, gt_boxes, train_labels
    else:
        print('Succeeded for ' + str(len(train_images_np)) + ' of ' + str(num_images), flush=True)
        return train_images_np

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

def read_image_sizes(file_name):
    image = skimage.io.imread(IMAGE_PATH + file_name)
    return list(image.shape)
    
def draw_category_images(var,cols=5):
    categories = (honey_bee_df.groupby([var])[var].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=cols, figsize=(2*cols,2*len(categories)))
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = honey_bee_df[honey_bee_df[var]==cat].sample(cols)
        for j in range(0,cols):
            file=IMAGE_PATH + sample.iloc[j]['file']
            im=imageio.imread(file)
            ax[i, j].imshow(im, resample=True)
            ax[i, j].set_title(cat, fontsize=9)  
    plt.tight_layout()
    plt.show()
    
def draw_trace_box(dataset,var, subspecies):
    dfS = dataset[dataset['subspecies']==subspecies];
    trace = go.Box(
        x = dfS[var],
        name=subspecies,
        marker=dict(
                    line=dict(
                        color='black',
                        width=0.8),
                ),
        text=dfS['subspecies'], 
        orientation = 'h'
    )
    return trace

def draw_group(dataset, var, title,height=500):
    data = list()
    for subs in subspecies:
        data.append(draw_trace_box(dataset, var, subs))
        
    layout = dict(title = title,
              xaxis = dict(title = 'Size',showticklabels=True),
              yaxis = dict(title = 'Subspecies', showticklabels=True, tickfont=dict(
                family='Old Standard TT, serif',
                size=8,
                color='black'),), 
              hovermode = 'closest',
              showlegend=False,
                  width=600,
                  height=height,
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='subspecies-image')
    
def read_image(file_name):
    image = skimage.io.imread(IMAGE_PATH + file_name)
    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), mode='reflect')
    return image[:,:,:IMAGE_CHANNELS]

def categories_encoder(dataset, var='subspecies'):
    X = np.stack(dataset['file'].apply(read_image))
    y = pd.get_dummies(dataset[var], drop_first=False)
    return X, y

def create_trace(x,y,ylabel,color):
        trace = go.Scatter(
            x = x,y = y,
            name=ylabel,
            marker=dict(color=color),
            mode = "markers+lines",
            text=x
        )
        return trace
    
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1,len(acc)+1))
    #define the traces
    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")
    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")
    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")
    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")
    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',
                                                             'Training and validation loss'))
    #add traces to the figure
    fig.append_trace(trace_ta,1,1)
    fig.append_trace(trace_va,1,1)
    fig.append_trace(trace_tl,1,2)
    fig.append_trace(trace_vl,1,2)
    #set the layout for the figure
    fig['layout']['xaxis'].update(title = 'Epoch')
    fig['layout']['xaxis2'].update(title = 'Epoch')
    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])
    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])
    #plot
    iplot(fig, filename='accuracy-loss')
    
def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

def PD_dataset(file_list,size=(300,180),flattened=False):
    data = []
    for i, file in enumerate(file_list):
        image = io.imread(file)
        image = transform.resize(image, size, mode='constant')
        if flattened:
            image = image.flatten()

        data.append(image)

    labels = [1 if f.split("images")[-1][1] == 'P' else 0 for f in file_list]
    #labels = [print(f.split("images")[-1][1]) for f in file_list]

    return np.array(data), np.array(labels)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, CLIP_HIST_PERCENTAGE):
    img_float32 = np.float32(image)
    lab_image = cv.cvtColor(img_float32, cv.COLOR_RGB2HSV)
    gray = cv2.cvtColor(lab_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)