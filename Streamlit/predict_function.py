import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def openModel():
    PATH_TO_SAVED_MODEL="model/saved_model"
    #PATH_TO_LABELS="model/annotations/label_map.pbtxt"
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    #category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
    return detect_fn

def detectImage(image,model,category_index):
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    return image_np_with_detections

def detectVideo_online(frame,model,category_index):
    frame[:,:, [2, 0]] = frame[:,:, [0, 2]] 
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = frame.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    image_np_with_detections[:,:, [2, 0]] = image_np_with_detections[:,:, [0, 2]] 
    frame_res = cv2.resize(image_np_with_detections, (int(frame.shape[1]), int(frame.shape[0])), interpolation=cv2.INTER_CUBIC)
    return frame_res