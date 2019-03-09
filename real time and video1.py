# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:28:18 2018

@author: Pranav
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:19:04 2018

@author: Pranav
"""


# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import zipfile
import imageio
imageio.plugins.ffmpeg.download()
from datetime import datetime
#from collections import defaultdict

#from distutils.version import StrictVersion
#from collections import defaultdict
#from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#import visvis as vv 
#import pygame
from gtts import gTTS

import tkinter as tk
from tkinter import filedialog

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops

#if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup
# In[2]:


# This is needed to display the images.
#Sget_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.max_open_warning': 20})


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# ## Download Model

# In[5]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
#IMAGE_SIZE = (12, 8)


# In[10]:





def write_slogan():
    with detection_graph.as_default():    
        with tf.Session(graph=detection_graph)as sess:
            image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
            num_detection =detection_graph.get_tensor_by_name('num_detections:0')
            
            input_video = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes=(("jpeg files","*.jpg"),("All files", "*")))#'traffic'
            #input_video = input_video.split('/')[-1]
            video_reader = imageio.get_reader('%s'%input_video)
            print('Video Processing.......')
            video_writer = imageio.get_writer('%s.annotated.mp4'%input_video,fps = 10)
            ##loop through and process each frame
            t0 = datetime.now()
            n_frames = 0
            for frame in video_reader:
                image_np = frame
                n_frames+=1
                
                ##Expand dim since the expects images to have shape[1.None]
                image_np_expanded = np.expand_dims(image_np,axis = 0)
                #actual detection
                (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detection],feed_dict={image_tensor: image_np_expanded})
                #visualization
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                #os.system("mpg321 video.mp3")
                #Video Writer
                video_writer.append_data(image_np)
                #print(image_np)    
            fps = n_frames/(datetime.now()-t0).total_seconds()
            
            
            print('Frames processed: %s,Speed:%s fps'%(n_frames,fps))
            '''pygame.mixer.init()
            pygame.mixer.music.load("welcome.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                continue'''    
            #plt.show(image_np)
            #language = 'en-uk'
            #with open("Ouput.txt", encoding="utf-8") as file:
            #    file=file.read()
            #    speak = gTTS(text="I think its "+file, lang=language, slow=False)
            #    file=("sold%s.mp3")
            #    speak.save(file)
            #    #cleanup
            print('Video Processing Done.......')
            video_writer.close()
            
def cam():
        ####Real Time
        import cv2
        cap = cv2.VideoCapture(0)
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                  ret, image_np = cap.read()
                  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                  image_np_expanded = np.expand_dims(image_np, axis=0)
                  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                  # Each box represents a part of the image where a particular object was detected.
                  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                  # Each score represent how level of confidence for each of the objects.
                  # Score is shown on the result image, together with the class label.
                  scores = detection_graph.get_tensor_by_name('detection_scores:0')
                  classes = detection_graph.get_tensor_by_name('detection_classes:0')
                  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                  # Actual detection.
                  (boxes, scores, classes, num_detections) = sess.run(
                      [boxes, scores, classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                  # Visualization of the results of a detection.
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,
                      line_thickness=8)
                  
                  cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
                  if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    #language = 'en-uk'
                    #with open("Ouput.txt", encoding="utf-8") as file:
                    #    file=file.read()
                    #    speak = gTTS(text="I think its "+file, lang=language, slow=False)
                    ##    file=("sold%s.mp3")
                     #   speak.save(file)
                        #file.close()
                    break
                    
                
                    
#def image():

def image():  
      
    with detection_graph.as_default():    
        with tf.Session(graph=detection_graph)as sess:
            image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores=detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes=detection_graph.get_tensor_by_name('detection_classes:0')
            num_detection =detection_graph.get_tensor_by_name('num_detections:0')
            
            input_video = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes=(("jpeg files","*.jpg"),("All files", "*")))#'traffic'
            image = Image.open(input_video)
            #input_video = input_video.split('/')[-1]
            #video_reader = imageio.get_reader('%s'%input_video)
            print('Image Processing.......')
            video_writer = imageio.get_writer('%s.annotated.jpg'%input_video)
            ##loop through and process each frame
            #t0 = datetime.now()
            #n_frames = 0
            image_np = load_image_into_numpy_array(image)
            #image_np = video_reader
            #for frame in video_reader:
            #    image_np = frame
            #    n_frames+=1
                
            ##Expand dim since the expects images to have shape[1.None]
            image_np_expanded = np.expand_dims(image_np,axis = 0)
            #actual detection
            (boxes,scores,classes,num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detection],feed_dict={image_tensor: image_np_expanded})
            #visualization
            vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)
            #Video Writer
            video_writer.append_data(image_np)
            #print(image_np)    
            #fps = n_frames/(datetime.now()-t0).total_seconds()
            
            #print('Frames processed: %s,Speed:%s fps'%(n_frames,fps))
            print('Image Processing Done.....')   
            #pygame.mixer.init()
            #pygame.mixer.music.load("welcome1.mp3")
            #pygame.mixer.music.play()
            #while pygame.mixer.music.get_busy() == True:
            #    continue
            
            
                #z=z+1
            #cleanup
            #print([category_index.get(i)for i in classes[0]])
            #print(scores[[0]])
            #os.system("mpg321 image.mp3")
            video_writer.close()
            #f = open("Ouput.txt", "w+")
            #f.close()            #f = open("Ouput.txt", "a")
            #f.write(classes)
            #language = 'en-uk'
            #with open("Ouput.txt", encoding="utf-8") as file:
             #   file=file.read()
             #   speak = gTTS(text="I think its "+file, lang=language, slow=False)
             #   file=("sold%s.mp3")
             #   speak.save(file)
             #   print(file)
                #file.close()
            
            #plt.figure(figsize=IMAGE_SIZE)
            #plt.imshow(image_np)







root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, 
                   text="CAMERA", 
                   fg="red",
                   command=cam)
button.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="VIDEO",
                   command=write_slogan)
slogan.pack(side=tk.LEFT)
button1 = tk.Button(frame, 
                   text="IMAGE", 
                   fg="red",
                   command=image)
button1.pack(side=tk.RIGHT)

root.mainloop()