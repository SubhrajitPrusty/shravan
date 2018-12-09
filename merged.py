import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2

import speech_recognition as sr  

import pyttsx3 as p
import time

from utils import label_map_util
from utils import visualization_utils as vis_util
import requests
import geocoder

from weather import Weather, Unit

from map import *

# # Model preparation 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print ('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    print ('Download complete')
else:
    print ('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device
def getImage():
    LINK = "http://10.42.0.99:8080/shot.jpg"
    r = requests.get(LINK)
    nparr = np.fromstring(r.content, np.uint8)
    cimg = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    return cimg

cap = cv2.VideoCapture(0)
# cap = getImage()                        #phoneCam
def speak(str):
    engine =p.init()
    engine.say(str)
    engine.runAndWait()
    time.sleep(1)
    
    
def listener():
    retu=input('write: ')
    return (retu)
    
def listener1():
    r = sr.Recognizer()                                                                                   
    with sr.Microphone() as source:                                                                       
        print("Speak:")                                                                                   
        audio = r.listen(source)   

    try:
        print("You said " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    return(r.recognize_google(audio))
    

todo=0
        
def dodger():
	for i,b in enumerate(boxes[0]):
		if classes[0][i] == 3 or classes[0][i]==6 or classes[0][i]==1 or classes[0][i]==8:
			if scores[0][i]>0.5:
				mid_x = (boxes[0][i][3] +boxes[0][i][1])/2
				mid_y = (boxes[0][i][2] +boxes[0][i][0])/2
				apx_distance = round((1-(boxes[0][i][3] - boxes[0][i][1]))**4,1)
				cv2.putText(image_np,'{}'.format(apx_distance),(int(mid_x*800),int(mid_y*450)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
				if apx_distance<=0.5: 
					if mid_x >0.3 and mid_x<0.5:
						cv2.putText(image_np,'WARNING!',(int(mid_x*800)-50,int(mid_y*450)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
						speak("stop something is infront of you , try right")
					if mid_x >0.5 and mid_x<0.7:
						cv2.putText(image_np,'WARNING!',(int(mid_x*800)-50,int(mid_y*450)),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
						speak("stop something is infront of you, try left")
                         
                  
def ask():
    global todo
    speak("what do you want me to do")
    print("what do you want me to do?")
    print("1: find X\n2: walk with me\n3: how is the weather\n4: where am i")
    instruct=input("Answer : ")
    lis=instruct.split(" ")

    if lis[0]=='find':
        todo=1

        #speak('what do you want to search for')
        #print('speak')
        #stri=listener()
        #obj=stri
        
        obj=lis[-1]
        print('finding '+obj)
        return obj
        
    elif lis[0]=='walk':
        todo=2
        return None
    elif lis[-1]=='weather':
        todo=3
        return None
    elif lis[0] == 'where':
        todo=4
        return None
    else:
        todo=0
        return None
        
def finder(obj):
	for i in range(len(boxes[0])):
		score = scores[0][i]
		name = category_index[classes[0][i]]['name']
		if score > 0.5 and name == obj:
			speak(name)
			print(name, "is in center")
              
def climate():
	g = geocoder.ip('me')
	lat, lon = g.latlng
	print(lat, lon)

	weather = Weather(unit=Unit.CELSIUS)

	weather = Weather(Unit.CELSIUS)
	lookup = weather.lookup_by_latlng(lat, lon)
	condition = lookup.condition
	speak("it is "+condition.text+" outside")
              
# Running the tensorflow session
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		ret = True
		while (ret):
			ret,image_np = cap.read()
			# image_np = getImage()
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
			#      plt.figure(figsize=IMAGE_SIZE)
			#      plt.imshow(image_np)
			if todo==0:
				obj = func()
			if todo==1:
				finder(obj)
			if todo==2:
				dodger()
			if todo==3:
				climate()
				break
			if todo==4:
				lat, lon = getCoord()
				addr = getAddr(lat, lon)
				print("You are at " + addr['display_name'])
				speak("You are at " + addr['display_name'])
				break

			cv2.imshow('image',cv2.resize(image_np,(640,480)))
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				cap.release()
				break

cv2.destroyAllWindows()
cap.release()