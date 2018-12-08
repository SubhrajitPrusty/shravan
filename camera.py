# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3
import requests
import cv2
import numpy as np
import time


# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://172.16.0.153:8080/video'


while True:
    # Use urllib to get the image from the IP camera
    imgResp = requests.get(url).content

    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.content),dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;)
    img = cv2.imdecode(imgNp,-1)


	# put the image on screen
    cv2.imshow('IPWebcam',img)

    #To give the processor some less stress
    #time.sleep(0.1)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
