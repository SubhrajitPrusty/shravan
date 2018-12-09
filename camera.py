import cv2
import numpy as np
import requests


def getImage():
    LINK = "http://172.16.0.153:8080/shot.jpg"
    r = requests.get(LINK)
    nparr = np.fromstring(r.content, np.uint8)
    cimg = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    return cimg

if __name__ == "__main__":
    while True:
        img = getImage()
        cv2.imshow("CAMERA", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

