import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

#Load model
model = model_from_json(open("./fer.json","r").read())
model.load_weights('./fer.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

map_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
src = cv2.VideoCapture(0)
while(True):
    ret, frame = src.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(grayFrame, 1.32, 5)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = grayFrame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48))
        inp = image.img_to_array(roi_gray)
        inp = np.expand_dims(inp,axis=0)
        inp /= 255

        pred = model.predict(inp)
        emot = map_emotions[np.argmax(pred[0])]
        cv2.putText(frame, emot, (int(x), int(y)),cv2.FONT_HERSHEY_DUPLEX, 1, (2,3,255), 2)

        # roi_color = frame[y:y+h, x:x+w]
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
src.release() 
cv2.destroyAllWindows() 
    