import os

import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from tqdm import tqdm
import albumentations as A


cascPath = "haar_cascade_frontal_face_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(1)
anterior = 0


test_transforms = A.Compose([
    A.Resize(height=128, width=128)
])


images = sorted(os.listdir("helen_dataset/img"))[1:]

for name in tqdm(images):
    path = os.path.join("helen_dataset/img", name)
    frame = cv2.imread(path)
    faces = faceCascade.detectMultiScale(
        frame,
        minNeighbors=30,
        minSize=(200, 200)
    )

    for (x, y, w, h) in faces:
        face = frame[y: y+h, x: x+w]
        face = test_transforms(image=face)["image"]
        cv2.imwrite("helen_dataset/img_head_only/" + name, face)
