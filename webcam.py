import random

import cv2
import sys

from albumentations import PadIfNeeded

from model import get_model
import torch
import albumentations as A

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')
video_capture = cv2.VideoCapture(1)
model = get_model("weights/contour_resnet34_linear_1_small_img_epoch=975_loss=8.365e-05.pth")
model.to('cpu')
SIDE = 128
resize_transform = A.Compose([
    A.ToGray(),
    A.LongestMaxSize(512),
    A.PadIfNeeded(min_height=512, min_width=512, position=PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_REPLICATE),
])

test_transforms = A.Compose([
    A.Resize(height=SIDE, width=SIDE)
])


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # train_sample_frame = cv2.imread("train_sample_img.jpg")
    # frame = train_sample_frame
    # print(frame)
    # frame = resize_transform(image=frame)["image"]

    faces = faceCascade.detectMultiScale(
        frame,
        minNeighbors=50,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (nx, ny, nw, nh) in faces:
        # nx = int(x + 0.1 * w)
        # nw = int(0.4 * w)
        # ny = int(y + 0.2 * h)
        # nh = int(0.3 * h)
        # print("({},{}),({},{})".format(nx, ny, nx + nw, ny + nh))

        # cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
        eye = frame[ny: ny + nh, nx: nx + nw]
        eye = test_transforms(image=eye)["image"]
        with torch.no_grad():
            eye_tensor = torch.tensor(eye, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
            keypoints_ul = model(eye_tensor).numpy()[0] * 2
        # print(keypoints_ul)
        for i in [0, 5, 10, 15, 20, 25, 30, 35]:
            # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * nw), int(ny + ((keypoints_ul[2 * i + 1] * 300 - (150 - (nh * 150) / nw)) / (300 / nw)))),
            #            radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(eye, (int(keypoints_ul[2 * i] * SIDE), int(keypoints_ul[2 * i + 1] * SIDE)),
                       radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * 300 / 256 * nw), int(ny + (300 * keypoints_ul[2 * i + 1] - 300 + (256 * nh) / nw) / (256 / nw))),
            #            radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow('Video', eye)
        # eyes = eyeCascade.detectMultiScale(
        #     frame[ny: ny + nh, nx: nx + nw],
        #     minNeighbors=0
        # )
        # # draw a rectangle around eyes
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(frame, (nx + ex - int(0.15 * ew), ny + ey - int((1.3 * ew - eh) // 2)), (nx + ex + int(1.2 * ew), ny + ey + eh + int((1.3 * ew - eh) // 2)), (0, 255, 255), 2)
        #     eye1 = frame[ny + ey: ny + ey + eh, nx + ex: nx + ex + ew]
        #     # ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
        #     # cv2.imshow('Video', binary)

    # Display the resulting frame
    # entire frame
    # cv2.imshow('Video', frame)  # eye only

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
