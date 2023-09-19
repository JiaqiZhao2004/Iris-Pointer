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
model = get_model("weights/eye only/4_points_resnet34_linear_1000_epoch=32_loss=0.05105.pth")
model.to('cpu')

test_transforms = A.Compose([
    A.LongestMaxSize(256),
    A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_REPLICATE, position=A.PadIfNeeded.PositionType.CENTER),
])

train_transforms = A.Compose([
    A.LongestMaxSize(256),
    A.Resize(height=random.choice([10, 60, 100, 256, 400]), width=256),
    A.PadIfNeeded(min_height=120, min_width=256, position=PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_REPLICATE, p=1),
    A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.TOP_LEFT,
                  border_mode=cv2.BORDER_REPLICATE, p=0.5),
    A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.BOTTOM_RIGHT,
                  border_mode=cv2.BORDER_REPLICATE, p=1),
    A.PadIfNeeded(min_height=300, min_width=300, position=PadIfNeeded.PositionType.TOP_LEFT,
                  border_mode=cv2.BORDER_REPLICATE),
    A.Blur()
])

while True:
    # Capture frame-by-frame
    # ret, frame = video_capture.read()
    train_sample_frame = cv2.imread("train_sample_img.jpg")
    gray = cv2.cvtColor(train_sample_frame, cv2.COLOR_BGR2GRAY)
    frame = train_sample_frame

    faces = faceCascade.detectMultiScale(
        gray,
        minNeighbors=10,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        nx = int(x + 0.1 * w)
        nw = int(0.4 * w)
        ny = int(y + 0.2 * h)
        nh = int(0.3 * h)
        print("({},{}),({},{})".format(nx, ny, nx + nw, ny + nh))

        cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)
        eye = frame[ny: ny + nh, nx: nx + nw]
        eye = train_transforms(image=eye)["image"]
        eee = eye
        eye = torch.tensor(eye, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
        keypoints_ul = model(eye).cpu().detach().numpy()[0]
        print(keypoints_ul)
        for i in range(4):
            # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * nw), int(ny + ((keypoints_ul[2 * i + 1] * 300 - (150 - (nh * 150) / nw)) / (300 / nw)))),
            #            radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(eee, (int(keypoints_ul[2 * i] * 300), int(keypoints_ul[2 * i + 1] * nh * 300 / nw * nh/nw)),
                       radius=5, color=(0, 0, 255), thickness=-1)
            # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * 300 / 256 * nw), int(ny + (300 * keypoints_ul[2 * i + 1] - 300 + (256 * nh) / nw) / (256 / nw))),
            #            radius=5, color=(0, 0, 255), thickness=-1)
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
    cv2.imshow('Video', eee)  # entire frame
    # cv2.imshow('Video', frame)  # eye only

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
