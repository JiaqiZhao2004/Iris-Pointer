import math

import cv2
from model import get_model
import torch
import albumentations as A
import numpy as np
MODE = "eye"  # eye, face, eye_binary, full
LINE = True
SIDE = 128
STABILIZATION = 10
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

print("Setting up webcam...")
video_capture = cv2.VideoCapture(1)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2286)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1524)

print("Getting model...")
model = get_model("weights/4_points_resnet34_linear_1_small_img_epoch=334_loss=7.995e-05.pth")
model.to('cpu')
test_transforms = A.Compose([A.Resize(height=SIDE, width=SIDE)])

print("Initializing eye and face coordinates...")
face_coordinates = None
eye_coordinates = None

while face_coordinates is None or eye_coordinates is None:
    ret, frame = video_capture.read()

    faces = faceCascade.detectMultiScale(
        frame,
        minNeighbors=20,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (nx, ny, nw, nh) in faces:
        face_coordinates = np.array([nx, ny, nw, nh])

        face = frame[
               int(face_coordinates[1]): int(face_coordinates[1] + face_coordinates[3]),
               int(face_coordinates[0]): int(face_coordinates[0] + face_coordinates[2])
               ]
        face = test_transforms(image=face)["image"]
        with torch.no_grad():
            face_tensor = torch.tensor(face, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
            keypoints_ul = model(face_tensor).numpy()[0] * 2
            coord = []
            for i in range(4):
                coord.append([int(keypoints_ul[2 * i] * SIDE), int(keypoints_ul[2 * i + 1] * SIDE)])
            eye_coordinates = np.array(coord)

print("Detection Started")
while True:
    ret, frame = video_capture.read()

    faces = faceCascade.detectMultiScale(
        frame,
        minNeighbors=20,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (nx, ny, nw, nh) in faces:
        # print(str(nx) + "\t" + str(ny) + "\t" + str(nw) + "\t" + str(nh))
        # nx = int(x + 0.1 * w)
        # nw = int(0.4 * w)
        # ny = int(ny - 0.1 * nh)
        # nh = int(0.9 * nh)
        # print("({},{}),({},{})".format(nx, ny, nx + nw, ny + nh))
        # cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

        # stabilization
        face_coordinates = face_coordinates * (STABILIZATION - 1) / STABILIZATION
        face_coordinates += np.array([nx, ny, nw, nh]) / STABILIZATION

        face = frame[
               int(face_coordinates[1]): int(face_coordinates[1] + face_coordinates[3]),
               int(face_coordinates[0]): int(face_coordinates[0] + face_coordinates[2])
               ]
        face = test_transforms(image=face)["image"]
        with torch.no_grad():
            face_tensor = torch.tensor(face, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
            keypoints_ul = model(face_tensor).numpy()[0] * 2
            coord = []
            for i in range(4):
                coord.append([int(keypoints_ul[2 * i] * SIDE), int(keypoints_ul[2 * i + 1] * SIDE)])

            # stabilization: make eye detection an x-frame-average
            eye_coordinates = eye_coordinates * (STABILIZATION - 1) / STABILIZATION
            eye_coordinates += np.array(coord) / STABILIZATION

        to_show = frame

        if MODE == "full":
            for i in range(4):
                cv2.circle(frame, (int(eye_coordinates[i][0] / 128 * nw + nx), int(eye_coordinates[i][1] / 128 * nh + ny)), radius=3, color=(0, 0, 255),
                           thickness=-1)

        if MODE == "face":
            for i in range(4):
                cv2.circle(face, (int(eye_coordinates[i][0]), int(eye_coordinates[i][1])), radius=1, color=(0, 0, 255), thickness=-1)
            to_show = face

        elif MODE == "eye":
            try:
                to_show = frame[
                      int(eye_coordinates[2][1] / 128 * nh + ny): int(eye_coordinates[3][1] / 128 * nh + ny),
                      int(eye_coordinates[1][0] / 128 * nw + nx): int(eye_coordinates[0][0] / 128 * nw + nx)
                      ]
            except None:
                pass

        elif MODE == "eye_binary":
            try:
                eye = frame[
                      int(eye_coordinates[2][1] / 128 * nh + ny): int(eye_coordinates[3][1] / 128 * nh + ny),
                      int(eye_coordinates[1][0] / 128 * nw + nx): int(eye_coordinates[0][0] / 128 * nw + nx)
                      ]
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

                thresh = 0
                for i in range(50, 255):
                    ret, binary = cv2.threshold(gray_eye, i, 255, cv2.THRESH_BINARY)
                    if ((np.array(binary) == 0) / (binary.shape[0] * binary.shape[1])).sum() >= 0.3:
                        thresh = i
                        break
                ret, to_show = cv2.threshold(gray_eye, thresh, 255, cv2.THRESH_BINARY)
            except None:
                pass

        if LINE:
            assert MODE in ["eye", "eye_binary"], "Line function only applicable in MODE = eye or eye_binary"
            cut_off_left = int(to_show.shape[1] * 0.15)
            cut_off_right = int(to_show.shape[1] * 1)
            box = to_show[:, cut_off_left: cut_off_right, 2]
            pad = 0.06
            pad_left = int(box.shape[1] * pad)
            pad_right = pad_left
            box = np.pad(box, ((0, 0), (pad_left, pad_right)), 'maximum')

            if MODE == "eye":
                thresh = 0
                for i in range(50, 255):
                    ret, binary = cv2.threshold(box, i, 255, cv2.THRESH_BINARY)
                    if ((np.array(binary) == 0) / (binary.shape[0] * binary.shape[1])).sum() >= 0.3:
                        thresh = i
                        break
                ret, box = cv2.threshold(box, thresh, 255, cv2.THRESH_BINARY)

            binary_np = (np.array(box) == 0)
            vertical_accumulation = []
            window_half = int(binary_np.shape[1] * pad)
            for i in range(window_half, binary_np.shape[1] - window_half - 1):
                vertical_accumulation.append((binary_np[:, i - window_half: i + window_half]).sum())
            cv2.line(
                to_show,
                (vertical_accumulation.index(max(vertical_accumulation)) + cut_off_left + window_half, 0),
                (vertical_accumulation.index(max(vertical_accumulation)) + cut_off_left + window_half, to_show.shape[0]),
                (0, 0, 255),
                1
            )
        cv2.imshow("Video", to_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
