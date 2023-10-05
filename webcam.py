import math
import time

import cv2
from image_collection import extract_face, find_pupil_x_position, stabilized_pointer
import torch
import albumentations as A
import numpy as np
import pyautogui as mouse
from webcam_init import initialize_webcam
from triangulation import transform_x_window

CAMERA = 0
SHOW = True
MODE = "eye"  # eye, face, eye_binary, full
LINE = True
SIDE = 128
FACE_STABILIZATION = 1
EYE_STABILIZATION = 1
STICKINESS = 3
GENERAL_SLOWDOWN = 0.001
PUPIL_STABILIZATION = 2
THETA_X = 40 / 180 * math.pi
THETA_Y = 40 / 180 * math.pi
REAL_HEAD_WIDTH = 16
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
face_coordinates, eye_coordinates, video_capture, model, triangulator, x_position_mid, distance_mid = \
    initialize_webcam(faceCascade, camera_mode=CAMERA, theta_x=THETA_X, theta_y=THETA_Y, real_head_width=REAL_HEAD_WIDTH)

window = transform_x_window(distance=distance_mid)
print("Window = {}, Mid = {}, Distance init = {} cm".format(round(window, 3), round(x_position_mid, 3), round(distance_mid)))

x_position_stable = 0.5
test_transforms = A.Resize(height=128, width=128)
font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame_time = 0
new_frame_time = 0

print("Detection Started")

while True:
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
    prev_frame_time = new_frame_time
    fps = round(fps, 1)
    fps = str(fps)

    ret, frame = video_capture.read()

    x_min, x_max, y_min, y_max = extract_face(frame, faceCascade)
    # stabilization
    scale_factor = (FACE_STABILIZATION - 1) / FACE_STABILIZATION
    face_coordinates = face_coordinates * scale_factor
    face_coordinates += np.array([x_min, x_max, y_min, y_max]) / FACE_STABILIZATION
    x_min = face_coordinates[0]
    x_max = face_coordinates[1]
    y_min = face_coordinates[2]
    y_max = face_coordinates[3]
    width = x_max - x_min
    height = y_max - y_min
    face = frame[int(y_min): int(y_max), int(x_min): int(x_max), :]
    face = test_transforms(image=face)["image"]
    face_tensor = torch.tensor(face, dtype=torch.int16).permute([2, 0, 1]).unsqueeze(0).to('cpu')
    with torch.no_grad():
        keypoints = model(face_tensor)[0].numpy() * 2

    coord = []
    for i in range(4):
        coord.append([int(keypoints[2 * i] * SIDE), int(keypoints[2 * i + 1] * SIDE)])

    # stabilization: make eye detection an x-frame-average
    eye_coordinates = eye_coordinates * (EYE_STABILIZATION - 1) / EYE_STABILIZATION
    eye_coordinates += np.array(coord) / EYE_STABILIZATION
    center_x = 0
    center_y = 0
    for row in eye_coordinates:
        center_x += row[0]
        center_y += row[1]
    eye_center_x = center_x / 4 / 128 * width + x_min
    eye_center_y = center_y / 4 / 128 * height + y_min

    # triangulation
    distance = round(triangulator.find_distance(y_min, y_max))
    eye_displacement_x, eye_displacement_y = \
        triangulator.find_eye_displacement(
            eye_center_x=eye_center_x,
            eye_center_y=eye_center_y,
            y_min=y_min,
            y_max=y_max
        )

    x_left_margin = x_position_mid - window / 2
    x_right_margin = x_position_mid + window / 2

    to_show = frame
    text_board = np.zeros((300, 500))
    if MODE == "full":
        for i in range(4):
            cv2.circle(
                img=frame,
                center=(int(eye_coordinates[i][0] / 128 * (x_max - x_min) + x_min), int(eye_coordinates[i][1] / 128 * (y_max - y_min) + y_min)),
                radius=3,
                color=(0, 0, 255),
                thickness=-1
            )
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 1)

        cv2.putText(frame, "Distance=" + str(distance) + "cm", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    if MODE == "face":
        for i in range(4):
            cv2.circle(face, (int(eye_coordinates[i][0]), int(eye_coordinates[i][1])), radius=1, color=(0, 0, 255), thickness=-1)
        to_show = face

    elif MODE == "eye":
        try:
            to_show = frame[
                      int(eye_coordinates[2][1] / 128 * (y_max - y_min) + y_min): int(eye_coordinates[3][1] / 128 * (y_max - y_min) + y_min),
                      int(eye_coordinates[1][0] / 128 * (x_max - x_min) + x_min): int(eye_coordinates[0][0] / 128 * (x_max - x_min) + x_min)
                      ]
        except None:
            pass

    elif MODE == "eye_binary":
        try:
            eye = frame[
                  int(eye_coordinates[2][1] / 128 * (y_max - y_min) + y_min): int(eye_coordinates[3][1] / 128 * (y_max - y_min) + y_min),
                  int(eye_coordinates[1][0] / 128 * (x_max - x_min) + x_min): int(eye_coordinates[0][0] / 128 * (x_max - x_min) + x_min)
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
        assert MODE in ["eye"], "Line function only applicable in MODE = eye"
        x = find_pupil_x_position(to_show)  # with respect to eye frame
        if x is not None:
            x_position = x
            current = (1 - (x_position - x_left_margin) / (x_right_margin - x_left_margin))  # with respect to box
            # x_position_stable is with respect to box

            x_on_screen = stabilized_pointer(
                previous=x_position_stable, current=current, n_average=PUPIL_STABILIZATION,
                stickiness=STICKINESS, general_slowdown=GENERAL_SLOWDOWN)
            x_position_stable = x_on_screen
            x_on_screen = round(min(mouse.size()[0], max(x_on_screen * mouse.size()[0], 0)))

            if SHOW:
                cv2.line(
                    to_show,
                    (int(x_position * to_show.shape[1]), 0),
                    (int(x_position * to_show.shape[1]), to_show.shape[0]),
                    (0, 255, 0),
                    1
                )
                cv2.line(
                    to_show,
                    (int(to_show.shape[1] * x_position_mid), 0),
                    (int(to_show.shape[1] * x_position_mid), to_show.shape[0]),
                    (0, 255, 0),
                    1
                )
                cv2.rectangle(
                    to_show,
                    (int(to_show.shape[1] * x_left_margin), 0),
                    (int(to_show.shape[1] * x_right_margin), to_show.shape[0]),
                    (0, 0, 255),
                    1
                )
                cv2.putText(text_board, "x=" + str(round(x_position_stable, 2)), (10, 50), font, 1, (255, 255, 255))
                cv2.putText(text_board, "Distance=" + str(distance) + "cm", (10, 100), font, 1, (255, 255, 255))
                cv2.putText(text_board, "Move to " + str(x_on_screen), (10, 150), font, 1, (255, 255, 255))
                cv2.putText(text_board, fps, (10, 200), font, 1, (100, 255, 0))
                cv2.putText(text_board, "Eye Displacement = ({}, {})".format(eye_displacement_x, eye_displacement_y), (10, 250), font, 1, (255, 255, 255))
            mouse.moveTo(x_on_screen, None)

    if SHOW:
        cv2.imshow("Video", to_show)
        cv2.imshow("Text", text_board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
