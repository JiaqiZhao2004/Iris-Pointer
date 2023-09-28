import cv2
from image_collection import extract_face
import torch
import albumentations as A
import numpy as np
import pyautogui as mouse
from webcam_init import initialize_webcam

SHOW = True
MODE = "eye"  # eye, face, eye_binary, full
LINE = True
SIDE = 128
FACE_STABILIZATION = 5
EYE_STABILIZATION = 5
PUPIL_STABILIZATION = 20
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")

x_position_ul, x_position_ur, x_position_bl, x_position_br, face_coordinates, eye_coordinates, video_capture, model = initialize_webcam(faceCascade)
print("Top Left: {}, Top Right: {}, Bottom Left: {}, Bottom Right: {}.".format(x_position_ul, x_position_ur, x_position_bl, x_position_br))
x_position_stable = 0.5
test_transforms = A.Resize(height=128, width=128)

print("Detection Started")

while True:
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
    face = frame[int(y_min): int(y_max), int(x_min): int(x_max)]
    face = test_transforms(image=face)["image"]

    with torch.no_grad():
        face_tensor = torch.tensor(face, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
        keypoints = model(face_tensor).numpy()[0] * 2
        coord = []
        for i in range(4):
            coord.append([int(keypoints[2 * i] * SIDE), int(keypoints[2 * i + 1] * SIDE)])

        # stabilization: make eye detection an x-frame-average
        eye_coordinates = eye_coordinates * (EYE_STABILIZATION - 1) / EYE_STABILIZATION
        eye_coordinates += np.array(coord) / EYE_STABILIZATION

    to_show = frame

    if MODE == "full":
        for i in range(4):
            cv2.circle(frame, (int(eye_coordinates[i][0] / 128 * (x_max - x_min) + x_min), int(eye_coordinates[i][1] / 128 * (y_max - y_min) + y_min)), radius=3, color=(0, 0, 255),
                       thickness=-1)

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

        if SHOW:
            cv2.line(
                to_show,
                (vertical_accumulation.index(max(vertical_accumulation)) + cut_off_left + window_half, 0),
                (vertical_accumulation.index(max(vertical_accumulation)) + cut_off_left + window_half, to_show.shape[0]),
                (0, 0, 255),
                1
            )

        x_left_margin = x_position_ul
        x_right_margin = x_position_ur

        x_position_on_image = vertical_accumulation.index(max(vertical_accumulation)) + cut_off_left + window_half

        x_position = (x_position_on_image - (to_show.shape[1] * x_left_margin)) / (to_show.shape[1] * (x_right_margin - x_left_margin))

        if SHOW:
            cv2.rectangle(
                to_show,
                (int(to_show.shape[1] * x_left_margin), 0),
                (int(to_show.shape[1] * x_right_margin), to_show.shape[0]),
                (0, 0, 255),
                1
            )

        x_position_stable = x_position_stable * (PUPIL_STABILIZATION - 1) / PUPIL_STABILIZATION
        x_position_stable += x_position / PUPIL_STABILIZATION

        mouse.moveTo((1 - x_position_stable) * mouse.size()[0], None)

    if SHOW:
        cv2.imshow("Video", to_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
