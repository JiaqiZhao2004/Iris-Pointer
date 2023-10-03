import cv2
from image_collection import extract_face, find_pupil_x_position, stabilized_pointer
import torch
import albumentations as A
import numpy as np
import pyautogui as mouse
from webcam_init import initialize_webcam
from triangulation import find_distance, transform_x_window

SHOW = True
MODE = "eye"  # eye, face, eye_binary, full
LINE = True
SIDE = 128
FACE_STABILIZATION = 2
EYE_STABILIZATION = 2
PUPIL_STABILIZATION = 2
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")

x_position_ul, x_position_ur, x_position_bl, x_position_br, face_coordinates, eye_coordinates, video_capture, model, distance_init \
        = initialize_webcam(faceCascade, camera_mode=1)

window = x_position_ur - x_position_ul
window_should_be = transform_x_window(distance=distance_init)
print("Top Left: {}, Top Right: {}, Bottom Left: {}, Bottom Right: {}, Distance: {} cm."
      .format(x_position_ul, x_position_ur, x_position_bl, x_position_br, distance_init))
print("Window = {}, Should_Be {}.".format(round(window, 3), round(window_should_be, 3)))

while abs(window - window_should_be) > 0.1:
    x_position_ul, x_position_ur, x_position_bl, x_position_br, face_coordinates, eye_coordinates, video_capture, model, distance_init \
        = initialize_webcam(faceCascade, camera_mode=1)
    print("Top Left: {}, Top Right: {}, Bottom Left: {}, Bottom Right: {}, Distance: {} cm."
          .format(x_position_ul, x_position_ur, x_position_bl, x_position_br, distance_init))
    window = x_position_ur - x_position_ul
    window_should_be = transform_x_window(distance=distance_init)
    print("Window = {}, Should_Be {}.".format(round(window, 3), round(window_should_be, 3)))

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

    # triangulation
    distance = round(find_distance(frame.shape[0], y_min, y_max))
    x_window_default = x_position_ur - x_position_ul
    x_window_transformed = transform_x_window(distance) / transform_x_window(distance_init) * x_window_default

    x_left_margin = x_position_ul
    x_right_margin = x_position_ur

    x_left_margin = x_position_ul + (x_window_default - x_window_transformed) / 2
    x_right_margin = x_left_margin + x_window_transformed
    # print(x_window_default, x_window_transformed, x_left_margin, x_right_margin, distance)

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
    text_board = np.zeros((200, 500))
    if MODE == "full":
        for i in range(4):
            cv2.circle(frame, (int(eye_coordinates[i][0] / 128 * (x_max - x_min) + x_min), int(eye_coordinates[i][1] / 128 * (y_max - y_min) + y_min)), radius=3, color=(0, 0, 255),
                       thickness=-1)
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
        x_position = find_pupil_x_position(to_show)  # with respect to eye frame
        current = (1 - (x_position - x_left_margin) / (x_right_margin - x_left_margin))  # with respect to box
        # x_position_stable is with respect to box

        x_on_screen = stabilized_pointer(
            previous=x_position_stable, current=current, n_average=PUPIL_STABILIZATION,
            stickiness=3, general_slowdown=0.003)
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
            cv2.rectangle(
                to_show,
                (int(to_show.shape[1] * x_left_margin), 0),
                (int(to_show.shape[1] * x_right_margin), to_show.shape[0]),
                (0, 0, 255),
                1
            )
            cv2.putText(text_board, "x=" + str(round(x_position_stable, 2)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(text_board, "Distance=" + str(distance) + "cm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(text_board, "Move to " + str(x_on_screen), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        mouse.moveTo(x_on_screen, None)

    if SHOW:
        cv2.imshow("Video", to_show)
        cv2.imshow("Text", text_board)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
