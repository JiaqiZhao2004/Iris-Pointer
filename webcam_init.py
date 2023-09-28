import cv2
import matplotlib.pyplot as plt

from model import get_model
import torch
import albumentations as A
import numpy as np
from image_collection import extract_face, take_corner_image, keypoints_to_eye_frame, find_pupil_x_position


def initialize_webcam(face_cascade, camera_mode=1, camera_width=2286, camera_height=1524):
    print("Setting up webcam...")
    video_capture = cv2.VideoCapture(1)
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_width)

    print("Getting model...")
    model = get_model("weights/4_points_resnet34_linear_1_small_img_epoch=334_loss=7.995e-05.pth")
    model.to('cpu')
    test_transforms = A.Resize(height=128, width=128)

    print("Initializing eye and face coordinates...")
    face_coordinates = None
    eye_coordinates = None

    while face_coordinates is None or eye_coordinates is None:
        ret, frame = video_capture.read()
        x_min, x_max, y_min, y_max = extract_face(frame, face_cascade, True)

        face_coordinates = np.array([x_min, x_max, y_min, y_max], dtype='float16')
        print("Face Anchor: ", face_coordinates.tolist())
        face = frame[y_min: y_max, x_min: x_max]

        face = test_transforms(image=face)["image"]

        with torch.no_grad():
            face_tensor = torch.tensor(face, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to('cpu')
            keypoints_ul = model(face_tensor).numpy()[0] * 2
            coord = []
            for i in range(4):
                coord.append([int(keypoints_ul[2 * i] * 128), int(keypoints_ul[2 * i + 1] * 128)])
            eye_coordinates = np.array(coord)
            print("Eye Anchor: ", eye_coordinates.tolist())

    # image of eye looking at four corners
    image_ul = take_corner_image(corner='ul', camera_code=camera_mode)
    image_ur = take_corner_image(corner='ur', camera_code=camera_mode)
    image_bl = take_corner_image(corner='bl', camera_code=camera_mode)
    image_br = take_corner_image(corner='br', camera_code=camera_mode)

    x_min_ul, x_max_ul, y_min_ul, y_max_ul = extract_face(frame=image_ul, face_cascade=face_cascade)
    x_min_ur, x_max_ur, y_min_ur, y_max_ur = extract_face(frame=image_ur, face_cascade=face_cascade)
    x_min_bl, x_max_bl, y_min_bl, y_max_bl = extract_face(frame=image_bl, face_cascade=face_cascade)
    x_min_br, x_max_br, y_min_br, y_max_br = extract_face(frame=image_br, face_cascade=face_cascade)

    face_ul = image_ul[y_min_ul: y_max_ul, x_min_ul: x_max_ul, :]
    face_ur = image_ur[y_min_ur: y_max_ur, x_min_ur: x_max_ur, :]
    face_bl = image_bl[y_min_bl: y_max_bl, x_min_bl: x_max_bl, :]
    face_br = image_br[y_min_br: y_max_br, x_min_br: x_max_br, :]

    face_ul = test_transforms(image=face_ul)["image"]
    face_ur = test_transforms(image=face_ur)["image"]
    face_bl = test_transforms(image=face_bl)["image"]
    face_br = test_transforms(image=face_br)["image"]

    faces_corners = torch.tensor(np.array([face_ul, face_ur, face_bl, face_br]), dtype=torch.int16).permute([0, 3, 1, 2]).to('cpu')
    with torch.no_grad():
        keypoints = model(faces_corners).numpy() * 2

    eye_ul = keypoints_to_eye_frame(image_ul, keypoints[0], x_min_ul, x_max_ul, y_min_ul, y_max_ul)
    eye_ur = keypoints_to_eye_frame(image_ur, keypoints[1], x_min_ur, x_max_ur, y_min_ur, y_max_ur)
    eye_bl = keypoints_to_eye_frame(image_bl, keypoints[2], x_min_bl, x_max_bl, y_min_bl, y_max_bl)
    eye_br = keypoints_to_eye_frame(image_br, keypoints[3], x_min_br, x_max_br, y_min_br, y_max_br)

    x_position_ul = find_pupil_x_position(eye_ul)
    x_position_ur = find_pupil_x_position(eye_ur)
    x_position_bl = find_pupil_x_position(eye_bl)
    x_position_br = find_pupil_x_position(eye_br)
    return x_position_ul, x_position_ur, x_position_bl, x_position_br, face_coordinates, eye_coordinates, video_capture, model
