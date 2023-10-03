import cv2
import matplotlib.pyplot as plt

from model import get_model
import torch
import albumentations as A
import numpy as np
from image_collection import extract_face, corner_to_x_position
from triangulation import find_distance


def initialize_webcam(face_cascade, camera_mode=1, repeat=20):
    print("Setting up webcam...")
    video_capture = cv2.VideoCapture(camera_mode)

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

    # from corner 'xx' to x position
    x_position_ul, distance_ul = \
        corner_to_x_position(corner='ul', cam=camera_mode, face_cascade=face_cascade, test_transforms=test_transforms, model=model, repeat=repeat)
    x_position_ur, distance_ur = \
        corner_to_x_position(corner='ur', cam=camera_mode, face_cascade=face_cascade, test_transforms=test_transforms, model=model, repeat=repeat)
    # x_position_bl, distance_bl = \
    #     corner_to_x_position(corner='bl', cam=camera_mode, face_cascade=face_cascade, test_transforms=test_transforms, model=model, repeat=repeat)
    # x_position_br, distance_br = \
    #     corner_to_x_position(corner='br', cam=camera_mode, face_cascade=face_cascade, test_transforms=test_transforms, model=model, repeat=repeat)
    x_position_bl, x_position_br = 0, 0
    return x_position_ul, x_position_ur, x_position_bl, x_position_br, face_coordinates, eye_coordinates, video_capture, model, (distance_ul + distance_ur) // 2
