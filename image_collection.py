import time
import tkinter as tk
import cv2
import numpy as np
import torch

from triangulation import find_distance


class ImageCollector(tk.Frame):
    def __init__(self, rt):
        super().__init__()
        self.root = rt
        self.label_ul = tk.Label(self.root, name="level_ul", text="Look at the TOP-LEFT corner of the screen, then click 'Ready'.")
        self.label_ur = tk.Label(self.root, name="level_ur", text="Look at the TOP-RIGHT corner of the screen, then click 'Ready'")
        self.label_bl = tk.Label(self.root, name="level_ll", text="Look at the BOTTOM-LEFT corner of the screen, then click 'Ready'")
        self.label_br = tk.Label(self.root, name="level_lr", text="Look at the BOTTOM-RIGHT corner of the screen, then click 'Ready'")
        self.label_mid = tk.Label(self.root, name="level_mid", text="Look directly at the CAMERA, then click 'Ready'")
        self.close_button = tk.Button(self.root, text="Ready", command=self.close)

    def close(self):
        self.root.destroy()

    def ul(self):
        self.label_ul.pack()
        self.close_button.pack()

    def ur(self):
        self.label_ur.pack()
        self.close_button.pack()

    def bl(self):
        self.label_bl.pack()
        self.close_button.pack()

    def br(self):
        self.label_br.pack()
        self.close_button.pack()

    def mid(self):
        self.label_mid.pack()
        self.close_button.pack()


def take_corner_image(corner, camera_code, repeat):
    root = tk.Tk()
    image_collector = ImageCollector(root)
    match corner:
        case 'ul':
            image_collector.ul()
        case 'ur':
            image_collector.ur()
        case 'bl':
            image_collector.bl()
        case 'br':
            image_collector.br()
        case 'mid':
            image_collector.mid()
    image_collector.mainloop()

    cam = cv2.VideoCapture(camera_code)
    time.sleep(0.2)
    images = []
    for i in range(repeat):
        result, image = cam.read()
        assert result, "Unable to connect to camera..."
        assert sum(image[200][200]) > 0, "Image is totally dark"
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = cv2.flip(image, 1)  # flip horizontally
        images.append(image)
    return images


def extract_face(frame, face_cascade, verbose=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    test_min_neighbors = [5, 10, 20, 50, 100, 5]
    for i in range(len(test_min_neighbors)):
        min_neighbors = test_min_neighbors[i]
        faces = face_cascade.detectMultiScale(
            gray,
            minNeighbors=test_min_neighbors[i],
            minSize=(300, 300),
        )
        if len(faces) == 0 or i == len(test_min_neighbors) - 2:
            faces = face_cascade.detectMultiScale(
                gray,
                minNeighbors=test_min_neighbors[i - 1],
                minSize=(300, 300),
            )
            if verbose:
                print("Detected {} face(s) at min_neighbors = {}".format(len(faces), min_neighbors))
            break

    for (x, y, w, h) in faces:
        # if left:
        #     nx = int(x + 0.1 * w)
        # else:
        #     nx = int(x + 0.4 * w)
        # nw = int(0.5 * w)
        # ny = int(y + 0.2 * h)
        # nh = int(0.3 * h)
        # eyes = []
        # x_min = x - int(w * 0.1) if (x - int(w * 0.1)) > 0 else 0
        # x_max = x + int(w * 1.1)
        x_min = x
        x_max = x + h
        y_min = y
        y_max = y + h
        return x_min, x_max, y_min, y_max
        # # cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)  # draw rect around ROI
        # test_min_neighbors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 0]
        # for i in range(len(test_min_neighbors)):
        #     min_neighbors = test_min_neighbors[i]
        #     eyes = eye_cascade.detectMultiScale(
        #         eye_detecting_frame,
        #         minNeighbors=min_neighbors
        #     )
        #     if len(eyes) == 0 or i == len(test_min_neighbors) - 2:
        #         eyes = eye_cascade.detectMultiScale(
        #             eye_detecting_frame,
        #             minNeighbors=test_min_neighbors[i - 1]
        #         )
        #         if verbose:
        #             print("Detected {} eye(s) at min_neighbors = {}".format(len(eyes), test_min_neighbors[i - 1]))
        #         break

        # if len(eyes) > 0:
        #     for (ex, ey, ew, eh) in eyes:
        #         # print("Eye position ",
        #         #       (nx + ex - int(0.3 * ew), ny + ey - int((1.3 * ew - eh) // 2)),
        #         #       (nx + ex + int(1.3 * ew), ny + ey + eh + int((1.3 * ew - eh) // 2)))
        #         x_left = expansion
        #         x_right = x_left
        #         eye = frame[
        #               ny + ey - int(((1 + x_left + x_right) * ew - eh) / 2)   : ny + ey + eh + int(((1 + x_left + x_right) * ew - eh) / 2),
        #               nx + ex - int(x_left * ew)               : nx + ex + int((1 + x_right) * ew)
        #               ]
        #         # ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
        #         return eye, nx + ex - int(x_left * ew), ny + ey - int(((1 + x_left + x_right) * ew - eh) / 2)
        # else:
        #     print("No eye detected")
        #     eye = eye_detecting_frame
        #     return eye, nx, ny
    print("No Face Detected")
    return 0, gray.shape[0], 0, gray.shape[1]


def find_pupil_x_position(eye_frame):
    # gray = eye_frame[:, :, 2]
    cut_off_left = int(eye_frame.shape[1] * 0.15)
    box = eye_frame[:, cut_off_left:, 2]
    pad = 0.1
    pad_left = int(box.shape[1] * pad)
    pad_right = pad_left
    box = np.pad(box, ((0, 0), (pad_left, pad_right)), 'maximum')

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

    a = vertical_accumulation.index(max(vertical_accumulation)) + window_half
    x_coord_on_image = a + cut_off_left
    x_position = x_coord_on_image / eye_frame.shape[1]
    # print("x_position = ", x_position)
    return x_position


def keypoints_to_eye_frame(frame, keypoints, face_x_min, face_x_max, face_y_min, face_y_max):
    coord = []
    for i in range(4):
        coord.append([int(keypoints[2 * i] * 128), int(keypoints[2 * i + 1] * 128)])
    coord = np.array(coord)
    # dilation
    eye_frame = frame[
                int(coord[2][1] / 128 * (face_y_max - face_y_min) + face_y_min): int(coord[3][1] / 128 * (face_y_max - face_y_min) + face_y_min),
                int(coord[1][0] / 128 * (face_x_max - face_x_min) + face_x_min): int(coord[0][0] / 128 * (face_x_max - face_x_min) + face_x_min)
                ]

    return eye_frame


def corner_to_x_position(corner, cam, face_cascade, test_transforms, model, repeat):
    x_position = []
    distance = []
    images = take_corner_image(corner=corner, camera_code=cam, repeat=repeat)
    for image in images:
        x_min, x_max, y_min, y_max = extract_face(frame=image, face_cascade=face_cascade)
        distance.append(round(find_distance(image.shape[0], y_min, y_max)))

        face = image[y_min: y_max, x_min: x_max, :]
        face = test_transforms(image=face)["image"]
        faces_corners = torch.tensor(face, dtype=torch.int16).permute([2, 0, 1]).unsqueeze(0).to('cpu')
        with torch.no_grad():
            keypoints = model(faces_corners)[0].numpy() * 2
        eye = keypoints_to_eye_frame(image, keypoints, x_min, x_max, y_min, y_max)
        x_position.append(find_pupil_x_position(eye))

    x_position.sort()
    # print("x_position_list = ", x_position)
    x_position = x_position[repeat // 3:-repeat // 3]
    return sum(x_position) / len(x_position), sum(distance) / len(distance)


def stabilized_pointer(previous, current, n_average=2, stickiness=1.5, general_slowdown=0.3):  # decimal between 0 and 1
    current = (previous * (n_average - 1) + current) / n_average
    difference = abs(previous - current)
    previous_weight = general_slowdown / (difference + 0.05) ** stickiness
    previous_weight = previous_weight / (previous_weight + 1)
    current_weight = 1 - previous_weight
    output = previous * previous_weight + current * current_weight
    return output
