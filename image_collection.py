import time
import tkinter as tk
import cv2


class ImageCollector(tk.Frame):
    def __init__(self, rt):
        super().__init__()
        self.root = rt
        self.label_ul = tk.Label(self.root, name="level_ul", text="Look at the TOP-LEFT corner of the screen, then click 'Ready'.")
        self.label_ur = tk.Label(self.root, name="level_ur", text="Look at the TOP-RIGHT corner of the screen, then click 'Ready'")
        self.label_bl = tk.Label(self.root, name="level_ll", text="Look at the BOTTOM-LEFT corner of the screen, then click 'Ready'")
        self.label_br = tk.Label(self.root, name="level_lr", text="Look at the BOTTOM-RIGHT corner of the screen, then click 'Ready'")
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


def take_corner_image(corner, camera_code):
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
    image_collector.mainloop()

    cam = cv2.VideoCapture(camera_code)
    time.sleep(0.2)
    result, image = cam.read()
    assert result, "Unable to connect to camera..."
    assert sum(image[200][200]) > 0, "Image is totally dark"
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
    image = cv2.flip(image, 1)  # flip horizontally
    return image


def extract_eye(frame, left, face_cascade, verbose=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    test_min_neighbors = [0, 1, 2, 3, 4, 5, 10, 15, 20, 50, 80, 100, 0]
    for i in range(len(test_min_neighbors)):
        min_neighbors = test_min_neighbors[i]
        faces = face_cascade.detectMultiScale(
            gray,
            minNeighbors=test_min_neighbors[i],
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0 or i == len(test_min_neighbors) - 2:
            faces = face_cascade.detectMultiScale(
                gray,
                minNeighbors=test_min_neighbors[i - 1],
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
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
        x_min = x
        x_max = x + w
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
    return 0, gray.shape[0], 0, gray.shape[1]
