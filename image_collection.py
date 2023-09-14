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


def extract_eye(frame, left, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        minNeighbors=10,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        if left:
            nx = int(x + 0.1 * w)
        else:
            nx = int(x + 0.4 * w)

        nw = int(0.5 * w)
        ny = int(y + 0.2 * h)
        nh = int(0.3 * h)

        # cv2.rectangle(frame, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)  # draw rect around ROI

        eyes = eye_cascade.detectMultiScale(
            frame[ny: ny + nh, nx: nx + nw],
        )

        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(frame, (nx + ex, ny + ey), (nx + ex + ew, ny + ey + eh), (0, 255, 255), 2)  # draw a rectangle around eyes
            gray = gray[ny + ey: ny + ey + eh, nx + ex: nx + ex + ew]
            # ret, binary = cv2.threshold(eye1, 60, 255, cv2.THRESH_BINARY_INV)
    return gray
