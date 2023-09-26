import cv2
from albumentations import PadIfNeeded
from model import get_model
import torch
import albumentations as A
import numpy as np

# eye, face
MODE = "face"
SIDE = 128
STABILIZATION = 2
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')
video_capture = cv2.VideoCapture(1)
model = get_model("weights/4_points_resnet34_linear_1_small_img_epoch=334_loss=7.995e-05.pth")
model.to('cpu')
resize_transform = A.Compose([
    A.ToGray(),
    A.LongestMaxSize(512),
    A.PadIfNeeded(min_height=512, min_width=512, position=PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_REPLICATE),
])

test_transforms = A.Compose([
    A.Resize(height=SIDE, width=SIDE)
])

eye_coordinates = np.array([[50.785, 48.86], [30.65, 47.124], [38.83098592, 43.5915493], [37.0565, 37.0565]])
face_coordinates = np.array([798.4959016, 300.1639344, 447.7663934, 447.7663934])

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # train_sample_frame = cv2.imread("train_sample_img.jpg")
    # frame = train_sample_frame
    # print(frame)
    # frame = resize_transform(image=frame)["image"]

    faces = faceCascade.detectMultiScale(
        frame,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (nx, ny, nw, nh) in faces:
        # print(str(nx) + "\t" + str(ny) + "\t" + str(nw) + "\t" + str(nh))
        # nx = int(x + 0.1 * w)
        # nw = int(0.4 * w)
        # ny = int(ny - 0.1 * nh)
        # nh = int(1.1 * nh)
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




        if MODE == "face":
            for i in range(4):
                # print(str(int(keypoints_ul[2 * i] * SIDE)) + "\t" + str(int(keypoints_ul[2 * i + 1] * SIDE)))
                # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * nw), int(ny + ((keypoints_ul[2 * i + 1] * 300 - (150 - (nh * 150) / nw)) / (300 / nw)))),
                #            radius=5, color=(0, 0, 255), thickness=-1)
                cv2.circle(face, (int(eye_coordinates[i][0]), int(eye_coordinates[i][1])), radius=1, color=(0, 0, 255), thickness=-1)
                # cv2.circle(frame, (int(nx + keypoints_ul[2 * i] * 300 / 256 * nw), int(ny + (300 * keypoints_ul[2 * i + 1] - 300 + (256 * nh) / nw) / (256
                # / nw))), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.imshow('Video', face)
        elif MODE == "eye":
            eye = face[int(eye_coordinates[2][1]): int(eye_coordinates[3][1]), int(eye_coordinates[1][0]): int(eye_coordinates[0][0])]
            cv2.imshow('Video', eye)
        # eyes = eyeCascade.detectMultiScale( frame[ny: ny + nh, nx: nx + nw], minNeighbors=0 ) # draw a rectangle around eyes for (ex, ey, ew,
        # eh) in eyes: cv2.rectangle(frame, (nx + ex - int(0.15 * ew), ny + ey - int((1.3 * ew - eh) // 2)), (nx + ex + int(1.2 * ew), ny + ey + eh + int((
        # 1.3 * ew - eh) // 2)), (0, 255, 255), 2) eye1 = frame[ny + ey: ny + ey + eh, nx + ex: nx + ex + ew] # ret, binary = cv2.threshold(eye1, 60, 255,
        # cv2.THRESH_BINARY_INV) # cv2.imshow('Video', binary)

    # Display the resulting frame
    # entire frame
    # cv2.imshow('Video', frame)  # eye only

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
