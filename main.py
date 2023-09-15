import cv2
import matplotlib.pyplot as plt
from image_collection import take_corner_image, extract_eye

CAMERA_MODE = 1
LEFT = False

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

# image of eye looking at four corners
image_ul = take_corner_image(corner='ul', camera_code=CAMERA_MODE)
image_ur = take_corner_image(corner='ur', camera_code=CAMERA_MODE)
image_bl = take_corner_image(corner='bl', camera_code=CAMERA_MODE)
image_br = take_corner_image(corner='br', camera_code=CAMERA_MODE)

# extract eye (gray image)
eye_ul = extract_eye(frame=image_ul, left=LEFT, face_cascade=faceCascade, eye_cascade=eyeCascade)[0]
eye_ur = extract_eye(frame=image_ur, left=LEFT, face_cascade=faceCascade, eye_cascade=eyeCascade)[0]
eye_bl = extract_eye(frame=image_bl, left=LEFT, face_cascade=faceCascade, eye_cascade=eyeCascade)[0]
eye_br = extract_eye(frame=image_br, left=LEFT, face_cascade=faceCascade, eye_cascade=eyeCascade)[0]

# plotting
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(eye_ul)
ax[0, 1].imshow(eye_ur)
ax[1, 0].imshow(eye_bl)
ax[1, 1].imshow(eye_br)
ax[0, 0].set_title("upper left")
ax[0, 1].set_title("upper right")
ax[1, 0].set_title("lower left")
ax[1, 1].set_title("lower right")
plt.show()


# fig, ax = plt.subplots(4, 4)
# for i in range(4):
#     for j in range(4):
#         ax[i, j].imshow(cv2.threshold(eye_ul, 48 + 6 * (4 * i + j), 255, cv2.THRESH_BINARY_INV)[1])
#         ax[i, j].set_title(48 + 6 * (4 * i + j))
#
# plt.show()
