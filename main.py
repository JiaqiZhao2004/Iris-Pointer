import cv2
import matplotlib.pyplot as plt
import torch
import albumentations as A
from image_collection import take_corner_image, extract_eye
from model import get_model
from PIL import Image

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CAMERA_MODE = 1
SIZE = 256
LEFT = True
WEIGHT_PATH = "weights/New Folder With Items/4_points_resnet34_linear_1000_epoch=5_loss=0.01118.pth"

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

# image of eye looking at four corners
image_ul = take_corner_image(corner='ul', camera_code=CAMERA_MODE)
image_ur = take_corner_image(corner='ur', camera_code=CAMERA_MODE)
image_bl = take_corner_image(corner='bl', camera_code=CAMERA_MODE)
image_br = take_corner_image(corner='br', camera_code=CAMERA_MODE)

# extract eye (gray image)
resize_and_pad = A.Compose([
    A.LongestMaxSize(100),
    A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_REPLICATE, position=A.PadIfNeeded.PositionType.BOTTOM_LEFT),
])

eye_ul, nx_ul, ny_ul, nw_ul, nh_ul = extract_eye(frame=image_ul, left=LEFT, face_cascade=faceCascade)
eye_ur, nx_ur, ny_ur, nw_ur, nh_ur = extract_eye(frame=image_ur, left=LEFT, face_cascade=faceCascade)
eye_bl, nx_bl, ny_bl, nw_bl, nh_bl = extract_eye(frame=image_bl, left=LEFT, face_cascade=faceCascade)
eye_br, nx_br, ny_br, nw_br, nh_br = extract_eye(frame=image_br, left=LEFT, face_cascade=faceCascade)
eye_ul = resize_and_pad(image=eye_ul)['image']
eye_ur = resize_and_pad(image=eye_ur)['image']
eye_bl = resize_and_pad(image=eye_bl)['image']
eye_br = resize_and_pad(image=eye_br)['image']

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

# Eye keypoint detection
model = get_model(WEIGHT_PATH)
model.to(DEVICE)
keypoints_ul = model(torch.tensor(eye_ul, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]
print(keypoints_ul)
for i in range(4):
    ax[0, 0].scatter(int(keypoints_ul[2 * i] * 300), int((keypoints_ul[2 * i + 1] * 300))),
plt.show()


# fig, ax = plt.subplots(4, 4)
# for i in range(4):
#     for j in range(4):
#         ax[i, j].imshow(cv2.threshold(eye_ul, 48 + 6 * (4 * i + j), 255, cv2.THRESH_BINARY_INV)[1])
#         ax[i, j].set_title(48 + 6 * (4 * i + j))
#
# plt.show()
