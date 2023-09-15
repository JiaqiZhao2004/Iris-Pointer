from helen_dataset_into_list import data_into_list
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from image_collection import extract_eye
import cv2
# noinspection PyPep8Naming
import albumentations as A
from tqdm import tqdm

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

image_dir = "helen_dataset/img"
annotation_dir = "helen_dataset/annotation"

train_data = data_into_list(image_dir, annotation_dir)
test_data = train_data[int(0.9 * len(train_data)):]
train_data = train_data[:int(0.9 * len(train_data))]


resize_transform = A.Compose([
    A.LongestMaxSize(1024),
    A.PadIfNeeded(min_width=1024),
    A.PadIfNeeded(min_width=2048),
    # A.PadIfNeeded(min_height=2048),
], keypoint_params=A.KeypointParams(format='xy'))


train_transforms = A.Compose([
    A.Resize(height=128, width=128),
    # A.GaussianBlur(),
    # A.ColorJitter(),
    # A.GridDropout(ratio=0.3)
], keypoint_params=A.KeypointParams(format='xy'))

test_transforms = A.Compose([
    A.Resize(height=128, width=128),
], keypoint_params=A.KeypointParams(format='xy'))


class HelenEyeDataset(Dataset):
    def __init__(self, data, transform=None, resize=None, left=True, eye_extractor=extract_eye):
        super().__init__()
        self.data = data
        self.eye_extractor = eye_extractor
        self.transform = transform
        self.left = left
        self.resize = resize

    def __getitem__(self, index):
        """
        data = [['img_path',
                (left eye)[[x1, y1], [x2, y2], ..., [x20, y20]],
                (right eye)[[x1, y1], [x2, y2], ...,[x20, y20]]
                ],
                ...]
        """
        # print(index)
        # print(self.data[index][0])

        image = cv2.imread(self.data[index][0])
        if self.left:
            key_points = self.data[index][1]  # [[x1, y1], [x2, y2], ..., [x20, y20]]
        else:
            key_points = self.data[index][2]

        resized = self.resize(image=image, keypoints=key_points)
        image = resized['image']
        key_points = [list(ele) for ele in resized['keypoints']]
        eye_crop, x_bleed, y_bleed = self.eye_extractor(image, left=self.left, face_cascade=faceCascade, eye_cascade=eyeCascade)
        for index in range(len(key_points)):  # shift position of label points
            key_points[index][0] -= x_bleed
            key_points[index][1] -= y_bleed
        transformed = self.transform(image=eye_crop, keypoints=key_points)
        eye_crop = transformed['image']
        key_points = [list(ele) for ele in transformed['keypoints']]

        eye_crop = torch.tensor(eye_crop)
        key_points = torch.tensor(key_points)
        return eye_crop, key_points, image

    def __len__(self):
        return len(self.data)


train_dataset = HelenEyeDataset(data=train_data, transform=train_transforms, resize=resize_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
test_dataset = HelenEyeDataset(data=test_data, transform=test_transforms, resize=resize_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# iterator = iter(train_loader)
# eyes, labels, images = next(iterator)
#
# fig, ax = plt.subplots(4, 4)
# for i in range(4):
#     for j in range(4):
#         eye1 = eyes[i*4+j]
#         label1 = labels[i*4+j]
#         ax[i, j].imshow(eye1.numpy())
#         for k in range(len(label1)):
#             ax[i, j].scatter(label1[k][0], label1[k][1])
#
# plt.show()

# train eye model
loop = tqdm(train_loader)
for (i, (eye, label, img)) in enumerate(loop):
    pass
# #ValueError: Expected x for keypoint (90.67617391193983, 76.1610401357508, 0.0, 0.0) to be in the range [0.0, 62], got 90.67617391193983.
