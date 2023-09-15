from helen_dataset_into_list import data_into_list
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from image_collection import extract_eye
import cv2
# noinspection PyPep8Naming
import albumentations as A

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

image_dir = "helen_dataset/img"
annotation_dir = "helen_dataset/annotation"

train_data = data_into_list(image_dir, annotation_dir)
test_data = train_data[int(0.9 * len(train_data)):]
train_data = train_data[:int(0.9 * len(train_data))]


train_transforms = A.Compose([
    A.Resize(height=128, width=128),
    A.GaussianBlur(),
    A.ColorJitter(),
    # A.GridDropout(ratio=0.3)
])


class HelenEyeDataset(Dataset):
    def __init__(self, data, transform=None, left=True, eye_extractor=extract_eye):
        super().__init__()
        self.data = data
        self.eye_extractor = eye_extractor
        self.transform = transform
        self.left = left

    def __getitem__(self, index):
        """
        data = [['img_path',
                (left eye)[[x1, y1], [x2, y2], ..., [x20, y20]],
                (right eye)[[x1, y1], [x2, y2], ...,[x20, y20]]
                ],
                ...]
        """
        index = 338
        print(index)
        img = cv2.imread(self.data[index][0])
        print(self.data[index][0])
        eye, x_bleed, y_bleed = self.eye_extractor(img, left=self.left, face_cascade=faceCascade, eye_cascade=eyeCascade)
        if self.left:
            label = self.data[index][1]  # [[x1, y1], [x2, y2], ..., [x20, y20]]
        else:
            label = self.data[index][2]
        for i in range(len(label)):  # shift position of label points
            label[i][0] -= x_bleed
            label[i][1] -= y_bleed
        eye = self.transform(image=eye)['image']
        eye = torch.tensor(eye)
        label = torch.tensor(label)
        return eye, label

    def __len__(self):
        return len(self.data)


train_dataset = HelenEyeDataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

iterator = iter(train_loader)
eyes, labels = next(iterator)
eye1 = eyes.squeeze(0)
label1 = labels[0]
print(label1)
plt.imshow(eye1.numpy())
plt.show()

# train eye model
"""
510
helen_dataset/img/2233368704_3.jpg
"""
