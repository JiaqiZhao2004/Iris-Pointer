from helen_dataset_into_list import data_into_list
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from image_collection import extract_eye
import cv2
# noinspection PyPep8Naming
import albumentations as A
from tqdm import tqdm
from model import get_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
VERBOSE = False
BATCH_SIZE = 16
WEIGHT_PATH = "weights/resnet_34_2linear_epoch=13_loss=0.00622.pth"
NUM_EPOCHS = 50
COMPLETED = 0

faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")
eyeCascade = cv2.CascadeClassifier('haar_cascade_eye.xml')

image_dir = "helen_dataset/img"
annotation_dir = "helen_dataset/annotation"

train_data = data_into_list(image_dir, annotation_dir)
test_data = train_data[int(0.9 * len(train_data)):]
train_data = train_data[:int(0.9 * len(train_data))]

resize_transform = A.Compose([
    A.LongestMaxSize(1024),
    A.PadIfNeeded(min_height=1024, min_width=1024),
], keypoint_params=A.KeypointParams(format='xy'))

train_transforms = A.Compose([
    A.LongestMaxSize(256),
    A.PadIfNeeded(min_height=310, min_width=310, border_mode=cv2.BORDER_REPLICATE),
    A.RandomSizedCrop(min_max_height=(256, 256), height=256, width=256)
    # A.GaussianBlur(),
    # A.ColorJitter(),
    # A.GridDropout(ratio=0.3)
], keypoint_params=A.KeypointParams(format='xy'))

test_transforms = A.Compose([
    A.LongestMaxSize(256),
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_REPLICATE),
    # A.GaussianBlur(),
    # A.ColorJitter(),
    # A.GridDropout(ratio=0.3)
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
        if VERBOSE:
            print(index)
            print(self.data[index][0])

        image = cv2.imread(self.data[index][0])
        if self.left:
            key_points = self.data[index][1]  # [[x1, y1], [x2, y2], ..., [x20, y20]]
        else:
            key_points = self.data[index][2]

        resized = self.resize(image=image, keypoints=key_points)
        image = resized['image']
        key_points = [list(ele) for ele in resized['keypoints']]
        eye_crop, x_bleed, y_bleed = self.eye_extractor(image, left=self.left, face_cascade=faceCascade, eye_cascade=eyeCascade, verbose=VERBOSE)
        for index in range(len(key_points)):  # shift position of label points
            key_points[index][0] -= x_bleed
            key_points[index][1] -= y_bleed
        transformed = self.transform(image=eye_crop, keypoints=key_points)
        eye_crop = transformed['image']
        key_points = [list(ele) for ele in transformed['keypoints']]

        target = []
        for i in range(len(key_points)):
            target.append(key_points[i][0])
            target.append(key_points[i][1])
        eye_crop = torch.tensor(eye_crop)
        target = torch.tensor(target) / 256
        target = target.to(torch.float32)
        return eye_crop, target, image

    def __len__(self):
        return len(self.data)


train_dataset = HelenEyeDataset(data=train_data, transform=train_transforms, resize=resize_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = HelenEyeDataset(data=test_data, transform=test_transforms, resize=resize_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# iterator = iter(train_loader)
# eyes, labels, images = next(iterator)
# fig, ax = plt.subplots(4, 4)
# for i in range(4):
#     for j in range(4):
#         eye1 = eyes[i*4+j]
#         label1 = labels[i*4+j] * 256
#         ax[i, j].imshow(eye1.numpy())
#         for k in range(20):
#             ax[i, j].scatter(label1[2 * k], label1[2 * k + 1])
#
# plt.show()

# train eye model

eye_key_points_detector = get_model(WEIGHT_PATH)
eye_key_points_detector.to(DEVICE)


def train_one_epoch(model, optim, data_loader, epoch_index, device=DEVICE):
    print("Epoch {}".format(COMPLETED + epoch_index + 1))
    model.train()
    loop = tqdm(data_loader)
    total_loss = []
    for (i, (eyes, labels, _)) in enumerate(loop):
        eyes = torch.permute(eyes, [0, 3, 1, 2])
        eyes = eyes.to(device)
        labels = labels.to(device)
        outputs = model(eyes)
        loss = F.mse_loss(outputs, labels)

        total_loss.append(loss.item())
        loop.set_postfix({"Training Loss=": sum(total_loss) / (i + 1)})

        optim.zero_grad()
        loss.backward()
        optim.step()
    return sum(total_loss) / len(total_loss)


def evaluate(model, data_loader, device=DEVICE):
    model.eval()
    loop = tqdm(data_loader)
    total_loss = []
    for (i, (eye, label, _)) in enumerate(loop):
        with torch.no_grad():
            eye = torch.permute(eye, [0, 3, 1, 2])
            eye = eye.to(device)
            label = label.to(device)
            output = model(eye)
            loss = F.mse_loss(output, label)

            total_loss.append(loss.item())
            loop.set_postfix({"Validation Loss=": sum(total_loss) / (i + 1)})
    return sum(total_loss) / len(total_loss)


params = [p for p in eye_key_points_detector.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

min_evaluation_loss = 1e9

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model=eye_key_points_detector, optim=optimizer, data_loader=train_loader, device=DEVICE, epoch_index=epoch)
    lr_scheduler.step()
    print("Evaluation")
    evaluation_loss = evaluate(model=eye_key_points_detector, data_loader=test_loader, device=DEVICE)
    if evaluation_loss < min_evaluation_loss:
        min_evaluation_loss = min(min_evaluation_loss, evaluation_loss)
        # Save model weights after training
        print("Saving Weights...")
        torch.save(eye_key_points_detector.state_dict(), 'weights/resnet_34_2linear_1epoch=13+{}_loss={}.pth'.format(COMPLETED + epoch + 1, round(evaluation_loss, 5)))
