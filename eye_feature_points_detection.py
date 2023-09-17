import random

from helen_dataset_into_list import data_into_list
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from image_collection import extract_eye
import cv2
# noinspection PyPep8Naming
import albumentations as A
from tqdm import tqdm
from model import get_model
from albumentations.augmentations.geometric.transforms import PadIfNeeded

TRAINING = True
TRAINING_FOR_1_BATCH = False
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
VERBOSE = False
BATCH_SIZE = 8
WEIGHT_PATH = None
WEIGHT_PATH = "weights/4_points_resnet34_linear_1000_epoch=1_loss=0.06207.pth"
NUM_EPOCHS = 100
COMPLETED = 1
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")

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
    A.Resize(height=random.choice([10, 60, 100, 256, 400]), width=256),
    A.PadIfNeeded(min_height=120, min_width=256, position=PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_REPLICATE, p=1),
    A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.TOP_LEFT,
                  border_mode=cv2.BORDER_REPLICATE, p=0.5),
    A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.BOTTOM_RIGHT,
                  border_mode=cv2.BORDER_REPLICATE, p=1),
    A.PadIfNeeded(min_height=300, min_width=300, position=PadIfNeeded.PositionType.TOP_LEFT,
                  border_mode=cv2.BORDER_REPLICATE),
    A.Blur()
], keypoint_params=A.KeypointParams(format='xy'))

test_transforms = A.Compose([
    A.LongestMaxSize(200),
    A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_REPLICATE, position=PadIfNeeded.PositionType.CENTER),
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
        eye_crop, x_bleed, y_bleed, _, _ = self.eye_extractor(image, left=self.left, face_cascade=faceCascade, verbose=VERBOSE)
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
optimizer = torch.optim.Adam(params, lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

min_evaluation_loss = 1e9

if TRAINING:
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model=eye_key_points_detector, optim=optimizer, data_loader=train_loader, device=DEVICE, epoch_index=epoch)
        lr_scheduler.step()
        print("Evaluation")
        evaluation_loss = evaluate(model=eye_key_points_detector, data_loader=test_loader, device=DEVICE)
        if evaluation_loss < min_evaluation_loss:
            min_evaluation_loss = min(min_evaluation_loss, evaluation_loss)

            print("Saving Weights...")
            torch.save(eye_key_points_detector.state_dict(), 'weights/4_points_resnet34_linear_1000_epoch={}_loss={}.pth'.format(COMPLETED + epoch + 1, round(evaluation_loss, 5)))

if TRAINING_FOR_1_BATCH:
    loop = tqdm(range(30))
    iterator = iter(train_loader)
    eyes, labels, images = next(iterator)
    for i in loop:
        eyes_ = torch.permute(eyes, [0, 3, 1, 2])
        eyes_.to(DEVICE)
        labels.to(DEVICE)
        outputs = eye_key_points_detector(eyes_)
        loss = F.mse_loss(outputs, labels)

        loop.set_postfix({"Training Loss=": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots(4, 4)
    for i in range(2):
        for j in range(4):
            eye1 = eyes[i * 4 + j]
            ax[i, j].imshow(eye1.numpy())
    eyes = torch.permute(eyes, [0, 3, 1, 2])
    eyes = eyes.to(DEVICE)
    labels = labels * 256
    output = eye_key_points_detector(eyes).detach() * 256
    for j in range(4):
        label1 = labels[j]
        output1 = output[j]
        for k in range(4):
            ax[0, j].scatter(label1[2 * k], label1[2 * k + 1])
            ax[1, j].scatter(output1[2 * k], output1[2 * k + 1])
    plt.show()

if not TRAINING_FOR_1_BATCH:
    fig, ax = plt.subplots(4, 4)
    iterator = iter(train_loader)
    eyes, labels, images = next(iterator)
    for i in range(2):
        for j in range(4):
            eye1 = eyes[i*4+j]
            ax[i, j].imshow(eye1.numpy())
    eyes = torch.permute(eyes, [0, 3, 1, 2])
    eyes = eyes.to(DEVICE)
    labels = labels * 256
    output = eye_key_points_detector(eyes).detach() * 256
    for j in range(4):
        label1 = labels[j]
        output1 = output[j]
        for k in range(4):
            ax[0, j].scatter(label1[2 * k], label1[2 * k + 1])
            ax[1, j].scatter(output1[2 * k], output1[2 * k + 1])

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    iterator = iter(test_loader)
    eyes, labels, images = next(iterator)
    labels = labels * 256
    for i in range(2):
        for j in range(4):
            eye1 = eyes[i*4+j]
            ax[2 + i, j].imshow(eye1.numpy())
    eyes = torch.permute(eyes, [0, 3, 1, 2])
    eyes = eyes.to(DEVICE)
    output = eye_key_points_detector(eyes).detach() * 256
    for j in range(4):
        output1 = output[j]
        label1 = labels[j]
        for k in range(4):
            ax[2, j].scatter(label1[2 * k], label1[2 * k + 1])
            ax[3, j].scatter(output1[2 * k], output1[2 * k + 1])

    plt.show()
