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
WEIGHT_PATH = None



SAVING_NAME = "weights/4_points_alexnet_linear_1_sigmoid"
TRAINING = False
TRAINING_FOR_1_BATCH = True
# WEIGHT_PATH = "weights/4_points_resnet18_linear_2_epoch=200_loss=0.04217.pth"
NUM_EPOCHS = 200
COMPLETED = 0


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
VERBOSE = False
BATCH_SIZE = 8
LAST_LOSS = 1e9
faceCascade = cv2.CascadeClassifier("haar_cascade_frontal_face_default.xml")

image_dir = "helen_dataset/img"
annotation_dir = "helen_dataset/annotation"

train_data = data_into_list(image_dir, annotation_dir)
test_data = train_data[int(0.9 * len(train_data)):]
train_data = train_data[:int(0.9 * len(train_data))]

resize_transform = A.Compose([
    # A.Resize(512, 512),
    A.LongestMaxSize(1024),
    A.PadIfNeeded(min_height=1024, min_width=1024, position=PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_REPLICATE),
    # A.SmallestMaxSize(800),
    # A.RandomSizedCrop(min_max_height=(800, 800), width=512, height=512),
], keypoint_params=A.KeypointParams(format='xy'))

train_transforms = A.Compose([
    # A.LongestMaxSize(256),
    # A.Resize(height=random.choice([30, 60, 100, 256]), width=256),
    # A.PadIfNeeded(min_height=120, min_width=256, position=PadIfNeeded.PositionType.CENTER,
    #               border_mode=cv2.BORDER_REPLICATE, p=1),
    # A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.TOP_LEFT,
    #               border_mode=cv2.BORDER_REPLICATE, p=0.5),
    # A.PadIfNeeded(min_height=random.choice([0, 300]), min_width=random.choice([0, 300]), position=PadIfNeeded.PositionType.BOTTOM_RIGHT,
    #               border_mode=cv2.BORDER_REPLICATE, p=1),
    # A.PadIfNeeded(min_height=300, min_width=300, position=PadIfNeeded.PositionType.TOP_LEFT,
    #               border_mode=cv2.BORDER_REPLICATE),
    # A.Blur(),
    A.Resize(height=512, width=512)
], keypoint_params=A.KeypointParams(format='xy'))

test_transforms = A.Compose([
    A.LongestMaxSize(256),
    A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_REPLICATE, position=PadIfNeeded.PositionType.CENTER),
    A.Resize(height=256, width=256)
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
        key_points = self.data[index][1] + self.data[index][2]
        resized = self.resize(image=image, keypoints=key_points)
        image = resized['image']
        key_points = [list(ele) for ele in resized['keypoints']]
        x_min, x_max, y_min, y_max = self.eye_extractor(image, left=self.left, face_cascade=faceCascade, verbose=VERBOSE)
        crop = A.Compose([
            A.Crop(x_min, y_min, x_max, y_max),
            A.Resize(512, 512)
        ], keypoint_params=A.KeypointParams(format='xy'))

        transformed = crop(image=image, keypoints=key_points)
        face_crop = transformed['image']
        key_points = [list(ele) for ele in transformed['keypoints']]

        target = []
        for i in range(len(key_points)):
            target.append(key_points[i][0])
            target.append(key_points[i][1])
        image = torch.tensor(face_crop)
        target = torch.tensor(target) / 256
        target = target.to(torch.float32)
        return image, target

    def __len__(self):
        return len(self.data)


train_dataset = HelenEyeDataset(data=train_data, transform=train_transforms, resize=resize_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = HelenEyeDataset(data=test_data, transform=None, resize=resize_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# train eye model
eye_key_points_detector = get_model(WEIGHT_PATH)
eye_key_points_detector.to(DEVICE)


def train_one_epoch(model, optim, data_loader, epoch_index, device=DEVICE):
    print("Epoch {}".format(COMPLETED + epoch_index + 1))
    model.train()
    loop = tqdm(data_loader)
    total_loss = []
    for (i, (images, targets)) in enumerate(loop):
        images = torch.permute(images, [0, 3, 1, 2])
        images.to(device)
        targets.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, targets)

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
    for (i, (image, label)) in enumerate(loop):
        with torch.no_grad():
            image = torch.permute(image, [0, 3, 1, 2])
            image.to(device)
            label.to(device)
            output = model(image)
            loss = F.mse_loss(output, label)

            total_loss.append(loss.item())
            loop.set_postfix({"Validation Loss=": sum(total_loss) / (i + 1)})
    return sum(total_loss) / len(total_loss)


params = [p for p in eye_key_points_detector.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

min_evaluation_loss = 1e9

if TRAINING:
    train_history = []
    eval_history = []
    evaluation_loss = LAST_LOSS
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model=eye_key_points_detector, optim=optimizer, data_loader=train_loader, device=DEVICE, epoch_index=epoch)
        train_history.append(train_loss)
        lr_scheduler.step()
        print("Evaluation")
        evaluation_loss = evaluate(model=eye_key_points_detector, data_loader=test_loader, device=DEVICE)
        eval_history.append(evaluation_loss)
        if evaluation_loss < min_evaluation_loss:
            min_evaluation_loss = min(min_evaluation_loss, evaluation_loss)
            print("Saving Weights...")
            torch.save(eye_key_points_detector.state_dict(),
                       '{}_epoch={}_loss={}.pth'.format(SAVING_NAME, COMPLETED + epoch + 1, round(evaluation_loss, 5)))
    print("Saving Weights...")
    torch.save(eye_key_points_detector.state_dict(),
               '{}_epoch={}_loss={}.pth'.format(SAVING_NAME, COMPLETED + NUM_EPOCHS, round(evaluation_loss, 5)))

if TRAINING_FOR_1_BATCH:
    history = []
    loop = tqdm(range(100))
    iterator = iter(train_loader)
    images, targets = next(iterator)
    for i in loop:
        images_ = torch.permute(images, [0, 3, 1, 2])
        images_ = images_.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = eye_key_points_detector(images_)
        loss = F.mse_loss(outputs, targets)

        loop.set_postfix({"Training Loss=": loss.item()})
        history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots(4, 4, figsize=(20,20))
    for j in range(4):
        eye1 = images[j]
        ax[0, j].imshow(eye1.numpy())
        ax[1, j].imshow(eye1.numpy())
    images = torch.permute(images, [0, 3, 1, 2])
    images = images.to(DEVICE)
    targets = targets.cpu() * 256
    output = eye_key_points_detector(images).cpu().detach() * 256
    for j in range(4):
        target1 = targets[j]
        output1 = output[j]
        for k in range(8):
            ax[0, j].scatter(target1[2 * k], target1[2 * k + 1])
            ax[1, j].scatter(output1[2 * k], output1[2 * k + 1])

    images, targets = next(iterator)
    for j in range(4):
        image1 = images[j]
        ax[2, j].imshow(image1.numpy())
        ax[3, j].imshow(image1.numpy())
    images = torch.permute(images, [0, 3, 1, 2])
    images = images.to(DEVICE)
    targets = targets.cpu() * 256
    output = eye_key_points_detector(images).cpu().detach() * 256
    for j in range(4):
        target1 = targets[j]
        output1 = output[j]
        for k in range(8):
            ax[2, j].scatter(target1[2 * k], target1[2 * k + 1])
            ax[3, j].scatter(output1[2 * k], output1[2 * k + 1])
    plt.show()

if not TRAINING_FOR_1_BATCH:
    fig, ax = plt.subplots(4, 4, figsize=(20, 20))
    iterator = iter(train_loader)
    images, targets = next(iterator)
    for j in range(4):
        image1 = images[j]
        ax[0, j].imshow(image1.numpy())
        ax[1, j].imshow(image1.numpy())
    images = torch.permute(images, [0, 3, 1, 2])
    images = images.to(DEVICE)
    targets = targets.cpu() * 256
    output = eye_key_points_detector(images).cpu().detach() * 256
    for j in range(4):
        target1 = targets[j]
        output1 = output[j]
        for k in range(8):
            ax[0, j].scatter(target1[2 * k], target1[2 * k + 1])
            ax[1, j].scatter(output1[2 * k], output1[2 * k + 1])

    # test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # iterator = iter(test_loader)
    # images, targets = next(iterator)
    # targets = targets.cpu() * 256
    # for j in range(4):
    #     image1 = images[j]
    #     ax[2, j].imshow(image1.numpy())
    #     ax[3, j].imshow(image1.numpy())
    # images = torch.permute(images, [0, 3, 1, 2])
    # images = images.to(DEVICE)
    # output = eye_key_points_detector(images).cpu().detach() * 256
    # for j in range(4):
    #     output1 = output[j]
    #     target1 = targets[j]
    #     for k in range(8):
    #         ax[2, j].scatter(target1[2 * k], target1[2 * k + 1])
    #         ax[3, j].scatter(output1[2 * k], output1[2 * k + 1])
    #
    plt.show()
