import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

WEIGHT_PATH = "weights/resnet_18_epoch=23_loss=0.00924.pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class EyeKeyPointsDetector(nn.Module):
    def __init__(self, out_features=40):
        super().__init__()
        self.resnet = models.resnet18()
        self.linear1 = nn.Linear(in_features=1000, out_features=out_features)

    def forward(self, x):
        x = x / 255.0
        # x = self.conv(x)
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        return x


def get_model(weights_path=None):
    model = EyeKeyPointsDetector()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


eye_key_points_detector = get_model(WEIGHT_PATH)
eye_key_points_detector.to(DEVICE)
eye = np.array(Image.open("16547388_1.jpg"))
keypoints_ul = eye_key_points_detector(torch.tensor(eye, dtype=torch.float32).permute([2, 0, 1]).unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0] * 256
print(keypoints_ul)
plt.imshow(eye)
for i in range(20):
    plt.scatter(keypoints_ul[2 * i], keypoints_ul[2 * i + 1])
plt.show()
