import torch
import torch.nn as nn
from torchvision import models


class EyeKeyPointsDetector(nn.Module):
    def __init__(self, out_features=40):
        super().__init__()
        self.resnet = models.resnet34()
        self.linear1 = nn.LazyLinear(out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=out_features)

    def forward(self, x):
        x = x / 255.0
        # x = self.conv(x)
        x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def get_model(weights_path=None):
    model = EyeKeyPointsDetector()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model
