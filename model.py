import torch
import torch.nn as nn
from timm.models import hrnet
from torchvision import models


class EyeKeyPointsDetector(nn.Module):
    def __init__(self, out_features=80):
        super().__init__()
        self.resnet = models.resnet34()
        self.linear_end = nn.LazyLinear(out_features=out_features)

    def forward(self, x):
        x = x / 255.0
        x = self.resnet(x)
        x = self.linear_end(x)
        return x


def get_model(weights_path=None):
    model = EyeKeyPointsDetector()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model
