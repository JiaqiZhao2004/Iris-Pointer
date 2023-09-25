import torch
import torch.nn as nn
from timm.models import hrnet
from torchvision import models


class EyeCornerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34()
        self.linear_end = nn.LazyLinear(out_features=16)
        self.linear1 = nn.LazyLinear(out_features=16)

    def forward(self, x):
        x = x / 255.0
        x = self.resnet(x)
        x = self.linear_end(x)
        return x


def get_model(weights_path=None):
    head = EyeCornerDetector()
    if weights_path:
        state_dict = torch.load(weights_path)
        head.load_state_dict(state_dict)
        print("Loading Weights...")

    for param in head.parameters():
        param.requires_grad = False

    class EyeContourDetector(nn.Module):
        def __init__(self, head_=head):
            super().__init__()
            self.head = head_
            self.linear_end = nn.Linear(in_features=16, out_features=80)

        def forward(self, x):
            x = self.head(x)
            print(x[0])
            x = self.linear_end(x)
            return x

    model = EyeContourDetector()
    return model
