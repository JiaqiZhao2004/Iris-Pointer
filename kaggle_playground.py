from timm.models.resnet import resnet18

from model import get_model
import torchsummary as summary
model = get_model()



summary.summary(model, input_size=(1, 128, 128), device='cpu')
