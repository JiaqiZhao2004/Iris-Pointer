from helen_dataset_into_list import data_into_list
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from image_collection import extract_eye
import cv2
from main import CAMERA_MODE, faceCascade, eyeCascade


image_dir = "helen_dataset/img"
annotation_dir = "helen_dataset/annotation"

train_data = data_into_list(image_dir, annotation_dir)
test_data = train_data[int(0.9*len(train_data)):]
train_data = train_data[:int(0.9*len(train_data))]



# train left eye model

# train right eye model
