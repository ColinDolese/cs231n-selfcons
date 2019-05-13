import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision.transforms import ToTensor
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
import parse_data
import torch.nn.functional as F
from layers import SiameseNet, PatchClassifier

class SelfConsistency(nn.Module):

    def __init__(self, exifSize):
        super(SelfConsistency, self).__init__()
        self.siameseNet = SiameseNet(exifSize)
        self.classifier = PatchClassifier(exifSize)

    def forward(self, x):
        x = self.siameseNet(x)
        x = self.classifier(x)
        return x

