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

class PatchClassifier(nn.Module):
    def __init__(self, exifSize):
        super(PatchClassifier, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(exifSize, 512),
            nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU())

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x


class SiameseNet(nn.Module):
   def __init__(self, exifSize):
      super(SiameseNet, self).__init__()
      self.resnet = models.resnet50(pretrained=True)
      # self.mlp1 = nn.Sequential(
      #       nn.Linear(2000, 1000),
      #       nn.ReLU())
      # self.mlp2 = nn.Sequential(
      #       nn.Linear(1000, 500),
      #       nn.ReLU())
      # self.mlp3 = nn.Sequential(
      #       nn.Linear(500, 250),
      #       nn.ReLU())
      self.mlp4 = nn.Sequential(
            nn.Linear(2000, exifSize),
            nn.ReLU(),
            nn.Sigmoid())

   def forward(self, x):
        res = []
        for x_i in torch.split(x, split_size_or_sections=1):
            x_i = torch.squeeze(x_i)
            x_i = self.resnet(x_i)
            x_i = torch.reshape(x_i, (1,-1))
            res.append(x_i)

        x = torch.stack(res) 
        # x = self.mlp1(x)
        # x = self.mlp2(x)
        # x = self.mlp3(x)
        x = self.mlp4(x)
        return x
