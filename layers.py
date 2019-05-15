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

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class PatchClassifier(nn.Module):
    def __init__(self, exifSize):
        super(PatchClassifier, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(exifSize, 512),
            nn.ReLU())

        self.mlp1.apply(init_weights)
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.ReLU())

        self.mlp2.apply(init_weights)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x


class SiameseNet(nn.Module):
   def __init__(self, exifSize):
      super(SiameseNet, self).__init__()
      self.resnet = models.resnet50(pretrained=True)
      self.mlp1 = nn.Sequential(
            nn.Linear(2000, 1048),
            nn.ReLU())

      self.mlp1.apply(init_weights)
      self.mlp2 = nn.Sequential(
            nn.Linear(1048, 624),
            nn.ReLU())

      self.mlp2.apply(init_weights)
      self.mlp3 = nn.Sequential(
            nn.Linear(624, 312),
            nn.ReLU())

      self.mlp3.apply(init_weights)
      self.mlp4 = nn.Sequential(
            nn.Linear(312, exifSize),
            nn.ReLU(),
            nn.Sigmoid())
      self.mlp4.apply(init_weights)


   def forward(self, x):
        res = []
        for x_i in torch.split(x, split_size_or_sections=1):
            x_i = torch.squeeze(x_i)
            x_i = self.resnet(x_i)
            x_i = torch.reshape(x_i, (1,-1))
            res.append(x_i)

        x = torch.stack(res) 
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x
