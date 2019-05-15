import sys
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
import random
from model import SelfConsistency
from layers import SiameseNet, PatchClassifier



import numpy as np

class ExifTrainDataset(torch.utils.data.Dataset): 
    def __init__(self): 
        transform = T.Compose([
                T.CenterCrop(2048),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        self.data = dset.ImageFolder("./FlickrCentral", transform=transform)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index): 
        #target = parse_data.get_attribute_vec(index)
        #y = torch.from_numpy(target).float().to(self.device)
        target = self.data[index][1]
        x = self.data[index][0]
        xInd = random.randint(0, 2048-128)
        yInd = random.randint(0, 2048-128)
        x = x.narrow(1, xInd, 128)
        x = x.narrow(2, yInd, 128)
        return (x, target, index)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
def check_accuracy(loader, model):
    num_correct_exif = 0
    num_samples_exif = 0
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y, i) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)

            N, C, H, W = x.shape

            if N % 2 != 0:
                continue

            x = torch.reshape(x, (N//2, 2, C, H, W))
            y = torch.reshape(y, (N//2, 2))

            classScores, exifScores = model(x)

            exifTarget = []
            for pair in torch.split(y, split_size_or_sections=1):
            	exifVec = parse_data.exif_vec(pair[0][0], pair[0][1])
            	exifTarget.append(exifVec)
            exifTarget = torch.stack(exifTarget).to(device=device, dtype=torch.float)

            y = y.permute(1,0)
            
            exifPreds = exifScores.round()
            exifPreds = exifPreds.reshape(exifPreds.size(0), -1)
            print(exifTarget.shape)
            print(exifPreds.shape)
            num_correct_exif += (exifPreds == exifTarget).sum()
            num_samples_exif += (exifPreds.size(0) * exifPreds.size(1))


            classPreds = classScores.round()
            classPreds = classPreds.reshape(exifPreds.size(0))
            target = y[0] == y[1]
            target = target.to(device=device, dtype=torch.float)
            num_correct += (classPreds == target).sum()
            num_samples += classPreds.size(0)

        exifAcc = float(num_correct_exif) / num_samples_exif
        classAcc = float(num_correct) / num_samples
        print('Got %d / %d exifs correct (%.2f)' % (num_correct_exif, num_samples_exif, 100 * exifAcc))
        print('Got %d / %d classes correct (%.2f)' % (num_correct, num_samples, 100 * classAcc))


def train(model, optimizer, loader_train, loader_val, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    lossFunc = nn.MultiLabelSoftMarginLoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
    	print("----------- Starting  Epoch " + str(e) + " -----------")
    	for t, (x, y, i) in enumerate(loader_train):
            print('Iteration %d' % (t))
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.float)
            N, C, H, W = x.shape

            if N % 2 != 0:
                continue

            x = torch.reshape(x, (N//2, 2, C, H, W))
            y = torch.reshape(y, (N//2, 2))
            classScores, exifScores = model(x)

            exifTarget = []
            for pair in torch.split(y, split_size_or_sections=1):
            	exifVec = parse_data.exif_vec(pair[0][0], pair[0][1])
            	exifTarget.append(exifVec)

            y = y.permute(1,0)
            exifTarget = torch.stack(exifTarget).to(device=device, dtype=torch.float)
            exifLoss = lossFunc(exifScores, exifTarget)
            classTarget = y[0] == y[1]
            classTarget = classTarget.to(device=device, dtype=torch.float)
            classLoss = lossFunc(classScores, classTarget)

            totalLoss = sum([exifLoss, classLoss])

            print("Exif Loss: " + str(exifLoss))
            print("Class Loss: " + str(classLoss))
            print("Total Loss: " + str(totalLoss))

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            #total_loss = sum([exifLoss, classLoss])
            totalLoss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 10 == 0:
                check_accuracy(loader_val, model)
                print()



def main():

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])

    # print("----------- Starting Attribute Search -----------")
    numImages, numAttributes = parse_data.getAttributes()
    # print("----------- Finished Attribute Search -----------")

    # numImages = 7477
    # numAttributes = 37

    trainEnd = numImages // 2

    valStart = (numImages // 2) + 1
    valEnd = valStart + (numImages // 4)

    testStart = valEnd + 1
    testEnd = numImages - 1

    img_data = ExifTrainDataset()

    print("----------- Loading Data -----------")
    loader_train = DataLoader(img_data, batch_size=batch_size, 
                              sampler=sampler.SubsetRandomSampler(range(trainEnd)))


    loader_val = DataLoader(img_data, batch_size=batch_size, 
                              sampler=sampler.SubsetRandomSampler(range(valStart, valEnd)))

    loader_test = DataLoader(img_data, batch_size=batch_size, 
                            sampler=sampler.SubsetRandomSampler(range(testStart, testEnd)))

    print("----------- Finished Loading Data -----------")
    model = SelfConsistency(numAttributes)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(model, optimizer, loader_train, loader_val, epochs=epochs)
    print("----------- Testing -----------")
    check_accuracy(loader_test, model)

if __name__ == '__main__':
    main()
