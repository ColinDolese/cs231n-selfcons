import sys
import math
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
        #xInd = random.randint(0, 2048-128)
        #yInd = random.randint(0, 2048-128)
        #x = x[:, xInd:xInd+128, yInd:yInd+128]
        #x = x.narrow(1, xInd, 128)
        #x = x.narrow(2, yInd, 128)
        return (x, target, index)

class Columbia(torch.utils.data.Dataset): 
    def __init__(self): 
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        #self.data = dset.ImageFolder("./Columbia", transform=transform)
        self.data = dset.ImageFolder("../../colin/cs231n-selfcons/Columbia", transform=transform)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index): 
        target = self.data[index][1]
        x = self.data[index][0]
        return (x, target, index)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
def check_accuracy_train(loader, model):
    num_correct_exif = 0
    num_samples_exif = 0
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        print("Checking Accuracy")
        for t, (x, y, i) in enumerate(loader):

            N, C, H, W = x.shape

            X = torch.zeros(N, 2, C, 128, 128)
            Y = torch.zeros(N, 2)

            for b in range(N):
                xInd1 = random.randint(0, 2048-128)
                yInd1 = random.randint(0, 2048-128)

                xInd2 = random.randint(0, 2048-128)
                yInd2 = random.randint(0, 2048-128)

                patch1 = x[b, :, xInd1:xInd1+128, yInd1:yInd1+128]
                X[b, 0, :, :, :] = patch1
                Y[b, 0] = y[b]
                #copy = bool(random.randint(0, 1))


                if b < N // 2:
                    patch2 = x[b, :, xInd2:xInd2+128, yInd2:yInd2+128]
                    Y[b, 1] = y[b]

                else:
                    patch2Ind = random.choice([ind for ind in range(N-1) if ind != b])
                    #patch2Ind = random.randint(0, N-1)
                    patch2 = x[patch2Ind, :, xInd2:xInd2+128, yInd2:yInd2+128]
                    Y[b, 1] = y[patch2Ind]

                X[b, 1, :, :, :] = patch2

            x = X.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = Y.to(device=device, dtype=torch.float)

            # halfN = N // 2

            # if N % 2 != 0:
            #     continue

            # copyInd = random.randint(0,(halfN)-1)

            # xCopy = x[copyInd:copyInd+halfN, :, :, :]
            # yCopy = y[copyInd:copyInd+halfN]

            # x = torch.reshape(x, (N//2, 2, C, H, W))
            # y = torch.reshape(y, (N//2, 2))

            # xFinal = torch.zeros((N//2 + halfN, 2, C, H, W))
            # xFinal[:N//2, :, :, :] = x

            # for h in range(halfN):
            #     xFinal[N//2+h, 0, :, :, :] = xCopy[h]
            #     xFinal[N//2+h, 1, :, :, :] = xCopy[h]

            # yFinal = torch.zeros((N//2 + halfN, 2))
            # yFinal[:N//2, :] = y
            # for h in range(halfN):
            #     yFinal[N//2+h, 0] = yCopy[h]
            #     yFinal[N//2+h, 1] = yCopy[h]

            # x = xFinal
            # y = yFinal

            # x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            # y = y.to(device=device, dtype=torch.float)

            classScores, exifScores = model(x)

            exifTarget = []
            for pair in torch.split(y, split_size_or_sections=1):
            	exifVec = parse_data.exif_vec(pair[0][0], pair[0][1])
            	exifTarget.append(exifVec)
            exifTarget = torch.stack(exifTarget).to(device=device, dtype=torch.float)

            y = y.permute(1,0)

            exifPreds = exifScores.round()
            exifPreds = exifPreds.reshape(exifPreds.size(0), -1)

            num_correct_exif += (exifPreds == exifTarget).sum()
            num_samples_exif += (exifPreds.size(0) * exifPreds.size(1))

            if num_samples_exif == 0.0:
                print(exifPreds.size(0))
                print(exifPreds.size(1))

            classPreds = classScores.round()
            classPreds = classPreds.reshape(exifPreds.size(0))
            target = y[0] == y[1]
            target = target.to(device=device, dtype=torch.float)
            num_correct += (classPreds == target).sum()
            num_samples += classPreds.size(0)

        exifAcc = 0.0
        classAcc = 0.0

        if num_samples_exif > 0.0:
            exifAcc = float(num_correct_exif) / num_samples_exif

        if num_samples > 0.0:
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
    lossFunc = nn.BCELoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print("----------- Starting  Epoch " + str(e) + " -----------")
        for t, (x, y, i) in enumerate(loader_train):
            print('Iteration %d' % (t))
            model.train()  # put model to training mode
            N, C, H, W = x.shape

            X = torch.zeros(N, 2, C, 128, 128)
            Y = torch.zeros(N, 2)

            for b in range(N):
                xInd1 = random.randint(0, 2048-128)
                yInd1 = random.randint(0, 2048-128)

                xInd2 = random.randint(0, 2048-128)
                yInd2 = random.randint(0, 2048-128)

                patch1 = x[b, :, xInd1:xInd1+128, yInd1:yInd1+128]
                X[b, 0, :, :, :] = patch1
                Y[b, 0] = y[b]
                #copy = bool(random.randint(0, 1))


                if b < N // 2:
                    patch2 = x[b, :, xInd2:xInd2+128, yInd2:yInd2+128]
                    Y[b, 1] = y[b]

                else:
                    patch2Ind = random.choice([ind for ind in range(N-1) if ind != b])
                    #patch2Ind = random.randint(0, N-1)

                X[b, 1, :, :, :] = patch2

            x = X.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = Y.to(device=device, dtype=torch.float)

            # halfN = N // 2

            # if N % 2 != 0:
            #     continue

            # copyInd = random.randint(0,(halfN)-1)

            # xCopy = x[copyInd:copyInd+halfN, :, :, :]
            # yCopy = y[copyInd:copyInd+halfN]

            # x = torch.reshape(x, (N//2, 2, C, H, W))
            # y = torch.reshape(y, (N//2, 2))

            # xFinal = torch.zeros((N//2 + halfN, 2, C, H, W))
            # xFinal[:N//2, :, :, :] = x

            # for h in range(halfN):
            #     xFinal[N//2+h, 0, :, :, :] = xCopy[h]
            #     xFinal[N//2+h, 1, :, :, :] = xCopy[h]

            # yFinal = torch.zeros((N//2 + halfN, 2))
            # yFinal[:N//2, :] = y
            # for h in range(halfN):
            #     yFinal[N//2+h, 0] = yCopy[h]
            #     yFinal[N//2+h, 1] = yCopy[h]

            # x = xFinal
            # y = yFinal

            # x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            # y = y.to(device=device, dtype=torch.float)

            classScores, exifScores = model(x)
            exifTarget = []
            for pair in torch.split(y, split_size_or_sections=1):
                exifVec = parse_data.exif_vec(pair[0][0], pair[0][1])
                exifTarget.append(exifVec)

            y = y.permute(1,0)

            exifScores = torch.squeeze(exifScores, dim=1)
            exifTarget = torch.stack(exifTarget).to(device=device, dtype=torch.float)

            exifLoss = lossFunc(exifScores, exifTarget)

            classTarget = y[0] == y[1]
            classTarget = classTarget.to(device=device, dtype=torch.float)

            classScores = torch.reshape(classScores, (N,))

            print(classScores)

            classLoss = lossFunc(classScores, classTarget)
            totalLoss = sum([exifLoss, classLoss])

            print("Exif Loss: " + str(exifLoss))
            print("Class Loss: " + str(classLoss))
            print("Total Loss: " + str(totalLoss))
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            if t % 100 == 0 and t > 0:
                check_accuracy_train(loader_val, model)
                print()

        print("Saving model at epoch: " + str(e))
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, 'model.pt')

def test_columbia(model, loader_test, numPatches):

    tp = 0
    tn = 0
    fn = 0
    fp = 0

    model.eval()  # set model to evaluation mode
    for x, y, index in loader_test:

        print("Testing image " + str(index + 1))

        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        _, xDim, yDim = x.shape

        xSize = 128 - numPatches
        ySize = 128 - numPatches
        tamper = False
        for i in range(numPatches):

            if tamper:
                break

            for j in range(numPatches):
                scores = torch.zeros((numPatches,numPatches))
                scores[i,j] = 1.0           
                curX = torch.zeros((3,128,128))
                centerX = numPatches // 2
                centerY = numPatches // 2
                curX[:,centerX:centerX + xSize-1, centerY:centerY+ySize-1] = x[:,i:i+xSize-1, j:j+ySize-1]

                for k in range(numPatches):

                    for l in range(numPatches):

                        if k==i and l == j:
                            continue

                        testX = torch.zeros((3,128,128))
                        testX[:,centerX:centerX + xSize-1, centerY:centerY+ySize-1] = x[:,k:k+xSize-1, l:l+ySize-1]
                        pair = torch.stack([curX, testX]).to(device=device, dtype=torch.float)
                        pair = torch.unsqueeze(pair, 0)

                        classScores, exifScores = model(pair)
                        if classScores[0,0,0] < 0.7:
                            print(classScores[0,0,0])
                        if classScores[0,0,0] < 0.5:
                            scores[k,l] = 0.0
                        else :
                            scores[k,l] = 1.0

                if torch.sum(scores) < (numPatches*numPatches // 2):
                    tamper = True
                    break

        truth = (y == 1)

        print("Image " + str(index + 1) + " classified as tamper = " + str(tamper))
        print("Image " + str(index + 1) + " is actually tamper = " + str(truth))

        if tamper and truth:
            tp += 1

        if not tamper and not truth:
            tn += 1

        if not tamper and truth:
            fn += 1

        if tamper and not truth:
            fp += 1

    print("tp: " + str(tp))
    print("tn: " + str(tn))
    print("fp: " + str(fp))
    print("fn: " + str(fn))  

    f1 = 0.0
    f1_denom = (2.0*tp + fn + fp)

    if f1_denom > 0.0:
        f1 = 2.0*tp / f1_denom     

    mcc = 0.0
    mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if mcc_denom > 0.0:
        mcc = (tp*tn - fp*fn) / mcc_denom

    print("F1 score: " + str(f1))
    print("MCC score: " + str(mcc))



def main():


    minOccur = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    numPatches = int(sys.argv[4])
    loadTrainModel = bool(int(sys.argv[5]))
    testBestModel = bool(int(sys.argv[6]))

    numImages, numAttributes = parse_data.getAttributes(minOccur)

    lr = 1e-4
    model = SelfConsistency(numAttributes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not testBestModel:

        if loadTrainModel:
            checkpoint = torch.load('model.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print("Loaded model trained with " + str(epoch) + " epochs")


        # numImages = 7477
        # numAttributes = 37

        valStart = max((numImages // 2) + 1, numImages - 100)


        img_data_train = ExifTrainDataset()

        print("----------- Loading Data -----------")
        loader_train = DataLoader(img_data_train, batch_size=batch_size, 
                                  sampler=sampler.SubsetRandomSampler(range(valStart-1)))


        loader_val = DataLoader(img_data_train, batch_size=batch_size, 
                                  sampler=sampler.SubsetRandomSampler(range(valStart, numImages-1)))



        print("----------- Finished Loading Data -----------")

        train(model, optimizer, loader_train, loader_val, epochs=epochs)

        print("----------- Testing -----------")

        print("Saving best model")
        torch.save(model, "model_best.pt")


    else:

        model = torch.load("model_best.pt")
        img_columbia = Columbia()
        test_columbia(model, img_columbia, numPatches)

if __name__ == '__main__':
    main()
