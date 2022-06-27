#%%
# ★★ AE_classifier ★★
import pdb
import torch
import numpy as np  
import pandas as pd                     
from torch._C import device                       
import torchvision                       
from torch import nn                       
from torch.autograd import Variable                       
from torchvision.datasets import MNIST
from torchvision.datasets.cifar import CIFAR100                      
from torchvision.transforms import transforms                       
from torchvision.utils import save_image  
from collections import OrderedDict                     
import matplotlib.pyplot as plt
import itertools

class AEclassifier(nn.Module):
    def __init__(self, epochs, batchSize, learningRate):
        #We will then want to call the super method:
        super(AEclassifier, self).__init__() #call parent class of Autoencoder
        # Encoder Network
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                    )
        # Classifier Network
        self.classifier = nn.Sequential(
                                        nn.Flatten(),
                                        nn.Linear(2048, 100)
                                        #nn.ReLU(True),
                                        #nn.Linear(1024, 100)
                                        #nn.LogSoftmax(dim=1)
                                        )

    def forward(self, x):
        x = self.encoder(x)
        #x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    epochs=100
    batchSize = 128
    learningRate = 1e-3

    model = AEclassifier(epochs=epochs, batchSize=batchSize, learningRate=learningRate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)
    #optimizer.zero_grad #zero the parameter graadients
    criterion = nn.CrossEntropyLoss() #NLLLoss()
    print(model)


    #initialize parameter 
    model.zero_grad()

    # Data + Data Loaders
    imageTransforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                    ])
    trainset = CIFAR100('./Data', train=True, transform=imageTransforms,download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=batchSize,
                                            shuffle=True)
    testset = CIFAR100('./Data', train=False, transform=imageTransforms,download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=batchSize,
                                            shuffle=True)

    #for p in model.encoder.parameters() :
    #    p.requires_grad = False

    train_losses = []
    accuracies = [] #per epochs
    for epoch in range(epochs):
        running_loss = 0.0
        #len(trainloader) = 469
        correct=0
        total=0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #inputs = inputs.view(inputs.shape[0], -1)
            #zero the parameter gradients
            optimizer.zero_grad()

            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f ' % (epoch+1, i+1, running_loss/100))
                train_losses.append(running_loss/100)
                running_loss = 0.0

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / len(trainset)
        accuracies.append(accuracy)
        print('Accuracy: %d %%' % (accuracy))
        #print('total : %d and len(trainset) : %d' % (total, len(trainset))) # =60000으로 같다.

    print('Finished Training')

    Path2='./plain_100ep_CF100.pth'
    torch.save(model.state_dict(), Path2)

    plt.subplot(2,1,1)
    plt.plot(train_losses, color='blue')
    plt.title('Training Loss')
    plt.xlabel('epoch')

    plt.subplot(2,1,2)
    plt.plot(accuracies, color='red')
    plt.title('Training Accuracy')
    plt.xlabel('epoch')

    plt.tight_layout()
    plt.show()

    acc=pd.DataFrame(accuracies)
    acc.to_excel(excel_writer='training_acc_plain_100ep_CF100.xlsx')
    lss=pd.DataFrame(train_losses)
    lss.to_excel(excel_writer='training_loss_plain_100ep_CF100.xlsx')
    
    '''
    i=1
    for key, value in state_dict_enc.items(): #type of state_dict_enc: collections.OrderedDict
        print (key, value)
        print("number of param group is : ", i)
        i=i+1
    i=1
    for x in model.encoder.parameters():
        print(x)
        print("##", i)
        i=i+1
    '''
# %%
