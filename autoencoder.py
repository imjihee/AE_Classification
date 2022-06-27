#%%%
import numpy as np                       
import torch                       
import torchvision                       
from torch import nn                       
from torch.autograd import Variable                       
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image                      
from torchvision.transforms import transforms                       
from torchvision.utils import save_image                       
import matplotlib.pyplot as plt
import pdb

class Autoencoder(nn.Module):
    def __init__(self, epochs=100, batchSize=128, learningRate=1e-3):
        #We will then want to call the super method:
        super(Autoencoder, self).__init__() #call parent class of Autoencoder
        #super 통해서 module의 init 함수도 호출된다. 즉, Autoencoder, mudule 두 init 함수 호출
        # Encoder Network
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                    )
        # Decoder Network
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 3, kernel_size = 2, stride = 2, padding=0),
                                    nn.Sigmoid()
                                    )

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate

        # Data + Data Loaders
        self.imageTransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data = CIFAR10('./Data', transform=self.imageTransforms,download=True)
        self.dataLoader = torch.utils.data.DataLoader(dataset=self.data,
                                                      batch_size=self.batchSize,
                                                      shuffle=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def trainModel(self):
        for epoch in range(self.epochs):
            for data in self.dataLoader:
                image, _ = data
                #image = image.view(image.size(0), -1)
                image = Variable(image)
                # Predictions
                output = self(image)

                # Calculate Loss
                loss = self.criterion(output, image)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.epochs, loss.data))

    def testImage(self, indexSample):
        sample, _ = self.data[indexSample]
        sample = sample.unsqueeze(0)
        #sample = sample.view(sample.size(0), -1) #3,32,32 -> 3,1024로 flatten해주는 메서드
        sample = Variable(sample)
        return self(sample)

if __name__ == '__main__':
    FILE = "param_temp.pth"
    model = Autoencoder()
    print(model)

    model.trainModel()

    #save model
    torch.save(model.state_dict(), FILE)

    idx = 10
    idx1 = 200

    tensor = model.testImage(idx)
    tensor1 = model.testImage(idx1)

    # Reshape
    tensor = torch.reshape(tensor, (3, 32, 32))
    tensor1 = torch.reshape(tensor1, (3, 32, 32))
    '''
    # toImage function
    toImage = torchvision.transforms.ToPILImage()
    # Convert to image
    image = toImage(tensor)
    image1 = toImage(tensor1)
    # Save image
    image.save('After1.png')
    image1.save('After11.png')
    
    sample, _ = model.data[idx]
    image_original = toImage(sample)
    #image_original.save('Before1.png')

    sample1, _ = model.data[idx1]
    image_original1 = toImage(sample1)
    image_original1.save('Before11.png')
    '''
# %%
