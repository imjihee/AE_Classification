import numpy as np                       
import torch                       
import torchvision                       
from torch import nn                       
from torch.autograd import Variable                       
from torchvision.datasets import MNIST                       
from torchvision.transforms import transforms                       
from torchvision.utils import save_image                       
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, epochs=100, batchSize=128, learningRate=1e-3):
        #We will then want to call the super method:
        super(Autoencoder, self).__init__() #call parent class of Autoencoder
        #super 통해서 module의 init 함수도 호출된다. 즉, Autoencoder, mudule 두 init 함수 호출
        # Encoder Network
        self.encoder = nn.Sequential(nn.Linear(784, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 32),
                                     )
        # Decoder Network
        self.decoder = nn.Sequential(nn.Linear(32, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 784),
                                     nn.Tanh())

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate

        # Data + Data Loaders
        self.imageTransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data = MNIST('./Data', transform=self.imageTransforms,download=True)
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
                image = image.view(image.size(0), -1)
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
        sample = sample.view(sample.size(0), -1)
        sample = Variable(sample)
        return self(sample)

if __name__ == '__main__':
    model = Autoencoder()
    model.trainModel()
    tensor = model.testImage(7777)
    # Reshape
    tensor = torch.reshape(tensor, (28, 28))
    # toImage function
    toImage = torchvision.transforms.ToPILImage()
    # Convert to image
    image = toImage(tensor)
    # Save image
    image.save('After.png')