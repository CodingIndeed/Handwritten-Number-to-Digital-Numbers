# Import necessary libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# Define data transformations for input images
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (128,))])

# Load MNIST training and testing datasets
train_data = MNIST('data', train=True, download=True, transform=transforms)
test_data = MNIST('data', train=False, download=True, transform=transforms)

# Create DataLoader for efficiently loading batches of training and testing data
trainloader = DataLoader(train_data, shuffle=True, batch_size=100)
testloader = DataLoader(test_data, shuffle=True, batch_size=100)

# Define a Convolutional Neural Network (CNN) model
class CNN(nn.Module):
    def __init__(self):
        # initialize layers
        super().__init__()
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(10*26*26, 128)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(128, 10)
        self.conv = nn.Conv2d(1, 10, kernel_size=2, stride=1)

    def forward(self, x):
        # define forward pass
        x = self.conv(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flat(x)
        x = self.relu2(self.fc(x))
        x = self.output(x)
        return x

# Instantiate the CNN model, define loss function, and choose an optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(20):
   running_loss = 0.0
   for i, data in enumerate(trainloader, 0):
       inputs, labels = data
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()
       running_loss += loss.item()
   print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

# Save the trained model
torch.save(model.state_dict(), "mnist_cnn.pth")

print('Finished Training')