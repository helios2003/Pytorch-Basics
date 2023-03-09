# CNN for CIFAR10 dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# access GPU if possible else only CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperarameters
num_epochs = 100
batch_size = 100
learning_rate = 0.01

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# loading the dataset, root='./data' means load all the data in the root directory
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Classes to be separated into
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 3: RGB channels (input size)
        # 6: Initialize a random number (output size)
        # 5: kernel size
        self.conv1 = nn.Conv2d(3, 6, 5) 
        # 2: max pooling kernel size
        # 2: stride length
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Flattening the neural network
        # fc: fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 10 : because the dataset consists of 10 classes
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flattedned tensor for an ANN task
        x = x.view(-1, 16*5*5) 
        #x = x.view(-1, 6*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

# loading the model
model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss() # final output softmax already included here
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch + 1}, loss {loss.item()}')

# Testing and evaluation of the model
with torch.no_grad():
    n_samples = 0
    n_correct = 0 
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        # _, means we dont actually need that vaue
        # value, index
        _, predictions = torch.max(output, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'accuracy = {accuracy}')
    