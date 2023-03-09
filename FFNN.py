# Feed forward neural network for MNIST dataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# access GPU if possible else only CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperarameters
input_size = 784 #MNIST image_size is 28*28
hidden_size = 100
num_classes = 10 # from [0-9]
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# loading the dataset, root='./data' means load all the data in the root directory
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), 
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Analysing one batch of the training dataset
examples = iter(train_loader)
# unpacking the data into samples and labels
samples, labels = next(examples)
print(samples.shape, labels.shape)

# Just for visualizing the data
for i in range(6): # i is the index
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()

class neuralnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(neuralnet, self).__init__()
        # creating the layers
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No softmax at the output layer
        return out

model = neuralnet(input_size, hidden_size, num_classes)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # unpacking the data
        # originally unpacking made the data size as [100, 1, 28, 28]
        # we need to make it [100, 784]
        images = images.reshape(-1, 28*28).to(device) # we nake it a 28*28 image
        labels = labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels) # predcited values, labels
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch + 1}, loss {loss.item()}')

# Testing and evaluation of the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        # _, means we dont actually need that vaue
        # value, index
        _, predictions = torch.max(output, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy = {accuracy}')