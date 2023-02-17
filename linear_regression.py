import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
#data generation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

#model
input_size = n_features
output_size = 1 # we just want one straight line
model = nn.Linear(input_size, output_size)

#loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    #backward pass
    loss.backward()
    #update
    optimizer.step()
    #emptying the gradients
    optimizer.zero_grad()
    print(f'epoch: {epoch+1}, loss = {loss.item()}')

#plotting the graph
predicted = model(X).detach().numpy() #while plotting we dont want the requires_grad attribute to be false, so
                                      #we detatch it
plt.plot(X_numpy, y_numpy, '*')
plt.plot(X_numpy, predicted, 'b')
plt.show()
