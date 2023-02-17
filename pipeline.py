# 1> Design a model
# 2> Construct loss and optimizer
# 3> Training loop(FP, BP, updation)
import torch
import torch.nn as nn
import numpy as np

#f = w*x
#f = 2*x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)
# we need not initialize the weights otherwise the pytorch model already knows about it from before
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size) #in and out features are the params for which it doesnt accept a 1D array
#model prediction

#print(f'Predcition before training f(5) = ')
learning_rate = 0.01
n_iters = 10
loss = nn.MSELoss() # no calculating loss manually
optimizer = torch.optim.SGD(model.parameters(), learning_rate) # params of SGD should be passed as a list of weights and the LR
for epoch in range(n_iters):
    #forward pass
    y_pred = model(X)
    #loss function
    l = loss(Y, y_pred)

    #gradients
    l.backward()

    #update weights
    optimizer.step() #no need to calculate manually
    optimizer.zero_grad() #for setting gradients to zero after each iteration
    [w, b] = model.parameters()
    print(w[0][0].item(), model(X_test).item())

    


