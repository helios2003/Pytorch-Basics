import torch
import numpy as np

#f = w*x
#f = 2*x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
#model prediction
def forward(x):
    return w * x
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

#gradient
#MSE error must be calculated


#print(f'Predcition before training f(5) = ')
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    #forward pass
    y_pred = forward(X)
    #loss function
    l = loss(Y, y_pred)
    #gradients
    l.backward()
    #update weights
    with torch.no_grad():
        w -= learning_rate * w.grad #to make computations faster by setting require_grad flag to False
    w.grad.zero_()
    print(w, l)

#print(f'prediction after traiing')


