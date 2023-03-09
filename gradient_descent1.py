import torch
import numpy as np
# toy example
#f = w*x
#f = 2*x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0
#model prediction
def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

#gradient
#MSE error must be calculated
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean() # dl/dw

#print(f'Predcition before training f(5) = ')
learning_rate = 0.001
n_iters = 1000
for epoch in range(n_iters):
    #forward pass
    y_pred = forward(X)
    #loss function
    l = loss(Y, y_pred)
    #gradients
    dw = gradient(X, Y, y_pred)
    #update weights
    w -= learning_rate * dw
    print(w, l)

#print(f'prediction after traiing')

