import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0]) # we have one hot encoded Y
                        # predictions being made are for the 1st entry in the class
                 
Y_predict_good = np.array([0.1, 0.72, 3.1])
print(cross_entropy(Y, Y_predict_good))

loss = nn.CrossEntropyLoss()
#crossentropy() already applies log loss and softmax layer, and here Y is ot one hot encoded
y = torch.tensor([0]) # just put the correct class label and not one hot encoded
# y_good should ahve dimensions n_samples*classes = 1 sample * 3 classes
# y_pred has raw scores not softmax
y_pred_good = torch.tensor([[5.0, 2.3, 0.3]], dtype=float) # class 1 has higher predicted good value
l1 = loss(y_pred_good, y) # loss function always takes predicted value before taking argument as actual value
print(l1.item())

