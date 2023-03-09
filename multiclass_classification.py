import torch
import torch.nn as nn

# multiclass problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet1, self).__init__()
        self.linear1= nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU
        # number of classes we want to classify stuff into
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # No softmax at the output layer
        return out
    
model = NeuralNet1(input_size=32, hidden_size=28, num_classes=4)
# for calculating the loss function
criterion = nn.CrossEntropyLoss()
