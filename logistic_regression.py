import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#preparing the dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
If we will use the fit method on our test data too, we will compute a new mean 
and variance that is a new scale for each feature and will let our model learn 
about our test data too. Thus, what we want to keep as a surprise is no longer 
unknown to our model and we will not get a good estimate of how our model is 
performing on the test (unseen) data
"""
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) #built in function to reshape data
y_test = y_test.view(y_test.shape[0], 1)

#setting up the model
# f = w*x + b
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features): #it initialises the class with n input features
                                          #which give 1 output class i.e predicted probability of belonging to the positive class
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(n_features)
#loss 
lr = 0.01
criterion = nn.BCELoss()
#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr)
#training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    #backward pass
    loss.backward()
    #updating weights
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()
    print(f'epoch: {epoch+1}, loss = {loss.item()}')

with torch.no_grad(): #so that it isn't the part of any computational graph
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy:{accuracy}')




