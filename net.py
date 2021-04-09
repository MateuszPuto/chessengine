import torch
import torch.nn as nn
import torch.nn.functional as F

n = 2048

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(256, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, n)
        self.fc4 = nn.Linear(n, n)
        
        self.fc_value = nn.Linear(n, 1)
        self.fc_policy = nn.Linear(n, 128)
        
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        value = self.fc_value(x)
        policy = torch.reshape(self.fc_policy(x), (2, 64))
        
        return self.sig(value), [self.softmax(policy[0]), self.softmax(policy[0])]