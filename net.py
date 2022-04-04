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
        self.fc_policy1 = nn.Linear(n, 64)
        self.fc_policy2 = nn.Linear(n, 64)
        
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        value = self.fc_value(x)
        policy1 = self.fc_policy1(x)
        policy2 = self.fc_policy2(x)
                
        return self.sig(value), [self.softmax(policy1), self.softmax(policy2)]