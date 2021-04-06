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
        self.fc_policy = nn.Linear(n, 4096)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        ## Temp.
        ## This needs to be passed to sigmoid, softmax etc.
        value = self.fc_value(x)
        policy = self.fc_policy(x)
        
        return value, policy