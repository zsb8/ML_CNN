import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)  # Increase the number of channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)  # Increase the number of channels
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=512)  # Increase neurons
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(0.25)  # add dropout

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = self.dropout(x)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x