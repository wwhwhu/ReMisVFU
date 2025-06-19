import torch.nn as nn
import torch.nn.functional as F
import torch
class FirstNet(nn.Module):
    """
    Client-side feature extractor: two Conv2d + ReLU + MaxPool layers.
    Input: [batch, 1, 28, 14]  (left or right half of MNIST image)
    Output: [batch, 64, 7, 7]
    """
    def __init__(self, in_channels: int = 1):
        super(FirstNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # → [batch,32,28,14]
        self.pool = nn.MaxPool2d(2, 2)                           # → [batch,32,14,7]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # → [batch,64,14,7]
        # second pool: [batch,64,7,3] but since input width=14 after splitting, width=7 after first pool and 7 after second conv, pooling yields width=3 or 3.5 → floor 3
        self.pool2 = nn.MaxPool2d((2, 2))                        # → [batch,64,7,3]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x


class SecondNet(nn.Module):
    """
    Server-side classifier: takes flattened concatenated features from two clients.
    Input: features_A ([batch,64,7,3]) and features_B ([batch,64,7,3])
    Output: logits ([batch,10])
    """
    def __init__(self, output_dim=10, client_num=2, in_dim = 1):
        super(SecondNet, self).__init__()
        self.client_num = client_num
        flattened_dim = 64 * 7 * 2 * self.client_num
        if in_dim != 1:
            if client_num == 3:
                flattened_dim = 3584
        # flattened_dim = 2560
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x_list):
        # Flatten all feature maps
        if not isinstance(x_list, list):
            raise ValueError("x_list must be a list of tensors")
        B = x_list[0].size(0)  # Batch size
        flattened_features = [x.view(B, -1) for x in x_list]
        x = torch.cat(flattened_features, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
