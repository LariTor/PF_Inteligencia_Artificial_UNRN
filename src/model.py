import torch
import torch.nn as nn
import torch.nn.functional as F

class DiabetesModel(nn.Module):
    def __init__(self, input_dim):
        super(DiabetesModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
