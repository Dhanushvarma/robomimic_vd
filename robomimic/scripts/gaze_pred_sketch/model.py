import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal



# Define the MLP model
class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GaussianMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * 2)  # Two times output_dim for mean and std for each dimension

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        mean_std = x.view(-1, 2, 2)  # Reshape the output to separate means and std for each dimension
        mean_x, mean_y = mean_std[:, 0, :], mean_std[:, 1, :]  # Split into mean_x and mean_y
        std_x, std_y = torch.exp(mean_x), torch.exp(mean_y)  # Use exp to ensure positive standard deviations
        return mean_x, std_x, mean_y, std_y




