import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size=8, num_hidden_layers=3, output_size=2):
        super(NeuralNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class HybridNeuralNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=8, num_hidden_layers=3, output_size=2):
        super(HybridNeuralNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        fdm = x[:, 2:]  # Extract Sn_fdm_real and Sn_fdm_imag
        correction = self.network(x)
        return fdm + correction
