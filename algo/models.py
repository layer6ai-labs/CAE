import torch.nn as nn


class CNetworkCont(nn.Module):
    def __init__(self, d_obs, d_goal, d_actions, hidden_layer_sizes):
        super(CNetworkCont, self).__init__()

        layers = []
        prev_layer_size = d_obs + d_goal + d_actions + 1  # Input is [state, goal, action, horizon]
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=prev_layer_size, out_features=h_size))
            prev_layer_size = h_size
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=prev_layer_size, out_features=1))
        layers.append(nn.Sigmoid())  # Ensure these are between 0 and 1

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)


class ActorNetwork(nn.Module):
    def __init__(self, d_obs, d_goal, d_actions, hidden_layer_sizes):
        super(ActorNetwork, self).__init__()
        layers = []
        prev_layer_size = d_obs + d_goal + 1  # Input is [state, goal, horizon]
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=prev_layer_size, out_features=h_size))
            prev_layer_size = h_size
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=prev_layer_size, out_features=d_actions))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)

class CNetworkDisc(nn.Module):
    def __init__(self, d_obs, d_goal, n_actions, hidden_layer_sizes):
        super(CNetworkDisc, self).__init__()

        layers = []
        prev_layer_size = d_obs + d_goal + 1  # Input is [state, goal, horizon]
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=prev_layer_size, out_features=h_size))
            prev_layer_size = h_size
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=prev_layer_size, out_features=n_actions))

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.net(x)