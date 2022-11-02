import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class actor(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size, bias=True):
        super().__init__()
        assert action_space_size > 0 and observation_state_size > 0
        self.linear1 = torch.nn.Linear(observation_state_size, 128, bias=bias, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, bias=bias, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, action_space_size, bias=bias, device=TORCH_DEVICE)

    def forward(self, observations):
        assert isinstance(observations, torch.Tensor)
        output = torch.tanh(self.linear1(observations))
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))
        return output


class critic(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size, bias=False):
        super().__init__()
        assert action_space_size > 0 and observation_state_size > 0
        self.linear1 = torch.nn.Linear(action_space_size + observation_state_size, 128, bias=bias, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, bias=bias, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, 1, bias=bias, device=TORCH_DEVICE)

    def forward(self, observations , actions):
        assert isinstance(observations, torch.Tensor) and isinstance(actions, torch.Tensor)
        output = torch.tanh(self.linear1(torch.cat((observations, actions), dim = 1)))
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))
        return output