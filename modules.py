import torch


class actor(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size, max_action=1, bias=True, device='cpu'):
        super().__init__()
        assert action_space_size > 0 and observation_state_size > 0
        self.linear1 = torch.nn.Linear(observation_state_size, 256, bias=bias, device=device)
        self.linear2 = torch.nn.Linear(256, 256, bias=bias, device=device)
        self.linear3 = torch.nn.Linear(256, action_space_size, bias=bias, device=device)
        self.max_action = max_action

    def forward(self, observations):
        assert isinstance(observations, torch.Tensor)
        output = torch.relu(self.linear1(observations))
        output = torch.relu(self.linear2(output))
        output = torch.tanh(self.linear3(output))
        output = torch.multiply(output, self.max_action)
        return output


class critic(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size, bias=False, device='cpu'):
        super().__init__()
        assert action_space_size > 0 and observation_state_size > 0
        self.linear1 = torch.nn.Linear(action_space_size + observation_state_size, 256, bias=bias, device=device)
        self.linear2 = torch.nn.Linear(256, 256, bias=bias, device=device)
        self.linear3 = torch.nn.Linear(256, 1, bias=bias, device=device)

    def forward(self, observations, actions):
        assert isinstance(observations, torch.Tensor) and isinstance(actions, torch.Tensor)
        output = torch.relu(self.linear1(torch.cat((observations, actions), dim=1)))
        output = torch.relu(self.linear2(output))
        output = (self.linear3(output))
        return output

class twin_critic(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size, bias=False, device='cpu'):
        super().__init__()
        assert action_space_size > 0 and observation_state_size > 0
        self.critic_0 = critic(action_space_size, observation_state_size, bias, device)
        self.critic_1 = critic(action_space_size, observation_state_size, bias, device)

    def forward(self, observations, actions):
        assert isinstance(observations, torch.Tensor) and isinstance(actions, torch.Tensor)
        output_0 = self.critic_0(observations, actions)
        output_1 = self.critic_1(observations, actions)
        return output_0, output_1


# source: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update_target_network(target, source, tau):
    assert tau >= 0 and tau <= 1
    assert isinstance(target, torch.nn.Module) and isinstance(source, torch.nn.Module)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
