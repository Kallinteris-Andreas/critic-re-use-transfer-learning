# Note this implamention of the ERB is not memory layout efficent, i would reccomend against using it, though it is an correct
import collections
import random
import torch

experience_replay = collections.namedtuple('experience_replay', 'old_state, actions, reward, new_state, is_terminal')
# note: actions, old_state, new_state are torch.Tensor, reward is float32 scalar, terminal is bool
agent_spaces = collections.namedtuple('agent_spaces', 'observation_space, action_space')

# TODO store rewards and is_terminal as torch.Tensor


class experience_replay_buffer():
    def __init__(self, max_size: int, device="cpu"):
        self.buffer = collections.deque(maxlen=max_size)
        self.torch_device = device

    def add_experience(self, old_state: torch.Tensor, actions: torch.Tensor, reward: float, new_state: torch.Tensor, is_terminal: bool) -> None:
        self.buffer.append(experience_replay(old_state.clone().to("cpu"), actions.clone().to("cpu"), reward, new_state.clone().to("cpu"), is_terminal))

    def sample_batch(self, batch_size: int) -> list[experience_replay]:
        assert len(self.buffer) != 0
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_batch_and_split(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mini_batch = self.sample_batch(batch_size)
        old_state_batch = torch.stack([exp.old_state for exp in mini_batch]).to(self.torch_device)
        new_state_batch = torch.stack([exp.new_state for exp in mini_batch]).to(self.torch_device)
        actions_batch = torch.stack([exp.actions for exp in mini_batch]).to(self.torch_device)
        reward_batch = torch.tensor([[exp.reward for exp in mini_batch]], dtype=torch.float32, device=self.torch_device).transpose(0, 1)
        terminal_batch = torch.tensor([[exp.is_terminal for exp in mini_batch]], device=self.torch_device).transpose(0, 1)
        return old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch

    def __str__(self):
        return str(self.buffer)

    def __repr__(self):
        return self.__str__()
