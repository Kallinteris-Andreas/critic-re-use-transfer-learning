import collections
import random
import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TORCH_DEVICE = "cpu"

experience_replay = collections.namedtuple('experience_replay', 'old_state, actions, reward, new_state, is_terminal')
# note: actions, old_state, new_state are torch.Tensor, reward is float32 scalar, terminal is bool
agent_spaces = collections.namedtuple('agent_spaces', 'observation_space, action_space')

# TODO store rewards and is_terminal as torch.Tensor


class experience_replay_buffer():
    def __init__(self, max_size: int):
        self.buffer = collections.deque(maxlen=max_size)

    def add_experience(self, old_state: torch.Tensor, actions: torch.Tensor, reward: float, new_state: torch.Tensor, is_terminal: bool) -> None:
        self.buffer.append(experience_replay(old_state, actions, reward, new_state, is_terminal))

    def sample_batch(self, batch_size: int) -> list[experience_replay]:
        assert len(self.buffer) != 0
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_batch_and_split(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mini_batch = self.sample_batch(batch_size)
        old_state_batch = torch.stack([exp.old_state for exp in mini_batch])
        new_state_batch = torch.stack([exp.new_state for exp in mini_batch])
        actions_batch = torch.stack([exp.actions for exp in mini_batch])
        reward_batch = torch.tensor([[exp.reward for exp in mini_batch]], dtype=torch.float32, device=TORCH_DEVICE).transpose(0, 1)
        terminal_batch = torch.tensor([[exp.is_terminal for exp in mini_batch]], device=TORCH_DEVICE).transpose(0, 1)
        return old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch

    def sample_batch_and_split_ma(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mini_batch = self.sample_batch(batch_size)
        old_state_global_batch = torch.stack([exp.old_state['global'] for exp in mini_batch])
        # convert old_state_locals_batch from a list of list to a list of torch.tensor
        old_state_locals_batch = [[] for agent_id in range(len(mini_batch[0].old_state['local']))]
        for exp in mini_batch:
            for index, l in enumerate(exp.old_state['local']):
                old_state_locals_batch[index].append(l)
        for agent_id in range(len(old_state_locals_batch)):
            old_state_locals_batch[agent_id] = torch.stack(old_state_locals_batch[agent_id])

        # convert new_state_locals_batch from a list of list to a list of torch.tensor
        new_state_global_batch = torch.stack([exp.new_state['global'] for exp in mini_batch])
        new_state_locals_batch = [[] for agent_id in range(len(mini_batch[0].new_state['local']))]
        for exp in mini_batch:
            for index, l in enumerate(exp.new_state['local']):
                new_state_locals_batch[index].append(l)
        for agent_id in range(len(new_state_locals_batch)):
            new_state_locals_batch[agent_id] = torch.stack(new_state_locals_batch[agent_id])

        actions_global_batch = torch.stack([exp.actions['global'] for exp in mini_batch])
        actions_locals_batch = [[] for agent_id in range(len(mini_batch[0].actions['local']))]
        for exp in mini_batch:
            for index, l in enumerate(exp.actions['local']):
                actions_locals_batch[index].append(l)
        for agent_id in range(len(actions_locals_batch)):
            actions_locals_batch[agent_id] = torch.stack(actions_locals_batch[agent_id])

        rewards_batch = [[] for agent_id in range(len(mini_batch[0].reward))]
        for exp in mini_batch:
            for index, l in enumerate(exp.reward):
                rewards_batch[index].append(l)

        terminals_batch = [[] for agent_id in range(len(mini_batch[0].is_terminal))]
        for exp in mini_batch:
            for index, l in enumerate(exp.is_terminal):
                terminals_batch[index].append(l)

        return old_state_global_batch, old_state_locals_batch, actions_global_batch, actions_locals_batch, rewards_batch, new_state_global_batch, new_state_locals_batch, terminals_batch
