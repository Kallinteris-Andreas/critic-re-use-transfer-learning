import collections
import random
import torch

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experience_replay = collections.namedtuple('Experience', 'old_state, actions, reward, new_state, terminal1')
#note: actions, old_state, new_state are torch.Tensor reward is float32 scalar, terminal is bool
agent_spaces = collections.namedtuple('agent_def', 'observation_space, action_space')

   #TODO store rewards are torch.Tensor
class experience_replay_buffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = collections.deque(maxlen=max_size)
    def add_experience(self, old_state, actions, reward, new_state, is_terminal):
        assert len(self.buffer) <= self.max_size
        if len(self.buffer) == self.max_size-1:
            print("filled the ERB")
        self.buffer.append(experience_replay(old_state, actions, reward, new_state, is_terminal))

    def sample_batch(self, batch_size):
        assert len(self.buffer) != 0
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_batch_and_split(self, batch_size):
        mini_batch = self.sample_batch(batch_size) 
        old_state_batch = torch.stack([exp.old_state for exp in mini_batch])
        new_state_batch = torch.stack([exp.new_state for exp in mini_batch])
        actions_batch = torch.stack([exp.actions for exp in mini_batch])
        reward_batch = torch.tensor([[exp.reward for exp in mini_batch]], dtype=torch.float32, device=TORCH_DEVICE).transpose(0,1)
        return old_state_batch, actions_batch, reward_batch, new_state_batch