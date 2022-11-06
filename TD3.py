import torch
import yaml
import copy
from ERB import *
from modules import *

class TD3_model():
    def __init__(self, num_actions, num_states, min_action, max_action, yaml_config):
        assert num_actions > 0 and num_states > 0 and min_action < max_action
        self.num_actions = num_actions
        self.min_action = min_action
        self.max_action = max_action
        self.total_step_iterations = 0

        self.discount_rate = yaml_config['TD3']['gamma']
        self.target_update_rate = yaml_config['TD3']['tau']
        self.mini_batch_size = yaml_config['TD3']['N']
        self.noise_exploration_standard_deviation = yaml_config['TD3']['sigma_explore']
        self.noise_policy_standard_deviation = yaml_config['TD3']['sigma_policy']
        self.noise_policy_clip = yaml_config['TD3']['noise_policy_clip']
        self.policy_update_frequency = yaml_config['TD3']['d']
        
        self.actor = actor(num_actions, num_states, max_action, bias=True, device=TORCH_DEVICE) # mu
        self.target_actor = copy.deepcopy(self.actor) # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), yaml_config['TD3']['optimizer_gamma'])
        self.critics= twin_critic(num_actions, num_states, bias=False, device=TORCH_DEVICE) # q0-1
        self.target_critics = copy.deepcopy(self.critics) # q0-1'
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), yaml_config['TD3']['optimizer_gamma'])

        self.erb = experience_replay_buffer(yaml_config['TD3']['experience_replay_buffer_size'])

    def query_actor(self, state, add_noise=True):
        if add_noise:
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE)*self.noise_exploration_standard_deviation, min = self.min_action, max = self.max_action)
        else:
            return self.actor(state)


    def train_model_step(self):
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch = self.erb.sample_batch_and_split(self.mini_batch_size)


        #update critic
        with torch.no_grad():
            #select target action
            target_policy_noise = (torch.randn(self.mini_batch_size, 1, device=TORCH_DEVICE)*self.noise_policy_standard_deviation).clamp(min= -self.noise_policy_clip, max=self.noise_policy_clip)
            target_actions_batch = torch.clamp(self.target_actor(new_state_batch) + target_policy_noise, min = self.min_action, max = self.max_action)
            #compute y
            qt0, qt1 = self.target_critics(new_state_batch, target_actions_batch)
            y = reward_batch + ~terminal_batch * self.discount_rate * torch.min(qt0, qt1)

        #compute critic losses
        q0, q1 = self.critics(old_state_batch, actions_batch)
        critics_loss = torch.nn.functional.mse_loss(q0, y) + torch.nn.functional.mse_loss(q1, y)

        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()

        if (++self.total_step_iterations % self.policy_update_frequency) == 0:
            #update actor
            policy_loss = (-self.critics(old_state_batch, self.actor(old_state_batch))[0]).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            #update target networks
            soft_update_target_network(self.target_actor, self.actor, self.target_update_rate)
            soft_update_target_network(self.target_critics, self.critics, self.target_update_rate)