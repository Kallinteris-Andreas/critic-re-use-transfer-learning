import torch
import yaml
import copy
from ERB import *
from modules import *

class DDPG_model():
    def __init__(self, num_actions, num_states, min_action, max_action, yaml_config):
        assert num_actions > 0 and num_states > 0 and min_action < max_action
        self.num_actions = num_actions
        self.min_action = min_action
        self.max_action = max_action

        self.discount_rate = yaml_config['DDPG']['gamma']
        self.target_update_rate = yaml_config['DDPG']['tau']
        self.mini_batch_size = yaml_config['DDPG']['N']
        self.noise_standard_deviation = yaml_config['DDPG']['sigma']
        
        self.actor = actor(num_actions, num_states, bias=True, device=TORCH_DEVICE) # mu
        self.target_actor = copy.deepcopy(self.actor) # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), yaml_config['DDPG']['optimizer_gamma'])
        self.critic = critic(num_actions, num_states, bias=False, device=TORCH_DEVICE) # q
        self.target_critic = copy.deepcopy(self.critic) # q'
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), yaml_config['DDPG']['optimizer_gamma'])

        self.erb = experience_replay_buffer(yaml_config['DDPG']['experience_replay_buffer_size'])

    def query_actor(self, state, add_noise=True):
        if add_noise:
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE)*self.noise_standard_deviation, min = self.min_action, max = self.max_action)
        else:
            return self.actor(state)


    def train_model_step(self):
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch = self.erb.sample_batch_and_split(self.mini_batch_size)

        #update critic
        q = self.critic(old_state_batch, actions_batch)
        y = reward_batch + ~terminal_batch * self.discount_rate * self.target_critic(new_state_batch, self.target_actor(new_state_batch))
        #print('q: ' + str(torch.transpose(q, 0, 1).detach().to('cpu')))
        #print('y: ' + str(torch.transpose(y, 0, 1).detach().to('cpu')))
        self.critic_optimizer.zero_grad()
        critic_loss = torch.nn.functional.mse_loss(q, y)
        #print('Critic Lost: ' + str(critic_loss.detach().to('cpu')))
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor
        self.actor_optimizer.zero_grad()
        policy_loss = (-self.critic(old_state_batch, self.actor(old_state_batch))).mean()
        #print('Policy Loss: ' + str(policy_loss.detach().to('cpu')))
        policy_loss.backward()
        self.actor_optimizer.step()

        #update target networks
        soft_update_target_network(self.target_actor, self.actor, self.target_update_rate)
        soft_update_target_network(self.target_critic, self.critic, self.target_update_rate)