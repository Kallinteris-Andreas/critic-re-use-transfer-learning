import numpy
import torch
import gymnasium
import copy
import time
import yaml
import os
import shutil
import sys
from ERB import *
from modules import *
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco


TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG_model():
    def __init__(self, num_actions, num_states, yaml_config):
        assert num_actions > 0 and num_agents > 0
        self.num_actions = num_actions

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
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE)*self.noise_standard_deviation, min = env.action_space.low[0], max = env.action_space.high[0])
        else:
            return self.actor(state)


    def train_model_step(self):
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch = self.erb.sample_batch_and_split(self.mini_batch_size)

        #update critic
        q = self.critic(old_state_batch, actions_batch)
        y = reward_batch + self.discount_rate * self.target_critic(new_state_batch, self.target_actor(new_state_batch))
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


class TD3_model():
    #TODO handle truncated cases
    def __init__(self, num_actions, num_states, yaml_config):
        assert num_actions > 0 and num_agents > 0
        self.num_actions = num_actions
        self.total_step_iterations = 0

        self.discount_rate = yaml_config['TD3']['gamma']
        self.target_update_rate = yaml_config['TD3']['tau']
        self.mini_batch_size = yaml_config['TD3']['N']
        self.noise_exploration_standard_deviation = yaml_config['TD3']['sigma_explore']
        self.noise_policy_standard_deviation = yaml_config['TD3']['sigma_policy']
        self.noise_policy_clip = yaml_config['TD3']['noise_policy_clip']
        self.policy_update_frequency = yaml_config['TD3']['d']
        
        self.actor = actor(num_actions, num_states, bias=True, device=TORCH_DEVICE) # mu
        self.target_actor = copy.deepcopy(self.actor) # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), yaml_config['TD3']['optimizer_gamma'])
        self.critics= twin_critic(num_actions, num_states, bias=False, device=TORCH_DEVICE) # q0-1
        self.target_critics = copy.deepcopy(self.critics) # q0-1'
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), yaml_config['TD3']['optimizer_gamma'])

        self.erb = experience_replay_buffer(yaml_config['TD3']['experience_replay_buffer_size'])

    def query_actor(self, state, add_noise=True):
        if add_noise:
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE)*self.noise_exploration_standard_deviation, min = env.action_space.low[0], max = env.action_space.high[0])
        else:
            return self.actor(state)


    def train_model_step(self):
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch = self.erb.sample_batch_and_split(self.mini_batch_size)


        #update critic
        target_policy_noise = (torch.randn(self.mini_batch_size, 1, device=TORCH_DEVICE)*self.noise_policy_standard_deviation).clamp(min= -self.noise_policy_clip, max=self.noise_policy_clip)
        target_actions_batch = torch.clamp(self.target_actor(new_state_batch) + target_policy_noise, min = env.action_space.low[0], max = env.action_space.high[0])
        qt0, qt1 = self.target_critics(new_state_batch, target_actions_batch)
        y = reward_batch + self.discount_rate * torch.min(qt0, qt1)
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


if __name__ == "__main__":
    config = yaml.safe_load(open('config.yaml', 'r'))

    env = gymnasium.make(config['domain']['name'] + '-v4')
    env_eval = gymnasium.make(config['domain']['name'] + '-v4', reset_noise_scale = 0, render_mode='human')

    num_agents = 1
    num_actions = env.action_space.shape[0] #agent_size_modifier
    num_states = env.observation_space.shape[0] #len(env.observation_space(env.possible_agents[0]).shape) * agent_size_modifier

    match config['domain']['algo']:
        case 'DDPG':
            model = DDPG_model(num_actions, num_states, config)
        case 'TD3':
            model = TD3_model(num_actions, num_states, config)
        case _:
            print('invalid learning algorithm')
            exit(1)

    #create evaluate file
    eval_path = 'results/' + config['domain']['algo'] + '_' + config['domain']['name'] + '_' + str(time.time()) 
    os.makedirs(eval_path)
    eval_file = open(eval_path + '/score.csv', 'w+')
    shutil.copyfile('./config.yaml', eval_path + '/config.yaml')


    for episode in range(config['domain']['episodes']):
        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)
        for step in range(env.spec.max_episode_steps):
            actions = model.query_actor(cur_state)

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            model.erb.add_experience(old_state = cur_state, actions = actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE), is_terminal = is_terminal or is_truncated)
            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)
            
            model.train_model_step()
            
            if is_terminal:
                break

        #evaluate episode
        cur_state = torch.tensor(env_eval.reset(seed=None)[0], dtype=torch.float32, device=TORCH_DEVICE)
        total_evalution_reward = 0
        for step in range(env_eval.spec.max_episode_steps):
            actions = model.query_actor(cur_state, add_noise=False)
            new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())
            #if episode > 100:
                #print('step: ' + str(step) + 'state: ' + str(cur_state.tolist()) + ' actions: ' + str(actions.tolist()) + ' reward: ' + str(reward))#this is a debug line
            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)
            total_evalution_reward += reward
            
            if is_terminal:
                break
        print("Episode: " + str(episode) + ' reward: ' + str(total_evalution_reward))
        eval_file.write(str(total_evalution_reward) + '\n')