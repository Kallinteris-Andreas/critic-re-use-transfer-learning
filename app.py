import numpy
import torch
import gymnasium
import copy
import time
import yaml
import os
import shutil
from ERB import *
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco


TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class centralized_ddpg_agent_actor(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(observation_state_size, 128, bias=True, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, bias=True, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, action_space_size, bias=True, device=TORCH_DEVICE)

    def forward(self, observations):
        assert isinstance(observations, torch.Tensor)
        output = torch.tanh(self.linear1(observations))
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))
        return output


class centralized_ddpg_agent_critic(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(action_space_size + observation_state_size, 128, bias=False, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, bias=False, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, 1, bias=False, device=TORCH_DEVICE)

    def forward(self, observations , actions):
        assert isinstance(observations, torch.Tensor)
        assert isinstance(actions, torch.Tensor)
        output = torch.tanh(self.linear1(torch.cat((observations, actions), dim = 1)))
        output = torch.tanh(self.linear2(output))
        value = torch.tanh(self.linear3(output))
        return value


#source: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update_target_network(target, source, tau):
    assert tau >= 0 and tau <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DDPG_model():
    def __init__(self, num_actions, num_states, yaml_config):
        assert num_actions > 0 and num_agents > 0

        self.learning_rate = yaml_config['DDPG']['gamma']
        self.target_rate = yaml_config['DDPG']['tau']
        self.mini_batch_size = yaml_config['DDPG']['N']
        self.noise_variance = yaml_config['DDPG']['noise_var']
        
        self.actor = centralized_ddpg_agent_actor(num_actions, num_states) # mu
        self.target_actor = copy.deepcopy(self.actor) # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), yaml_config['DDPG']['optimizer_gamma'])#TODO check learning rate
        self.critic = centralized_ddpg_agent_critic(num_actions, num_states) # q
        self.target_critic = copy.deepcopy(self.critic) # q'
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), yaml_config['DDPG']['optimizer_gamma'])
        #self.critic_criterion = torch.nn.MSELoss()

        experience_replay_buffer_size = yaml_config['DDPG']['experience_replay_buffer_size']
        self.erb = experience_replay_buffer(experience_replay_buffer_size)

    def query_actor(self, state):
        return torch.clamp(self.actor(state) + torch.randn(num_actions).to(TORCH_DEVICE)*(self.noise_variance**0.5), min = env.action_space.low[0], max = env.action_space.high[0])

    def query_eval_actor(self, state):
        return self.actor(state)

    
    def train_model_step(self):
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        #calulate input date for optimaizers from sampled mini-batch
        old_state_batch, actions_batch, reward_batch, new_state_batch = self.erb.sample_batch_and_split(self.mini_batch_size)
        q = self.critic(old_state_batch, actions_batch)
        y = reward_batch + self.learning_rate * self.target_critic(new_state_batch, self.target_actor(new_state_batch))

        #update critic
        self.critic_optimizer.zero_grad()
        critic_loss = self.critic_criterion(q, y)
        critic_loss.backward()
        self.critic_optimizer.step()

        #update actor
        self.actor_optimizer.zero_grad()
        policy_loss = (-self.critic(old_state_batch, self.actor(old_state_batch))).mean()
        print(policy_loss)
        policy_loss.backward()
        self.actor_optimizer.step()

        #update target networks
        soft_update_target_network(self.target_actor, self.actor, self.target_rate)
        soft_update_target_network(self.target_critic, self.critic, self.target_rate)


if __name__ == "__main__":
    config = yaml.safe_load(open('config.yaml', 'r'))
    domain = config['domain']['name']

    env = gymnasium.make(domain + '-v4')
    env_eval = gymnasium.make(domain + '-v4', reset_noise_scale = 0, render_mode='human')

    #create evaluate file
    eval_path = 'results/DDPG_' + domain + '_' + str(time.time()) 
    os.makedirs(eval_path)
    eval_file = open(eval_path + '/score.csv', 'w+')
    shutil.copyfile('./config.yaml', eval_path + '/config.yaml')

    agent_size_modifier = 1 #len(env.possible_agents)
    num_agents = 1
    num_actions = env.action_space.shape[0] #agent_size_modifier
    num_states = env.observation_space.shape[0] #len(env.observation_space(env.possible_agents[0]).shape) * agent_size_modifier

    DDPG = DDPG_model(num_actions, num_states, config)


    for episode in range(config['domain']['episodes']):
        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32).to(TORCH_DEVICE)
        for step in range(env.spec.max_episode_steps):
            actions = DDPG.query_actor(cur_state)

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            DDPG.erb.add_experience(old_state = cur_state, actions= actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE), is_terminal = is_terminal or is_truncated)
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)
            
            DDPG.train_model_step()
            
            if is_terminal:
                break

        #evaluate episode
        cur_state = torch.tensor(env_eval.reset(seed=None)[0], dtype=torch.float32).to(TORCH_DEVICE)
        total_evalution_reward = 0
        for step in range(env_eval.spec.max_episode_steps):
            actions = DDPG.query_eval_actor(cur_state)
            new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())
            #if episode > 100:
                #print('step: ' + str(step) + 'state: ' + str(cur_state.tolist()) + ' actions: ' + str(actions.tolist()) + ' reward: ' + str(reward))#this is a debug line
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)
            total_evalution_reward += reward
            
            if is_terminal:
                break
        print("Episode: " + str(episode) + ' reward: ' + str(total_evalution_reward))
        eval_file.write(str(total_evalution_reward) + '\n')
