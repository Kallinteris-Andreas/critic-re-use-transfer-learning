import numpy
import collections
import torch
import gymnasium
import random
import copy
import time
import yaml
import os
import shutil
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco

experience_replay = collections.namedtuple('Experience', 'old_state, actions, reward, new_state, terminal1')
#note: actions, old_state, new_state are torch.Tensor reward is float32 scalar, terminal is bool
agent_spaces = collections.namedtuple('agent_def', 'observation_space, action_space')

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class experience_replay_buffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add_experience(self, old_state, actions, reward, new_state, is_terminal):
        assert len(self.buffer) <= self.max_size
        if len(self.buffer) == self.max_size-1:
            print("filled the ERB")
        if len(self.buffer) != self.max_size:
            self.buffer.append(experience_replay(old_state, actions, reward, new_state, is_terminal))
        else:
            self.buffer[random.randint(0, max_size-1)] = experience_replay(old_state, actions, reward, new_state, is_terminal)

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





class centralized_ddpg_agent_actor(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(observation_state_size, 128, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, action_space_size, device=TORCH_DEVICE)

    def forward(self, observations):
        assert isinstance(observations, torch.Tensor)
        output = torch.tanh(self.linear1(observations))
        output = torch.tanh(self.linear2(output))
        output = torch.tanh(self.linear3(output))
        return output


class centralized_ddpg_agent_critic(torch.nn.Module):
    def __init__(self, action_space_size, observation_state_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(action_space_size + observation_state_size, 128, device=TORCH_DEVICE)
        self.linear2 = torch.nn.Linear(128, 256, device=TORCH_DEVICE)
        self.linear3 = torch.nn.Linear(256, 1, device=TORCH_DEVICE)

    def forward(self, observations , actions):
        #assert observations.size(1) == 27
        assert isinstance(observations, torch.Tensor)
        assert isinstance(actions, torch.Tensor)
        output = self.linear1(torch.cat((observations, actions), dim = 1))
        output = (self.linear2(output))
        value = (self.linear3(output))
        return value


#source: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update_target_network(target, source, tau):
    assert tau >= 0 and tau <= 1
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DDPG_model():
    def __init__(self, num_actions, num_states, yaml_config=None):
        assert num_actions > 0 and num_agents > 0

        self.learning_rate = 0.99 # gamma / discount_rate
        self.target_rate = 0.01 # tau
        self.mini_batch_size = 100
        self.noise_variance = 0.01
        if yaml_config != None:
            self.learning_rate = yaml_config['DDPG']['gamma']
            self.target_rate = yaml_config['DDPG']['tau']
            self.mini_batch_size = yaml_config['DDPG']['N']
            self.noise_variance = yaml_config['DDPG']['noise_var']
        
        self.actor = centralized_ddpg_agent_actor(num_actions, num_states) # mu
        self.target_actor = copy.deepcopy(self.actor) # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())#TODO check learning rate
        self.critic = centralized_ddpg_agent_critic(num_actions, num_states) # q
        self.target_critic = copy.deepcopy(self.critic) # q'
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_criterion = torch.nn.MSELoss()

        experience_replay_buffer_size = 1000_000
        if yaml_config != None:
            experience_replay_buffer_size = yaml_config['DDPG']['experience_replay_buffer_size']
        self.erb = experience_replay_buffer(experience_replay_buffer_size)

    def query_actor(self, state):
        return torch.clamp(self.actor(state) + torch.randn(num_actions).to(TORCH_DEVICE)*(self.noise_variance**0.5), min = env.action_space.low[0], max = env.action_space.high[0])

    def query_eval_actor(self, state):
        return self.actor(state)

    
    def update_model_step(self):
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
        policy_loss = -self.critic(old_state_batch, self.actor(old_state_batch)).mean()
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


    for episode in range(10000):
        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32).to(TORCH_DEVICE)
        for step in range(env.spec.max_episode_steps):
            actions = DDPG.query_actor(cur_state)

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            DDPG.erb.add_experience(old_state = cur_state, actions= actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE), is_terminal = is_terminal or is_truncated)
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)
            
            DDPG.update_model_step()
            
            if is_truncated:
                break

        #evaluate episode
        cur_state = torch.tensor(env_eval.reset(seed=None)[0], dtype=torch.float32).to(TORCH_DEVICE)
        total_evalution_reward = 0
        for step in range(env_eval.spec.max_episode_steps):
            actions = DDPG.query_eval_actor(cur_state)
            new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)
            total_evalution_reward += reward
            
            if is_truncated:
                break
        print("Episode: " + str(episode) + ' reward: ' + str(total_evalution_reward))
        eval_file.write(str(total_evalution_reward) + '\n')
