import numpy
import collections
import torch
import gymnasium
import random
import copy
import time
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco

experience_replay = collections.namedtuple('Experience', 'old_state, actions, reward, new_state, terminal1')
agent_spaces = collections.namedtuple('agent_def', 'observation_space, action_space')
#note: actions, old_state, new_state are torch.Tensor reward is float32 scalar, terminal is bool

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
        mini_batch = self.sample_batch(mini_batch_size) 
        old_state_batch = torch.stack([exp.old_state for exp in mini_batch])
        new_state_batch = torch.stack([exp.new_state for exp in mini_batch])
        actions_batch = torch.stack([exp.actions for exp in mini_batch])
        reward_batch = torch.tensor([[exp.reward for exp in mini_batch]], dtype=torch.float32, device=TORCH_DEVICE).transpose(0,1)
        return old_state_batch, actions_batch, reward_batch, new_state_batch


#source: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


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

def query_multi_agent_actor_team(actor_team, observations):
    #result = []
    #for actor in actor_team:
        

    None


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


if __name__ == "__main__":
    #domain = "Swimmer"
    domain = "Hopper"
    env = gymnasium.make(domain + '-v4')
    env_eval = gymnasium.make(domain + '-v4', reset_noise_scale = 0, render_mode='human')
    eval_file = open ('results/DDPG_' + domain + '_' + str(time.time()), 'w')
    agent_size_modifier = 1 #len(env.possible_agents)
    num_agents = 1
    num_actions = env.action_space.shape[0] #agent_size_modifier
    num_states = env.observation_space.shape[0] #len(env.observation_space(env.possible_agents[0]).shape) * agent_size_modifier

    learning_rate = 0.999999 # gamma / discount_rate
    learning_rate = 0.99 # gamma / discount_rate
    #learning_rate = 1
    target_rate = 0.01 # tau
    mini_batch_size = 100

    ddpg_agent_actor = centralized_ddpg_agent_actor(num_actions, num_states)
    ddpg_agent_target_actor = copy.deepcopy(ddpg_agent_actor)
    ddpg_agent_actor_optimizer = torch.optim.Adam(ddpg_agent_actor.parameters())#TODO check learning rate
    ddpg_agent_critic = centralized_ddpg_agent_critic(num_actions, num_states)
    ddpg_agent_target_critic = copy.deepcopy(ddpg_agent_critic)
    ddpg_agent_critic_optimizer = torch.optim.Adam(ddpg_agent_critic.parameters())
    critic_criterion = torch.nn.MSELoss()


    erb = experience_replay_buffer(max_size = 1000 * 1000)
    for episode in range(10000):
        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32).to(TORCH_DEVICE)
        for step in range(env.spec.max_episode_steps):
            actions = torch.clamp(ddpg_agent_actor(cur_state) + torch.randn(num_actions).to(TORCH_DEVICE)*0.01, min = env.action_space.low[0], max = env.action_space.high[0])#to_CPU

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            erb.add_experience(old_state = cur_state, actions= actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE), is_terminal = is_terminal or is_truncated)
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)

            #calulate input date for optimaizers from sampled mini-batch
            old_state_batch, actions_batch, reward_batch, new_state_batch = erb.sample_batch_and_split(mini_batch_size)
            q = ddpg_agent_critic(old_state_batch, actions_batch)
            y = reward_batch + learning_rate * ddpg_agent_target_critic(new_state_batch, ddpg_agent_target_actor(new_state_batch))

            #update critic
            ddpg_agent_critic_optimizer.zero_grad()
            critic_loss = critic_criterion(q, y)
            critic_loss.backward()
            ddpg_agent_critic_optimizer.step()

            #update actor
            ddpg_agent_actor_optimizer.zero_grad()
            policy_loss = -ddpg_agent_critic(old_state_batch, ddpg_agent_actor(old_state_batch)).mean()
            policy_loss.backward()
            ddpg_agent_actor_optimizer.step()

            #update target networks
            soft_update(ddpg_agent_target_actor, ddpg_agent_actor, target_rate)
            soft_update(ddpg_agent_target_critic, ddpg_agent_critic, target_rate)
            
            if is_truncated:
                break

        #evaluate episode
        cur_state = torch.tensor(env_eval.reset(seed=None)[0], dtype=torch.float32).to(TORCH_DEVICE)
        total_evalution_reward = 0
        for step in range(env_eval.spec.max_episode_steps):
            actions = ddpg_agent_actor.forward(cur_state)
            new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())
            cur_state = torch.tensor(new_state, dtype=torch.float32).to(TORCH_DEVICE)
            total_evalution_reward += reward
            #env.render()
            
            if is_truncated:
                break
        print("Episode: " + str(episode) + ' reward: ' + str(total_evalution_reward))
        eval_file.write(str(total_evalution_reward) + '\n')
