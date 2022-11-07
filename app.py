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
from DDPG import *
from TD3 import *
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    config = yaml.safe_load(open('config.yaml', 'r'))

    env = gymnasium.make(config['domain']['name'] + '-v4')
    env_eval = gymnasium.make(config['domain']['name'] + '-v4', reset_noise_scale = 0, render_mode='human')

    num_agents = 1
    num_actions = env.action_space.shape[0] #agent_size_modifier
    num_states = env.observation_space.shape[0] #len(env.observation_space(env.possible_agents[0]).shape) * agent_size_modifier

    match config['domain']['algo']:
        case 'DDPG':
            model = DDPG_model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], yaml_config=config)
        case 'TD3':
            model = TD3_model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], yaml_config=config)
        case _:
            assert false, 'invalid learning algorithm'

    #create evaluate file
    eval_path = 'results/' + config['domain']['algo'] + '_' + config['domain']['name'] + '_' + str(time.time()) 
    os.makedirs(eval_path)
    eval_file = open(eval_path + '/score.csv', 'w+')
    shutil.copyfile('./config.yaml', eval_path + '/config.yaml')


    for episode in range(config['domain']['episodes']):
        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)
        for step in range(env.spec.max_episode_steps):
            actions = model.query_actor(cur_state)
            #print('Actions: ' + str(actions) + ' State: ' + str(cur_state))

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            model.erb.add_experience(old_state = cur_state, actions = actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE), is_terminal = is_terminal)
            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)
            
            if episode >= config['domain']['learning_starts_ep']:
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
    print('Finished score can be found at: ' + eval_path + '/score.csv')