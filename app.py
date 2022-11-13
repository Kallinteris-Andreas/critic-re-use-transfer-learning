import numpy
import torch
import gymnasium
import copy
import time
import yaml
import os
import shutil
import sys
import math
import argparse
from ERB import *
from modules import *
from DDPG import *
from TD3 import *
#from multiagent_mujoco.mujoco_multi import MujocoMulti #https://github.com/schroederdewitt/multiagent_mujoco

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    env = gymnasium.make(config['domain']['name'])
    env_eval = gymnasium.make(config['domain']['name'], reset_noise_scale = 0, render_mode='human')
    #env_eval = gymnasium.make(config['domain']['name'], render_mode='human')

    num_agents = 1
    num_actions = env.action_space.shape[0] #agent_size_modifier
    num_states = env.observation_space.shape[0] #len(env.observation_space(env.possible_agents[0]).shape) * agent_size_modifier


    #create evaluate file
    eval_path = 'results/' + config['domain']['algo'] + '_' + config['domain']['name'] + '_' + str(time.time()) 
    os.makedirs(eval_path)
    shutil.copyfile(args.config, eval_path + '/config.yaml')


    for run in range(config['domain']['runs']):
        match config['domain']['algo']:
            case 'DDPG':
                model = DDPG_model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], yaml_config=config)
            case 'TD3':
                model = TD3_model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], yaml_config=config)
            case _:
                assert false, 'invalid learning algorithm'
        eval_file = open(eval_path + '/score' + str(run) + '.csv', 'w+')
        eval_max_return = -math.inf

        cur_state = torch.tensor(env.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)
        for steps in range(config['domain']['total_timesteps']):
            if steps >= config['domain']['init_learn_timestep']:
                actions = model.query_actor(cur_state)
            else:
                actions = torch.tensor(env.action_space.sample(), dtype=torch.float32, device=TORCH_DEVICE)

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())
            #print('step: ' + str(step) + ' state: ' + str(cur_state.tolist()) + ' actions: ' + str(actions.tolist()) + ' reward: ' + str(reward))#this is a debug line

            model.erb.add_experience(old_state = cur_state, actions = actions.detach(), reward = reward, new_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE), is_terminal = is_terminal)
            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)

            if steps >= config['domain']['init_learn_timestep']:
                model.train_model_step()

            if is_terminal or is_truncated:
                cur_state = torch.tensor(env.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)

            if steps % config['domain']['evaluation_frequency'] == 0 and steps >= config['domain']['init_learn_timestep']:#evaluate episode
                cur_state = torch.tensor(env_eval.reset(seed=None, options=None)[0], dtype=torch.float32, device=TORCH_DEVICE)
                total_evalution_return = 0
                for _ in range(env_eval.spec.max_episode_steps):
                    actions = model.query_actor(cur_state, add_noise=False)
                    new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())
                    cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)
                    total_evalution_return += reward

                    if is_terminal:
                        break
                print('Run: ' + str(run) + ' Training Step: ' + str(steps) + ' return: ' + str(total_evalution_return))
                eval_file.write(str(total_evalution_return) + '\n')
                if (eval_max_return < total_evalution_return):
                    eval_max_return = total_evalution_return
                    best_model = copy.deepcopy(model)
        print('Run: ' + str(run) + ' Max return: ' + str(eval_max_return))
        print('Finished score can be found at: ' + eval_path + '/score' + str(run) + '.csv')
        best_model.save(eval_path + '/' + 'best_' + str(run) + '_')