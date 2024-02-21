import torch
import gymnasium
import pickle
import time
import yaml
import os
import shutil
import math
import argparse
import DDPG
import random
import TD3
from kalli import m_deepcopy

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Runs policy for X episodes and returns return reward
# A fixed seed is used for the eval environment
def eval_policy(env_name: str, seed: int = 256, eval_episodes: int = 10) -> float:
    if config['domain']['name'] == "Ant-v5":
        eval_env = gymnasium.make(config['domain']['name'], include_cfrc_ext_in_observation=False)
    else:
        eval_env = gymnasium.make(config['domain']['name'])

    total_return = 0
    for i in range(eval_episodes):
        state = eval_env.reset(seed=seed + i)[0]
        terminated, truncated = 0, 0
        while not (terminated or truncated):
            action = model.query_actor(torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE), add_noise=False)
            state, reward, terminated, truncated, _ = eval_env.step(action.tolist())
            total_return += reward

    return total_return / eval_episodes


def generate_model(model_name: str):
    match model_name:
        case 'DDPG':
            return DDPG.model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], config=config)
        case 'TD3':
            return TD3.model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], config=config, torch_device=TORCH_DEVICE)
        case _:
            assert False, 'invalid learning algorithm'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config.yaml')
    parser.add_argument("--starting_run", default=0, type=int)
    parser.add_argument("--final_run", default=int(1e6), type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config['domain']['name'] += '-v5'

    if config['domain']['name'] in ["Humanoid-v5", "HumanoidStandup-v5"]:
        env = gymnasium.make(config['domain']['name'], include_cinert_in_observation=False, include_cvel_in_observation=False, include_qfrc_actuator_in_observation=False, include_cfrc_ext_in_observation=False)
    if config['domain']['name'] == "Ant-v5":
        env = gymnasium.make(config['domain']['name'], include_cfrc_ext_in_observation=False)
    else:
        env = gymnasium.make(config['domain']['name'])

    num_actions = env.action_space.shape[0]
    num_states = env.observation_space.shape[0]

    # create evaluate directory
    eval_path = 'results/' + config['domain']['algo'] + '_' + config['domain']['name'] + '_' + str(time.time())
    os.makedirs(eval_path)
    shutil.copyfile(args.config, eval_path + '/config.yaml')

    for run in range(args.starting_run, min(config['domain']['runs'], args.final_run + 1)):
        # seed all the things
        torch.manual_seed(config['domain']['seed'] + run)
        env.action_space.seed(config['domain']['seed'] + 1000 * run)
        random.seed(config['domain']['seed'] + run)

        model = generate_model(config['domain']['algo'])

        # create evaluation file
        eval_file = open(eval_path + '/score' + str(run) + '.csv', 'w+')
        eval_max_return = -math.inf

        cur_state = torch.tensor(env.reset(seed=config['domain']['seed'])[0] + run, dtype=torch.float32, device=TORCH_DEVICE)
        for step in range(config['domain']['total_timesteps']):
            if step >= config['domain']['init_learn_timestep']:
                actions = model.query_actor(cur_state, add_noise=True)
            else:
                actions = torch.tensor(env.action_space.sample(), dtype=torch.float32, device=TORCH_DEVICE)

            new_state, reward, is_terminal, is_truncated, info = env.step(actions.tolist())

            model.erb.add_experience(old_state=cur_state, actions=actions.detach(), reward=reward, new_state=torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE), is_terminal=is_terminal)
            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)

            if step >= config['domain']['init_learn_timestep']:
                model.train_model_step()

            if is_terminal or is_truncated:
                cur_state = torch.tensor(env.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)

            if step % config['domain']['evaluation_frequency'] == 0 and step >= config['domain']['init_learn_timestep']:  # evaluate episode
                total_evalution_return = eval_policy(config['domain']['name'])
                print('Run: ' + str(run) + ' Training Step: ' + str(step) + ' return: ' + str(total_evalution_return))
                eval_file.write(str(total_evalution_return) + '\n')
                if (eval_max_return < total_evalution_return):
                    eval_max_return = total_evalution_return
                    best_model = m_deepcopy(model, excluded_keys=['erb'])

        best_model.save(eval_path + '/' + 'best_run' + str(run))
        pickle.dump(model.erb, open(eval_path + '/' + 'best_run' + str(run) + '_erb', 'wb'))
        print('Run: ' + str(run) + ' Max return: ' + str(eval_max_return))
        print('Finished score can be found at: ' + eval_path + '/score' + str(run) + '.csv')
