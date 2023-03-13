from gymnasium_robotics import mamujoco_v0
import MATD3
import torch
import argparse
import yaml
import os
import shutil
import math
import time
import copy
import pickle
from icecream import ic

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TORCH_DEVICE = "cpu"


# Runs policy for X episodes and returns return reward
# A fixed seed is used for the eval environment
def eval_policy(env_name: str, conf: str, seed: int = 256, eval_episodes: int = 10) -> float:
    eval_env = mamujoco_v0.parallel_env(scenario=env_name, agent_conf=conf)

    total_return = 0
    for i in range(eval_episodes):
        cur_state_dict = eval_env.reset(return_info=True, seed=seed + i)[0]
        terminated, truncated = 0, 0
        while not (terminated or truncated):
            cur_state = [torch.tensor(v, dtype=torch.float32, device=TORCH_DEVICE) for v in cur_state_dict.values()]
            actions = model.query_actor(cur_state, add_noise=False)
            actions_dict_numpy = {eval_env.possible_agents[agent_id]: actions[agent_id].tolist() for agent_id in range(len(eval_env.possible_agents))}
            cur_state_dict, reward_dict, is_terminal_dict, is_truncated_dict, info_dict = eval_env.step(actions_dict_numpy)
            total_return += reward_dict['agent_0']
            terminated = is_terminal_dict['agent_0']
            truncated = is_truncated_dict['agent_0']

    return total_return / eval_episodes


def generate_model(model_name: str):
    assert model_name == "MATD3"
    return MATD3.model(num_actions_spaces, num_observations_spaces, len(env.state()), min_action, max_action, config)
    # match model_name:
    #    case 'MADDPG':
    #        return DDPG.model(num_actions, num_states, min_action=env.action_space.low[0], max_action=env.action_space.high[0], config=config)
    #    case _:
    #        assert False, 'invalid learning algorithm'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config.yaml')
    parser.add_argument("--starting_run", default=0, type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config['domain']['algo'] = 'MA' + config['domain']['algo']

    env = mamujoco_v0.parallel_env(scenario=config['domain']['name'], agent_conf=config['domain']['factorization'], agent_obsk=1)

    num_actions_spaces = [env.action_space(agent).shape[0] for agent in env.possible_agents]
    num_observations_spaces = [env.observation_space(agent).shape[0] for agent in env.possible_agents]
    min_action = env.action_space(env.possible_agents[0]).low[0]
    max_action = env.action_space(env.possible_agents[0]).high[0]
    # ic(num_actions_spaces)
    ic(num_observations_spaces)

    # create evaluate directory
    eval_path = 'results/' + config['domain']['algo'] + '_' + str(config['domain']['factorization']) + '_' + config['domain']['name'] + '_' + str(time.time())
    os.makedirs(eval_path)
    shutil.copyfile(args.config, eval_path + '/config.yaml')

    for run in range(args.starting_run, config['domain']['runs']):
        # seed all the things
        torch.manual_seed(config['domain']['seed'] + run)
        [act_space.seed(config['domain']['seed'] + indx + run * 1000) for indx, act_space in enumerate(env.action_spaces.values())]

        model = generate_model(config['domain']['algo'])
        model.twin_critics[0].load_state_dict(torch.load('best_run0_critic'))
        model.actors[0].load_state_dict(torch.load('best_run0_actor'))
        eval_file = open(eval_path + '/score' + str(run) + '.csv', 'w+')
        eval_max_return = -math.inf

        cur_state_dict = env.reset(seed=config['domain']['seed'] + run, return_info=True)[0]
        for step in range(config['domain']['total_timesteps']):
            # sample actions
            cur_state = [torch.tensor(v, dtype=torch.float32, device=TORCH_DEVICE) for v in cur_state_dict.values()]
            cur_state_full = torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE)
            with torch.no_grad():
                if step >= config['domain']['init_learn_timestep']:
                    actions = model.query_actor(cur_state, add_noise=True)
                else:
                    actions = [torch.Tensor(act_space.sample()) for act_space in env.action_spaces.values()]
            actions_dict = {env.possible_agents[agent_id]: actions[agent_id].detach() for agent_id in range(len(env.possible_agents))}
            actions_dict_numpy = {env.possible_agents[agent_id]: actions[agent_id].tolist() for agent_id in range(len(env.possible_agents))}

            # step
            new_state_dict, reward_dict, is_terminal_dict, is_truncated_dict, info_dict = env.step(actions_dict_numpy)

            # store to ERB
            model.erb.add_experience(old_state=cur_state_full, actions=torch.tensor(env.map_local_actions_to_global_action(actions_dict_numpy), dtype=torch.float32, device=TORCH_DEVICE), reward=reward_dict[env.possible_agents[0]], new_state=torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE), is_terminal=is_terminal_dict[env.possible_agents[0]])

            # update cur_state
            new_state = [torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE) for state in new_state_dict.values()]
            cur_state = new_state

            if step >= config['domain']['init_learn_timestep']:
                model.train_model_step(env)

            if is_terminal_dict[env.possible_agents[0]] or is_truncated_dict[env.possible_agents[0]]:
                cur_state_dict = env.reset(return_info=True)[0]

            # evaluate
            if step % config['domain']['evaluation_frequency'] == 0 and step >= config['domain']['init_learn_timestep']:  # evaluate episode
                total_evalution_return = eval_policy(config['domain']['name'], config['domain']['factorization'])
                print('Run: ' + str(run) + ' Training Step: ' + str(step) + ' return: ' + str(total_evalution_return))
                eval_file.write(str(total_evalution_return) + '\n')
                if (eval_max_return < total_evalution_return):
                    eval_max_return = total_evalution_return
                    best_model = copy.deepcopy(model)

        print('Run: ' + str(run) + ' Max return: ' + str(eval_max_return))
        print('Finished score can be found at: ' + eval_path + '/score' + str(run) + '.csv')
        best_model.save(eval_path + '/' + 'best_run' + str(run))
        pickle.dump(model.erb, open(eval_path + '/' + 'best_run' + str(run) + '_erb', 'wb'))

    env.close()
