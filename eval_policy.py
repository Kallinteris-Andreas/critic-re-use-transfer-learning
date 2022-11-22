import torch
import gymnasium
import argparse
import yaml
import icecream
from modules import *

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.res + 'config.yaml', 'r'))
    #env_eval = gymnasium.make(config['domain']['name'], render_mode='human')
    #env_eval = gymnasium.make(config['domain']['name'], reset_noise_scale = 0, render_mode='human')
    #env_eval = gymnasium.make(config['domain']['name'], reset_noise_scale = 0)
    env_eval = gymnasium.make(config['domain']['name'])
    
    mu = actor(env_eval.action_space.shape[0], env_eval.observation_space.shape[0], env_eval.action_space.high[0], config['TD3']['mu_bias'], device=TORCH_DEVICE)
    mu.load_state_dict(torch.load(args.res + '/best_actor.pt'))

    #eval
    for episode in range(25):
        #cur_state = torch.tensor(env_eval.reset(seed=None, options={'x_init': 0, 'y_init': 0})[0], dtype=torch.float32, device=TORCH_DEVICE)
        cur_state = torch.tensor(env_eval.reset()[0], dtype=torch.float32, device=TORCH_DEVICE)
        total_evalution_return = 0
        for step in range(env_eval.spec.max_episode_steps):
            actions = mu(cur_state, add_noise=False)
            new_state, reward, is_terminal, is_truncated, info = env_eval.step(actions.tolist())

            cur_state = torch.tensor(new_state, dtype=torch.float32, device=TORCH_DEVICE)

            total_evalution_return += reward
            
            if is_terminal:
                break
        print("Episode: " + str(episode) + ' Return: ' + str(total_evalution_return))