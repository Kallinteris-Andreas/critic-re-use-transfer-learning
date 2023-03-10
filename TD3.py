import torch
import copy
import pickle
import ERB
import modules

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model():
    def __init__(self, num_actions: int, num_states: int, min_action: float, max_action: float, config: dict):
        assert num_actions > 0 and num_states > 0 and min_action < max_action
        self.num_actions = num_actions
        self.min_action = min_action
        self.max_action = max_action
        self.total_step_iterations = 0

        self.discount_rate = config['TD3']['gamma']
        self.target_update_rate = config['TD3']['tau']
        self.mini_batch_size = config['TD3']['N']
        self.noise_exploration_standard_deviation = config['TD3']['sigma_explore']
        self.noise_policy_standard_deviation = config['TD3']['sigma_policy']
        self.noise_policy_clip = config['TD3']['noise_policy_clip']
        self.policy_update_frequency = config['TD3']['d']

        self.actor = modules.actor(num_actions, num_states, max_action, config['TD3']['mu_bias'], device=TORCH_DEVICE)  # mu
        self.target_actor = copy.deepcopy(self.actor)  # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config['TD3']['optimizer_gamma'])
        self.critics = modules.twin_critic(num_actions, num_states, config['TD3']['q_bias'], device=TORCH_DEVICE)  # q0-1
        self.target_critics = copy.deepcopy(self.critics)  # q0-1'
        self.critics_optimizer = torch.optim.Adam(self.critics.parameters(), config['TD3']['optimizer_gamma'])

        self.erb = ERB.experience_replay_buffer(config['TD3']['experience_replay_buffer_size'], device=TORCH_DEVICE)

    def query_actor(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        if add_noise:
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE) * self.noise_exploration_standard_deviation, min=self.min_action, max=self.max_action)
        else:
            return self.actor(state)

    def train_model_step(self) -> None:
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch = self.erb.sample_batch_and_split(self.mini_batch_size)

        # update critic
        with torch.no_grad():
            # select target action
            target_policy_noise = (torch.randn(self.mini_batch_size, 1, device=TORCH_DEVICE) * self.noise_policy_standard_deviation).clamp(min=-self.noise_policy_clip, max=self.noise_policy_clip)
            target_actions_batch = torch.clamp(self.target_actor(new_state_batch) + target_policy_noise, min=self.min_action, max=self.max_action)
            # compute y
            qt0, qt1 = self.target_critics(new_state_batch, target_actions_batch)
            y = reward_batch + ~terminal_batch * self.discount_rate * torch.min(qt0, qt1)

        # compute critic losses
        q0, q1 = self.critics(old_state_batch, actions_batch)
        critics_loss = torch.nn.functional.mse_loss(q0, y) + torch.nn.functional.mse_loss(q1, y)

        self.critics_optimizer.zero_grad()
        critics_loss.backward()
        self.critics_optimizer.step()

        self.total_step_iterations += 1
        if (self.total_step_iterations % self.policy_update_frequency) == 0:
            # update actor
            policy_loss = (-self.critics.critic_0(old_state_batch, self.actor(old_state_batch))).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            modules.soft_update_target_network(self.target_actor, self.actor, self.target_update_rate)
            modules.soft_update_target_network(self.target_critics, self.critics, self.target_update_rate)

    def save(self, filename: str) -> None:
        torch.save(self.critics.state_dict(), filename + "_twin_critic")
        torch.save(self.target_critics.state_dict(), filename + "_target_twin_critic")
        torch.save(self.critics_optimizer.state_dict(), filename + "_twin_critics_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.target_actor.state_dict(), filename + "_target_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        # pickle.dump(self.erb, open(filename + '_erb', 'wb'))

    def load(self, filename: str) -> None:
        self.critics.load_state_dict(torch.load(filename + "_twin_critic"))
        self.critics_target .load_state_dict(torch.load(filename + "_target_twin_critic"))
        self.critics_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        # self.critics_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.target_actor.load_state_dict(torch.load(filename + "_target_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actor)

        self.erb = pickle.load(open(filename + '_erb', 'rb'))
