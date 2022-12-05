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

        self.discount_rate = config['DDPG']['gamma']
        self.target_update_rate = config['DDPG']['tau']
        self.mini_batch_size = config['DDPG']['N']
        self.noise_standard_deviation = config['DDPG']['sigma']

        self.actor = modules.actor(num_actions, num_states, max_action, config['TD3']['mu_bias'], device=TORCH_DEVICE)  # mu
        self.target_actor = copy.deepcopy(self.actor)  # mu'
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), config['DDPG']['optimizer_gamma'])
        self.critic = modules.critic(num_actions, num_states, config['TD3']['q_bias'], device=TORCH_DEVICE)  # q
        self.target_critic = copy.deepcopy(self.critic)  # q'
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), config['DDPG']['optimizer_gamma'])

        self.erb = ERB.experience_replay_buffer(config['DDPG']['experience_replay_buffer_size'])

    def query_actor(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        if add_noise:
            return torch.clamp(self.actor(state) + torch.randn(self.num_actions, device=TORCH_DEVICE) * self.noise_standard_deviation, min=self.min_action, max=self.max_action)
        else:
            return self.actor(state)

    def train_model_step(self) -> None:
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch = self.erb.sample_batch_and_split(self.mini_batch_size)

        # update critic
        q = self.critic(old_state_batch, actions_batch)
        y = reward_batch + ~terminal_batch * self.discount_rate * self.target_critic(new_state_batch, self.target_actor(new_state_batch))
        critic_loss = torch.nn.functional.mse_loss(q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        policy_loss = (-self.critic(old_state_batch, self.actor(old_state_batch))).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        modules.soft_update_target_network(self.target_actor, self.actor, self.target_update_rate)
        modules.soft_update_target_network(self.target_critic, self.critic, self.target_update_rate)

    def save(self, filename: str) -> None:
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        # pickle.dump(self.erb, open(filename + '_erb', 'wb'))

    def load(self, filename: str) -> None:
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critics_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.erb = pickle.load(open(filename + '_erb', 'rb'))
