import torch
import copy
# import pickle
import ERB
import modules
# from icecream import ic

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TORCH_DEVICE = "cpu"


class model():
    def __init__(self, num_actions_spaces: list[int], num_states_spaces: list[int], num_state_global: int, min_action: float, max_action: float, config: dict):
        assert min_action < max_action and len(num_actions_spaces) == len(num_states_spaces)
        self.num_actions_spaces = num_actions_spaces
        self.num_states_spaces = num_states_spaces
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

        self.actors, self.target_actors, self.actors_optimizers = [], [], []
        for agent_id in range(len(num_actions_spaces)):
            self.actors.append(modules.actor(num_actions_spaces[agent_id], num_states_spaces[agent_id], max_action, config['TD3']['mu_bias'], device=TORCH_DEVICE))
            self.target_actors.append(copy.deepcopy(self.actors[agent_id]))
            self.actors_optimizers.append(torch.optim.Adam(self.actors[agent_id].parameters(), config['TD3']['optimizer_gamma']))

        self.twin_critics, self.target_twin_critics, self.twin_critics_optimizer = [], [], []
        for agent_id in range(len(num_actions_spaces)):
            self.twin_critics.append(modules.twin_critic(sum(num_actions_spaces), num_state_global, config['TD3']['q_bias'], device=TORCH_DEVICE))
            self.target_twin_critics.append(copy.deepcopy(self.twin_critics[agent_id]))
            self.twin_critics_optimizer.append(torch.optim.Adam(self.twin_critics[agent_id].parameters(), config['TD3']['optimizer_gamma']))

        self.erb = ERB.experience_replay_buffer(config['TD3']['experience_replay_buffer_size'], device=TORCH_DEVICE)

    @property
    def num_agents(self):
        return len(self.actors)

    def query_actor(self, state: list[torch.Tensor], add_noise: bool = True) -> list[torch.Tensor]:
        # eg state: [tensor([-0.0128,  0.0697,  0.0434,  0.0337, -0.0276]), tensor([-0.0708,  0.0141,  0.0259,  0.0337, -0.0276])]
        # assert len(state) == len(self.num_actions_spaces)
        # assert [s.size()[0] for s in state] == self.num_states_spaces
        actions = []
        for agent_id in range(len(state)):
            actions.append(self.actors[agent_id](state[agent_id]))

        if add_noise is True:
            for agent_id in range(len(state)):
                actions[agent_id] = (actions[agent_id] + torch.randn_like(actions[agent_id]) * self.noise_exploration_standard_deviation).clamp(self.min_action, self.max_action)

        return actions

    def map_states(self, zoo_env, state_batch: torch.Tensor):
        # global -> local
        observations_partions = zoo_env.map_global_state_to_local_observations([t for t in range(zoo_env.single_agent_env.observation_space.shape[0])])
        mapped_states = []
        for agent_id, partition in enumerate(observations_partions.values()):
            out_state = torch.empty(self.mini_batch_size, self.num_states_spaces[agent_id], device=TORCH_DEVICE)
            for idx, obs_idx in enumerate(partition):
                out_state[:, idx] = state_batch[:, int(obs_idx)]
            mapped_states.append(out_state)

        return mapped_states

    def map_actions(self, zoo_env, fact_acts):
        actions = torch.zeros(self.mini_batch_size, sum(self.num_actions_spaces), device=TORCH_DEVICE)
        agent_action_partition = self.action_part(zoo_env)
        for agent_id in range(self.num_agents):
            for act_id in range(len(agent_action_partition[agent_id])):
                actions[:, agent_action_partition[agent_id][act_id]] = fact_acts[agent_id][:, act_id]

        return actions

    def action_part(self, zoo_env) -> list[list[int]]:
        # return list of action partiions
        return [[part.act_ids for part in partion] for partion in zoo_env.agent_action_partitions]

    def train_model_step(self, zoo_env) -> None:
        if len(self.erb.buffer) < self.mini_batch_size:
            return

        for agent_id in range(self.num_agents):
            old_state_batch, actions_batch, reward_batch, new_state_batch, terminal_batch = self.erb.sample_batch_and_split(self.mini_batch_size)

            # remap
            new_state_batch_factored = self.map_states(zoo_env, new_state_batch)
            old_state_batch_factored = self.map_states(zoo_env, old_state_batch)

            # update critic
            with torch.no_grad():
                # select target action
                target_policy_noise = (torch.randn(self.mini_batch_size, 1, device=TORCH_DEVICE) * self.noise_policy_standard_deviation).clamp(min=-self.noise_policy_clip, max=self.noise_policy_clip)

                target_actions_pre = self.query_actor(new_state_batch_factored, add_noise=False)
                target_actions = self.map_actions(zoo_env, target_actions_pre)
                target_actions_batch = torch.clamp(target_actions + target_policy_noise, min=self.min_action, max=self.max_action)

                # compute y
                qt0, qt1 = self.target_twin_critics[agent_id](new_state_batch, target_actions_batch)
                y = reward_batch + ~terminal_batch * self.discount_rate * torch.min(qt0, qt1)

            # compute critic losses
            q0, q1 = self.twin_critics[agent_id](old_state_batch, actions_batch)
            critics_loss = torch.nn.functional.mse_loss(q0, y) + torch.nn.functional.mse_loss(q1, y)

            self.twin_critics_optimizer[agent_id].zero_grad()
            critics_loss.backward()
            self.twin_critics_optimizer[agent_id].step()

            if (++self.total_step_iterations % self.policy_update_frequency) == 0:
                # update actor
                actor_action = self.actors[agent_id](old_state_batch_factored[agent_id])
                agent_action_partition = self.action_part(zoo_env)[agent_id]
                # replace actor action
                new_actions = actions_batch.clone().detach()
                for index, act_id in enumerate(agent_action_partition):
                    new_actions[:, act_id] = actor_action[:, index]

                policy_loss = (-self.twin_critics[agent_id].critic_0(old_state_batch, new_actions)).mean()

                self.actors_optimizers[agent_id].zero_grad()
                policy_loss.backward()
                self.actors_optimizers[agent_id].step()

                # update target networks
                modules.soft_update_target_network(self.target_actors[agent_id], self.actors[agent_id], self.target_update_rate)
                modules.soft_update_target_network(self.target_twin_critics[agent_id], self.twin_critics[agent_id], self.target_update_rate)

    def save(self, filename: str) -> None:
        torch.save(self.twin_critics, filename + "_twin_critics")
        torch.save(self.target_twin_critics, filename + "_target_twin_critics")
        torch.save(self.twin_critics_optimizer, filename + "_twin_critics_optimizers")

        torch.save(self.actors, filename + "_actors")
        torch.save(self.target_actors, filename + "_target_actors")
        torch.save(self.actors_optimizers, filename + "_actors_optimizers")

        # pickle.dump(self.erb, open(filename + '_erb', 'wb'))

    def load(self, filename: str) -> None:
        # self.critics.load_state_dict(torch.load(filename + "_critic"))
        # self.critics_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        # self.critics_target = copy.deepcopy(self.critic)

        # self.actors.load_state_dict(torch.load(filename + "_actor"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actors)

        # self.erb = pickle.load(open(filename + '_erb', 'wb'))
        pass
