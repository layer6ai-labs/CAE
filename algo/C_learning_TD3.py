import numpy as np
import torch
import torch.nn as nn
from algo.models import CNetworkCont, ActorNetwork
from algo.sampling_method import sample_long_range_transitions
from algo.util import updateTargets, preproc_og


class CLearningTD3:
    def __init__(self, env, device, **kwargs):

        self.env = env
        self.device = device

        # create networks
        self.critic1 = CNetworkCont(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                    hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        self.critic2 = CNetworkCont(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                    hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        self.actor = ActorNetwork(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                    hidden_layer_sizes=kwargs['hidden_layer_sizes'])

        # create target networks
        self.target_critic1 = CNetworkCont(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                    hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        self.target_critic2 = CNetworkCont(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                    hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        self.target_actor = ActorNetwork(d_obs=env.d_obs, d_goal=env.d_goal, d_actions=env.d_actions,
                                  hidden_layer_sizes=kwargs['hidden_layer_sizes'])

        # Initialize target network to match network
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # to device
        self.critic1.to(device)
        self.critic2.to(device)
        self.actor.to(device)

        self.target_critic1.to(device)
        self.target_critic2.to(device)
        self.target_actor.to(device)

        # training loss
        self.training_loss = nn.BCELoss()

        #set parameters
        self.target_network_copy_freq = kwargs['target_network_copy_freq']
        self.horizon_sampling_const = kwargs['horizon_sampling_const']
        self.goal_directed_eps = kwargs['goal_directed_eps']
        self.HER_fraction = kwargs['HER_fraction']
        self.use_HER = kwargs['use_HER']
        self.policy_freq=kwargs["policy_freq"]

        self.optimizer_critic1 = None
        self.optimizer_critic2 = None
        self.optimizer_actor = None
        self.train_step_count = 0

    def set_optimizer(self, learning_rate_critic, learning_rate_actor):
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate_critic)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate_critic)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

    def goal_conditioned_c_learning_policy(self, rng, exploration_epsilon=0.2, eval=False,
                                           horizon=None, noise_epsilon=0.2, g_norm=None, o_norm=None):

        max_horizon = horizon if horizon is not None else 50
        horizon_vals = np.array([i + 1 for i in range(max_horizon)]).reshape((-1, 1))

        def policy(state, goal):

            state = o_norm.normalize(state)
            goal = g_norm.normalize(goal)

            tiled_state = np.tile(state, max_horizon).reshape((max_horizon, -1))
            tiled_goal = np.tile(goal, reps=max_horizon).reshape((max_horizon, -1))

            x_action = torch.cat(
                (torch.tensor(tiled_state).float(), torch.tensor(tiled_goal).float(),
                 torch.tensor(horizon_vals).float()),
                dim=1).to(self.device)

            exploration_p = rng.uniform(low=0.0, high=1.0)

            if exploration_p < exploration_epsilon and not eval:
                a = rng.uniform(-1, 1, size=self.env.d_actions)
            else:
                actions = self.actor.forward(x_action)
                if eval:
                    action_noise = actions
                else:
                    n = noise_epsilon * np.random.randn(self.env.d_actions)
                    tiled_noise = np.tile(n, max_horizon).reshape((max_horizon, -1))
                    tiled_noise = torch.tensor(tiled_noise).float().to(self.device)
                    action_noise = actions + tiled_noise
                    action_noise = torch.clamp(action_noise, - self.env.action_max, self.env.action_max)

                x = torch.cat((x_action, torch.tensor(action_noise).float()), dim=1)
                accessibilities = self.critic1.forward(x).detach().cpu().numpy()
                max_accessibility = accessibilities.max()
                filter_level = 0.9 * max_accessibility
                attainable_horizons = (accessibilities >= filter_level).any(axis=1)
                # argmax only extracts 'True' values, and always returns the first.
                min_attainable_horizon = attainable_horizons.argmax()
                a = action_noise[min_attainable_horizon].detach().cpu().numpy()
            return a

        return policy

    def run_episode(self, action_policy, ep_length):

        observation = self.env.reset()
        ep_history = []
        success = 0
        for _ in range(ep_length):
            selected_action = action_policy(observation['observation'], observation['desired_goal'])
            new_observation, reward, done, _ = self.env.step(selected_action)
            # Store transition in episode buffer
            ep_history.append({"observation": observation['observation'].copy(),
                               "action": selected_action.copy(),
                               "observation_next": new_observation['observation'].copy(),
                               "achieved_goal": new_observation['achieved_goal'].copy(),
                               "desired_goal":new_observation['desired_goal'].copy()})
            observation = new_observation
            if reward > -0.5:
                success = 1
                break
            if done:
                break

        return np.array(ep_history), success

    def learn(self, gde_ep, dataset, batch_size, batch_sampling_rng, o_norm, g_norm):
        # Sample shorter transitions towards the start of training and high transition towards the end
        horizon_sampling_probs = (np.arange(50) + 1) ** -(
                    self.horizon_sampling_const * (1.0 - gde_ep / self.goal_directed_eps))

        # sample minibatch
        states_sample, actions_sample, goals_sample, horizons_sample, s_prime_sample, goal_achieved_sample = \
            sample_long_range_transitions(dataset, batch_size=batch_size, rng=batch_sampling_rng,
                                          horizon_sampling_probs=horizon_sampling_probs, env =self.env,
                                          her_fraction=self.HER_fraction, use_HER=self.use_HER)

        if self.train_step_count % self.target_network_copy_freq == 0:
            updateTargets(self.target_critic1, self.critic1)
            updateTargets(self.target_critic2, self.critic2)
            updateTargets(self.target_actor, self.actor)

        # save the unnormalized goals
        goals_sample_unnormalized = goals_sample.copy()

        # normalize data
        states_sample = preproc_og(states_sample)
        goals_sample = preproc_og(goals_sample)
        s_prime_sample = preproc_og(s_prime_sample)
        states_sample = o_norm.normalize(states_sample)
        goals_sample = g_norm.normalize(goals_sample)
        s_prime_sample = o_norm.normalize(s_prime_sample)

        # critic loss
        horizons_sample = horizons_sample.reshape((-1, 1))
        x_lhs = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                           torch.tensor(horizons_sample).float(), torch.tensor(actions_sample).float()),
                          dim=1).to(self.device)

        accessibilities_lhs1 = self.critic1.forward(x_lhs)
        accessibilities_lhs2 = self.critic2.forward(x_lhs)

        with torch.no_grad():
            x_rhs = torch.cat((torch.tensor(s_prime_sample).float(), torch.tensor(goals_sample).float(),
                               torch.tensor(horizons_sample - 1).float()),
                              dim=1).to(self.device)
            action_target = self.target_actor.forward(x_rhs).detach().cpu()

            x_rhs2 = torch.cat((torch.tensor(s_prime_sample).float(), torch.tensor(goals_sample).float(),
                                torch.tensor(horizons_sample - 1).float(), action_target.float()),
                               dim=1).to(self.device)
            accessibilities_rhs1 = self.target_critic1.forward(x_rhs2).detach()
            accessibilities_rhs2 = self.target_critic2.forward(x_rhs2).detach()
            accessibilities_rhs = torch.min(accessibilities_rhs1, accessibilities_rhs2)

        # when h=1 or s_prim =g we are having a different target.
        for i in range(len(horizons_sample)):

            if self.env.compute_reward(goal_achieved_sample[i], goals_sample_unnormalized[i], None) > -0.5:
                accessibilities_rhs[i] = 1.0

            if horizons_sample[i] == 1:
                if self.env.compute_reward(goal_achieved_sample[i], goals_sample_unnormalized[i], None) > -0.5:
                    accessibilities_rhs[i] = 1.0
                else:
                    accessibilities_rhs[i] = 0.0

        self.optimizer_critic1.zero_grad()
        loss_this_batch1 = self.training_loss(accessibilities_lhs1, accessibilities_rhs)
        loss_this_batch1.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        loss_this_batch2 = self.training_loss(accessibilities_lhs2, accessibilities_rhs)
        loss_this_batch2.backward()
        self.optimizer_critic2.step()

        # actor_loss
        if self.train_step_count % self.policy_freq == 0:
            x_a = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                             torch.tensor(horizons_sample).float()),
                            dim=1).to(self.device)
            actions_actor = self.actor.forward(x_a).cpu()

            x_a2 = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                              torch.tensor(horizons_sample).float(), actions_actor.float()),
                             dim=1).to(self.device)
            loss = -torch.mean(self.critic1.forward(x_a2))

            self.optimizer_actor.zero_grad()
            loss.backward()
            self.optimizer_actor.step()

        self.train_step_count += 1
