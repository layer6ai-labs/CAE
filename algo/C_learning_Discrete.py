import numpy as np
import torch
import torch.nn as nn
from algo.models import CNetworkDisc
from algo.sampling_method import sample_long_range_transitions
from envs.precipice import PrecipiceEnv
from envs.dubin_maze import DiscreteDubinsCar
from envs.minimaze import MiniMazeEnv


class CLearningDiscrete:

    def __init__(self, env, device, **kwargs):
        self.env = env
        self.device = device

        # create network
        self.network = CNetworkDisc(d_obs=env.d_obs,
                                            d_goal=env.d_goal,
                                            n_actions=env.n_actions,
                                            hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        # create target network
        self.target_network = CNetworkDisc(d_obs=env.d_obs,
                                                   d_goal=env.d_goal,
                                                   n_actions=env.n_actions,
                                                   hidden_layer_sizes=kwargs['hidden_layer_sizes'])

        # Initialize target network to match network
        self.target_network.load_state_dict(self.network.state_dict())

        # to device
        self.network.to(device)
        self.target_network.to(device)

        # training loss
        self.training_loss = nn.BCEWithLogitsLoss()

        # set parameters
        self.target_network_copy_freq = kwargs['target_network_copy_freq']
        self.horizon_sampling_const = kwargs['horizon_sampling_const']
        self.goal_directed_eps = kwargs['goal_directed_eps']
        self.HER_fraction = kwargs['HER_fraction']
        self.use_HER = kwargs['use_HER']
        self.clipping = kwargs['c_clipping']
        self.optimizer = None
        self.scheduler = None
        self.train_step_count = 0

    def set_optimizer(self, learning_rate, use_scheduler=False, milestones=None, gamma=None):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones,
                                                                  gamma=gamma)

    def goal_conditioned_c_learning_policy(self, rng, exploration_epsilon=0.15, eval=False, horizon=None):

        gamma = 0.9
        max_horizon = horizon if horizon is not None else 50

        def policy(state, goal):

            if self.clipping:
                min_horizon = np.floor(np.linalg.norm(goal - state[:2], ord=self.env.norm))
                min_horizon = min(min_horizon, max_horizon)  # Ensures at least one horizon
                horizon_vals = np.arange(min_horizon, max_horizon + 1).reshape((-1, 1))
            else:
                horizon_vals = np.array([i + 1 for i in range(max_horizon)]).reshape((-1, 1))

            num_horizons = len(horizon_vals)

            tiled_state = np.tile(state, num_horizons).reshape((num_horizons, -1))
            tiled_goal = np.tile(goal, reps=num_horizons).reshape((num_horizons, -1))

            x = torch.cat(
                (
                    torch.tensor(tiled_state).float(),
                    torch.tensor(tiled_goal).float(),
                    torch.tensor(horizon_vals).float()
                ), dim=1).to(self.device)

            logit_accessibilities = self.network.forward(x).detach().cpu().numpy()

            exploration_p = rng.uniform(low=0.0, high=1.0)
            if (not eval) and exploration_p < exploration_epsilon:
                n_actions = logit_accessibilities.shape[1]
                action_index = rng.choice(n_actions)
            else:

                accessibilities = 1. / (1 + np.exp(-logit_accessibilities))
                max_accessibility = accessibilities.max()
                filter_level = gamma * max_accessibility
                attainable_horizons = (accessibilities >= filter_level).any(axis=1)

                if (not (attainable_horizons).any()):
                    n_actions = accessibilities.shape[1]
                    action_index = rng.choice(n_actions)

                else:
                    # argmax only extracts 'True' values, and always returns the first.
                    min_attainable_horizon = attainable_horizons.argmax()
                    action_index = logit_accessibilities[min_attainable_horizon, :].argmax()

            return action_index

        return policy

    def run_episode(self, action_policy, ep_length=200,  use_eval_reset=False):

        if use_eval_reset:
            state, goal = self.env.reset(eval=True)
        else:
            state, goal = self.env.reset()

        ep_history = []
        for step in range(ep_length):

            selected_action = action_policy(state, goal)
            new_state, _, done, _ = self.env.step(selected_action)
            obs_goal = self.env.state_to_goal(new_state)

            # Store transition in episode buffer
            ep_history.append({"observation":state,
                               "action": selected_action,
                               "observation_next": new_state,
                               "achieved_goal":obs_goal,
                               "desired_goal":goal})

            # Update the state
            state = new_state

            if isinstance(self.env, MiniMazeEnv) or isinstance(self.env, PrecipiceEnv):
                goal_achieved = (state == goal).all()
            elif isinstance(self.env, DiscreteDubinsCar):
                goal_achieved = (np.linalg.norm(goal - state[:2], ord=np.inf) <= 0.5)

            if goal_achieved or done:
                break

        return ep_history, goal_achieved


    def learn(self, gde_ep, dataset, batch_size, batch_sampling_rng, o_norm, g_norm):

        # Sample shorter transitions towards the start of training and high transition towards the end
        horizon_sampling_probs = np.arange(1, 51) ** -(self.horizon_sampling_const *
                                                       (1.0 - gde_ep / self.goal_directed_eps)
                                                       )
        # sample minibatch
        states_sample, actions_sample, goals_sample, horizons_sample, s_prime_sample, goal_achieved = \
            sample_long_range_transitions(dataset, batch_size=batch_size, rng=batch_sampling_rng, env =self.env,
                                          horizon_sampling_probs=horizon_sampling_probs, her_fraction=self.HER_fraction, use_HER=self.use_HER)

        if self.train_step_count % self.target_network_copy_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        horizons_sample = horizons_sample.reshape((-1, 1))
        self.network.zero_grad()

        x_lhs = torch.cat((torch.tensor(states_sample).float(), torch.tensor(goals_sample).float(),
                           torch.tensor(horizons_sample).float()),
                          dim=1).to(self.device)

        logit_accessibilities_lhs = self.network.forward(x_lhs)[np.arange(len(horizons_sample)), actions_sample]

        with torch.no_grad():
            x_rhs = torch.cat((torch.tensor(s_prime_sample).float(), torch.tensor(goals_sample).float(),
                               torch.tensor(horizons_sample - 1).float()),
                              dim=1).to(self.device)
            accessibilities_rhs = self.target_network.forward(x_rhs).sigmoid()

            # double Q-learning
            ind_max = torch.argmax(self.network.forward(x_rhs), dim=1)

            accessibilities_rhs = accessibilities_rhs.gather(1, ind_max.unsqueeze(1))
            accessibilities_rhs = accessibilities_rhs.squeeze()

        # when h=1 or s_prim =g we are having a different target.
        for i in range(len(horizons_sample)):

            if self.clipping:
                if np.linalg.norm(goals_sample[i] - goal_achieved[i],
                                  ord=self.env.norm) > horizons_sample[i] - 1:
                    accessibilities_rhs[i] = 0.0

            if np.linalg.norm(goals_sample[i] - goal_achieved[i],
                              ord=np.inf) <= 0.5:
                accessibilities_rhs[i] = 1.0

            if horizons_sample[i] == 1:
                if np.linalg.norm(goals_sample[i] - s_prime_sample[i, :2],
                                  ord=np.inf) <= 0.5:
                    accessibilities_rhs[i] = 1.0
                else:
                    accessibilities_rhs[i] = 0.0

        loss_this_batch = self.training_loss(logit_accessibilities_lhs, accessibilities_rhs)
        loss_this_batch.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.train_step_count += 1