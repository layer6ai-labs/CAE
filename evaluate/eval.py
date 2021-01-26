#!/usr/bin/env python
import torch
import glob
import json
import os
import numpy as np
import torch.nn as nn
import tqdm
import copy
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from algo.models import CNetworkDisc
from envs.dubin_maze import DiscreteDubinsCar
from envs.minimaze import MiniMazeEnv
from envs.precipice import PrecipiceEnv
from evaluate.eval_cont import conv_eval
MAX_TRAJ_DIST = 100
DUBINS_ANGLES = [-10, 0, 10]


def eval_discrete():
    def get_goal_set(top_left, bottom_right):
        valid_goals = []
        for i in np.arange(top_left[0], bottom_right[0]):
            for j in np.arange(top_left[1], bottom_right[1]):
                if not env.is_wall([i, j]):
                    valid_goals.append([i, j])
        return valid_goals

    easy_goals = get_goal_set(*goal_bounds['EASY_BOUNDARIES'])
    easy_goals.remove(list(env.reset_state))
    medium_goals = get_goal_set(*goal_bounds['MEDIUM_BOUNDARIES_1']) + get_goal_set(*goal_bounds['MEDIUM_BOUNDARIES_2'])
    hard_goals = get_goal_set(*goal_bounds['HARD_BOUNDARIES'])
    all_goals = easy_goals + medium_goals + hard_goals
    goal_dict = {
        'easy': easy_goals,
        'medium': medium_goals,
        'hard': hard_goals,
        'all': all_goals
    }


    def evaluate_episode_hist(episode, goal, metric):
        position_hist = np.array([env.state_to_goal(ep["observation_next"]) for ep in episode])
        goal_dist = np.linalg.norm(position_hist[-1] - goal, ord=env.norm)

        if metric == 'final_dist':
            return goal_dist

        success = 100 if goal_dist < 0.5 else 0
        if metric == 'success_rate':
            return success

        if metric == 'path_length_of_success':
            if success == 0:
                return np.nan

            position_prime_hist = np.array([env.state_to_goal(ep["observation"]) for ep in episode_hist])
            travel_dists = np.linalg.norm(position_hist - position_prime_hist, ord=env.norm, axis=1)
            return np.sum(travel_dists)

        # If we get here, we've asked for a metric that isn't implemented
        raise ValueError(metrics)

    rng = np.random.RandomState()  # Required for the policy function, even when policy is deterministic
    learning_methods = ['c-learning']
    metrics = ['success_rate', 'path_length_of_success']
    full_eval_table = {}
    aggregated_eval_table = {}
    all_runs = []

    for learning_method in learning_methods:
        print(f'Evaluating {learning_method}')
        full_eval_table[
            learning_method] = []  # For each run, will have a dictionary of metrics info. Store each dict in this list

        model_class = CNetworkDisc
        network = model_class(
            d_obs=env.d_obs,
            d_goal=env.d_goal,
            n_actions=env.action_space.n,
            hidden_layer_sizes=hidden_layer_sizes
        ).to(device)


        for run in glob.glob(f'{runs_dir}/*'):
            if not os.path.isdir(run):
                continue
            try:
                with open(f'{run}/config.json') as f:
                    config = json.load(f)
                network.load_state_dict(torch.load(f'{run}/critic1.pt'))
                network.eval()
            except FileNotFoundError:
                continue

            print(f'\tRun {run}')
            all_runs.append(run)

            all_metrics_dict = {}
            all_metrics_dict_unrolled = {}
            for metric in metrics:
                all_metrics_dict[metric] = {}
                all_metrics_dict_unrolled[metric] = []

            for difficulty in goal_dict:
                print(f'\t\tDifficulty {difficulty}')
                goal_list = goal_dict[difficulty]

                if difficulty == 'all':
                    # To avoid computing the same stuff again - saves time and avoids stochasticity
                    for metric in metrics:
                        all_metrics_dict[metric][difficulty] = np.nanmean(all_metrics_dict_unrolled[metric])

                    continue

                inner_metrics_dict = {}
                for metric in metrics: inner_metrics_dict[metric] = []

                for goal in tqdm.tqdm(goal_list):
                    for _ in range(num_runs_per_goal):
                        goal = np.array(goal)

                        zeroing = not (env_name == "DubinsCar")
                        policy = goal_conditioned_c_learning_policy(network, rng, device=device,
                                                                        norm=env.norm, eval=True, zeroing=zeroing)

                        episode_hist = run_episode(
                            env=env,
                            action_policy=policy,
                            ep_length=config['max_ep_len'],
                            stop_on_done=True,
                            goal=goal,
                            use_eval_reset=(env_name == 'Precipice')
                        )

                        fig_name = f'traj_{goal[0]}_{goal[1]}.png'
                        save_dir = os.path.join(run, 'goal_trajectories')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        if env_name == 'DubinsCar':
                            env.plot_trajectory(
                                trajectory=episode_hist,
                                save_as=fig_name,
                                goal=goal,
                                save_dir=save_dir
                            )
                        elif env_name == 'Minimaze':
                            env.plot_policy(
                                policy=policy,
                                save_as=fig_name,
                                goal=goal,
                                save_dir=save_dir
                            )

                        for metric in metrics:
                            metric_value = evaluate_episode_hist(episode_hist, goal, metric)
                            inner_metrics_dict[metric].append(metric_value)
                            all_metrics_dict_unrolled[metric].append(metric_value)

                for metric in metrics:
                    all_metrics_dict[metric][difficulty] = np.nanmean(inner_metrics_dict[metric])

            full_eval_table[learning_method].append(all_metrics_dict)

            single_run_table = copy.deepcopy(all_metrics_dict)
            for metric in metrics:
                for difficulty in goal_dict:
                    single_mean = single_run_table[metric][difficulty]
                    single_run_table[metric][difficulty] = f'{single_mean:.2f}'

            with open(f'{run}/metrics.json', 'w') as f:
                f.write(json.dumps(single_run_table, indent=4))

            if env_name == "Minimaze":
                try:
                    all_episodes = np.load(os.path.join(run, "dataset.npy"), allow_pickle=True)
                    env.plot_history_heatmap(all_episodes, run)
                except ValueError:
                    print('Episode in old format, skipping the heatmap')
                except FileNotFoundError:
                    print('Dataset not saved, skipping the heatmap')

        aggregated_eval_table[learning_method] = {}
        for metric in metrics:
            aggregated_eval_table[learning_method][metric] = {}

            for difficulty in goal_dict:
                metric_list = []

                for run in full_eval_table[learning_method]:
                    metric_list.append(run[metric][difficulty])

                metric_mean = np.nanmean(metric_list)
                metric_std = np.nanstd(metric_list)

                aggregated_eval_table[learning_method][metric][difficulty] = f'{metric_mean:.2f} +/- {metric_std:.2f}'

    with open(f'{runs_dir}/eval_table.json', 'w') as f:
        f.write(json.dumps(aggregated_eval_table, indent=4))
        f.write(f'\nRuns:{all_runs}')

def run_episode(env, action_policy, ep_length=200, stop_on_done=True, goal=np.array([0, 0]), use_eval_reset=False):
    """
    Runs an episode and returns the history of it.
    :param env: Environment. Must have a reset() and step(a) method.
    :param action_policy: Mapping from state to action. May be stochastic.
    :param ep_length: Maximum number of transitions to run for.
    :param stop_on_done: If False will ignore the done flag returned from step().
    :return: List of (prev_state, action, new_state) tuples.
    """
    if use_eval_reset:
        state, _ = env.reset(eval=True)
    else:
        state, _ = env.reset()

    ep_history = []
    env.set_goal(goal)
    for step in range(ep_length):

        selected_action = action_policy(state, goal)
        new_state, _, done, _ = env.step(selected_action)
        obs_goal = env.state_to_goal(new_state)

        # Store transition in episode buffer
        ep_history.append({"observation": state,
                           "action": selected_action,
                           "observation_next": new_state,
                           "achieved_goal": obs_goal,
                           "desired_goal": goal})

        # Update the state
        state = new_state

        if isinstance(env, MiniMazeEnv) or isinstance(env, PrecipiceEnv):
            goal_achieved = (state == goal).all()
        elif isinstance(env, DiscreteDubinsCar):
            goal_achieved = (np.linalg.norm(goal - state[:2], ord=np.inf) <= 0.5)

        if goal_achieved and stop_on_done:
            break
        if done:
            break

    return np.array(ep_history)

def goal_conditioned_c_learning_policy(network: nn.Module, rng: np.random.RandomState, device, norm,
                                       horizon=None, exploration_epsilon=0.15, eval=True, gamma=0.9, zeroing=True):

    max_horizon = horizon if horizon is not None else 50

    def policy(state, goal):
        if zeroing:
            # state[:2] is hacky but works for now in these two dimensional environments
            min_horizon = np.floor(np.linalg.norm(goal-state[:2], ord=norm))
        else:
            min_horizon = 1

        min_horizon = min(min_horizon, max_horizon)     # Ensures at least one horizon
        horizon_vals = np.arange(min_horizon, max_horizon+1).reshape((-1,1))
        num_horizons = len(horizon_vals)

        tiled_goal = np.tile(goal, num_horizons).reshape((num_horizons, -1))
        tiled_state = np.tile(state, num_horizons).reshape((num_horizons, -1))

        x = torch.cat(
            (
                torch.tensor(tiled_state).float(),
                torch.tensor(tiled_goal).float(),
                torch.tensor(horizon_vals).float()
            ), dim=1).to(device)

        logit_accessibilities = network.forward(x).detach().cpu().numpy()

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

                min_attainable_horizon = attainable_horizons.argmax()
                action_index = logit_accessibilities[min_attainable_horizon, :].argmax()

        return action_index

    return policy

OVERALL_RUNS_DIR = 'runs'

DUBINS_GOAL_BOUNDARIES = {
    'EASY_BOUNDARIES': [[0,0], [4,10]],
    'MEDIUM_BOUNDARIES_1': [[0,10], [4,16]],
    'MEDIUM_BOUNDARIES_2': [[4,0], [9,16]],
    'HARD_BOUNDARIES': [[9,0], [16,16]]
}
MINIMAZE_GOAL_BOUNDARIES = {
    'EASY_BOUNDARIES': [[0,0], [12,12]],
    'MEDIUM_BOUNDARIES_1': [[0,12], [12,24]],
    'MEDIUM_BOUNDARIES_2': [[12,0], [24,12]],
    'HARD_BOUNDARIES': [[12,12], [24,24]]
}
PRECIPICE_GOAL_BOUNDARIES = {
    'EASY_BOUNDARIES': [[0,0], [5,7]],
    'MEDIUM_BOUNDARIES_1': [[0,0], [0,0]],
    'MEDIUM_BOUNDARIES_2': [[0,0], [0,0]],
    'HARD_BOUNDARIES': [[0,0], [0,0]]
}

env_name = sys.argv[1]
runs_dir = f'{OVERALL_RUNS_DIR}/{env_name}'

#device
device = torch.device('cpu')

if env_name == 'DubinsCar':
    env = DiscreteDubinsCar(turning_angles=DUBINS_ANGLES, max_turns=MAX_TRAJ_DIST)
    goal_bounds = DUBINS_GOAL_BOUNDARIES
    hidden_layer_sizes = [400, 300]
    num_runs_per_goal = 1
    eval_discrete()
elif env_name == 'Minimaze':
    env = MiniMazeEnv()
    goal_bounds = MINIMAZE_GOAL_BOUNDARIES
    hidden_layer_sizes = [200, 100]
    num_runs_per_goal = 1
    eval_discrete()
elif env_name == 'Precipice':
    env = PrecipiceEnv()
    goal_bounds = PRECIPICE_GOAL_BOUNDARIES
    hidden_layer_sizes = [60, 40]
    num_runs_per_goal = 20
    eval_discrete()
elif env_name == 'HandManipulatePen-v0' or env_name == 'FetchPickAndPlace-v1':
    conv_eval(env_name)

