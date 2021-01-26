import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import os
from algo.models import ActorNetwork, CNetworkCont
import gym
from gym.wrappers import Monitor
import glob

# normalize the input
def normalize(v, n):
    return np.clip((v - n[0]) / (n[1]), -5, 5)

# policy
def goal_conditioned_c_learning_policy(critic: nn.Module, actor: nn.Module, rng: np.random.RandomState, device,
                                       action_dim, horizon=None, noise_epsilon=0.2, exploration_epsilon=0.2,
                                       eval=True, g_norm=None, o_norm=None):

    max_horizon = horizon if horizon is not None else 50
    horizon_vals = np.array([i+1 for i in range(max_horizon)]).reshape((-1, 1))

    def policy(state, goal):
        state = normalize(state, o_norm)
        goal = normalize(goal, g_norm)

        tiled_state = np.tile(state, max_horizon).reshape((max_horizon, -1))
        tiled_goal = np.tile(goal, reps=max_horizon).reshape((max_horizon, -1))

        x_action = torch.cat(
            (torch.tensor(tiled_state).float(), torch.tensor(tiled_goal).float(), torch.tensor(horizon_vals).float()),
            dim=1).to(device)

        exploration_p = rng.uniform(low=0.0, high=1.0)

        if exploration_p < exploration_epsilon and not eval:
            a = rng.uniform(-1, 1, size=action_dim)
        else:
            actions = actor.forward(x_action)
            if eval:
                action_noise = actions
            else:
                n = noise_epsilon * np.random.randn(action_dim)
                tiled_noise = np.tile(n, max_horizon).reshape((max_horizon, -1))
                tiled_noise = torch.tensor(tiled_noise).float().to(device)
                action_noise = actions + tiled_noise
                action_noise = torch.clamp(action_noise, -1, 1)

            x = torch.cat((x_action, torch.tensor(action_noise).float()), dim=1)
            accessibilities = critic.forward(x).detach().cpu().numpy()
            max_accessibility = accessibilities.max()
            filter_level = 0.9 * max_accessibility
            attainable_horizons = (accessibilities >= filter_level).any(axis=1)
            min_attainable_horizon = attainable_horizons.argmax()
            a = action_noise[min_attainable_horizon].detach().cpu().numpy()
        return a
    return policy



OVERALL_RUNS_DIR = 'runs'
env_name = sys.argv[1]
runs_dir = f'{OVERALL_RUNS_DIR}/{env_name}'
runs = glob.glob(f'{runs_dir}/*')

# create environment
env2 = gym.make(env_name)
env = gym.make(env_name)

# set the seed
env.seed(1)

# environment properties
obs = env2.reset()
goal = obs['desired_goal']
d_obs = obs['observation'].shape[0]
d_actions = env.action_space.shape[0]
d_goal = obs['desired_goal'].shape[0]
action_max = env.action_space.high[0]

# create the networks
critic1 = CNetworkCont(d_obs=d_obs, d_goal=d_goal, d_actions=d_actions,
                               hidden_layer_sizes=[600, 600, 600])
actor = ActorNetwork(d_obs=d_obs, d_goal=d_goal, d_actions=d_actions, hidden_layer_sizes=[600, 600, 600])

# load the saved networks
critic1.load_state_dict(torch.load(os.path.join(runs[0], 'critic1.pt'), map_location=torch.device('cpu')))
actor.load_state_dict(torch.load(os.path.join(runs[0], 'actor.pt'), map_location=torch.device('cpu')))

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the rng
exploration_rng = np.random.RandomState(2)

# load mean std of the normalized values
o_norm=np.load(os.path.join(runs[0], 'o_norm.npy'))
g_norm=np.load(os.path.join(runs[0], 'g_norm.npy'))

# call the policy
action_policy = goal_conditioned_c_learning_policy(critic1, actor, exploration_rng, device=device, action_dim=d_actions,
                                                                eval=True, g_norm=g_norm, o_norm=o_norm)

# generate 10 simulations
for i in range(10):
    env = Monitor(env, (os.path.join(runs[0], 'recording2')))
    observation = env.reset()
    for step in range(100):
        selected_action = action_policy(observation['observation'], observation['desired_goal'])
        new_observation, reward, done, _ = env.step(selected_action)
        observation = new_observation
        if done:
            break





