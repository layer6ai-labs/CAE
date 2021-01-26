import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from envs.gym_wrapper import BasicWrapper
import gym
from algo.train import train
from arguments import get_args

if __name__ == "__main__":
    args = get_args()
    args.hidden_layer_sizes = [600, 600, 600]
    args.target_network_copy_freq = 1
    env = gym.make('FetchPickAndPlace-v1', reward_type='sparse')
    env = BasicWrapper(env)
    train(params=vars(args), env=env)