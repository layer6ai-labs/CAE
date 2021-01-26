import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arguments import get_args
from algo.train import train
from envs.minimaze import MiniMazeEnv

if __name__ == "__main__":
    args = get_args()
    args.hidden_layer_sizes = [200,  100]
    args.goal_directed_eps = 3000
    args.max_ep_len = 120
    args.c_clliping = True
    env = MiniMazeEnv()
    train(params=vars(args), env=env)
