import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arguments import get_args
from algo.train import train
from envs.dubin_maze import DiscreteDubinsCar


if __name__ == "__main__":
    args = get_args()
    args.hidden_layer_sizes = [400,  300]
    args.goal_directed_eps = 5000
    args.max_ep_len = 100
    env = DiscreteDubinsCar(turning_angles=[-10, 0, 10], max_turns=100)
    train(params=vars(args), env=env)





