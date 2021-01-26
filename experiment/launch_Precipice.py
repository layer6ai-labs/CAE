import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arguments import get_args
from algo.train import train
from envs.precipice import PrecipiceEnv

if __name__ == "__main__":
    args = get_args()
    args.hidden_layer_sizes = [60,  40]
    args.goal_directed_eps = 1000
    args.c_clliping = True
    args.use_HER = False
    env = PrecipiceEnv()
    train(params=vars(args), env=env)
