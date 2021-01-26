import argparse


def get_args():

    parser = argparse.ArgumentParser(
        description='Trains a goal-conditioned policy using the method of arxiv.org/1912.06088')
    parser.add_argument('-s', '--overall-seed', type=int, default=21,
                        help='overall seed to use for training - all seeds derive from this.')
    parser.add_argument('--goal-evaluation-seed', type=int, default=100,
                        help='random seed to use for goal evaluation')
    parser.add_argument('--target-network-copy-freq', type=int, default=4,
                        help='Frequency with which we update the target network to match the model network.')
    parser.add_argument('--goal-achievement-epsilon', type=float, default=0.0,
                        help='Used in closure condition in A-learning at h=0.')
    parser.add_argument('--random-exploration-eps', type=int, default=15,
                        help='Number of random exploration rollouts to perform at the start')
    parser.add_argument('--goal-directed-eps', type=int, default=100000,
                        help='Number of rollouts to perform using the (partially trained) policy.')
    parser.add_argument('--max-ep-len', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size to use for training.')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate to use in training the policy network.')
    parser.add_argument('--critic-learning-rate', type=float, default=1e-4,
                        help='Learning rate to use in training the policy network.')
    parser.add_argument('--actor_learning-rate', type=float, default=1e-4,
                        help='Learning rate to use in training the policy network.')
    parser.add_argument('--train-steps-per-ep', type=int, default=80,
                        help='Number of training steps to perform each time we run an episode.')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='frequency (in terms of episodes) of evaluation')
    parser.add_argument('--num-eval-goals', type=int, default=1,
                        help='number of goals to use for evaluation')
    parser.add_argument('--policy-freq', type=int, default=2,
                        help='number of goals to use for evaluation')
    parser.add_argument('--noise-eps', type=float, default=0.3,
                        help='amount of noise added to the actions')
    parser.add_argument('--exploration-eps', type=float, default=0.3,
                        help='probability of random exploration')
    parser.add_argument('--horizon-sampling-const', type=float, default=2.3,
                        help='constant for choosing horizon to sample')
    parser.add_argument('--use-LR-scheduler', action='store_true',
                        help='use a learning rate that drops at some point')
    parser.add_argument('--LR-change-episode', type=int, default=2000,
                        help='epoch from which to change the learning rate')
    parser.add_argument('--LR-reduce-factor', type=float, default=10,
                        help='factor to reduce the learning rate at LR-change-episode')
    parser.add_argument('--use-decaying-epsilon', action='store_true',
                        help='use a decaying exploration rate')
    parser.add_argument('--epsilon-decay-denominator', type=int, default=150,
                        help='i nverse speed of epsilon decay in exploration')
    parser.add_argument('--use-HER', type=bool, default=True,
                        help='use HER-style relabelling')
    parser.add_argument('--HER-fraction', type=float, default=0.8,
                        help='fraction of goals to relabel in HER')
    parser.add_argument('--hidden-layer-sizes', type=list, default=[20, 20],
                        help='hidden layer sizes')
    parser.add_argument('--plot-freq', type=int, default=20,
                        help='frequency of plotting trajectories, should be a multiple of eval-freq')
    parser.add_argument('--c-clipping', type=bool, default=False,
                        help='Indicate whether to perform c-function cliping')

    args = parser.parse_args()

    return args


