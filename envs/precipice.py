import gym
import numpy as np


class PrecipiceEnv:
    """
    Discrete navigation environment with the following structure:
    0G000
    00000
    X0X00
    X0X00
    X0X00
    00000
    0S000
    where X represents a precipice and 0 a walkable surface. G and S are also walkable surfaces, and are the recommended
    states to showcase the tradeoff allowed by C-learning (as there is a safe policy which gets from S to G with
    probability 1). The agent will move in the intended direction with probability 0.8, and to each of the adjacent
    directions with probability 0.1. The coordinate (0,0) corresponds to the bottom left corner.
    """
    def __init__(self, max_steps=50, prob=0.8):
        self.width = 5
        self.height = 7
        self.max_steps = max_steps
        self.n_actions = 4  # 0, 1, 2, 3 correspond to up, right, down and left; respectively
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                               high=np.array([self.width, self.height]))
        self.state = None
        self.prob = prob
        self.rng = np.random.RandomState(0)
        self.reset_state = np.array([1,0])
        self.norm = 1
        self.d_obs = self.observation_space.shape[0]
        self.d_goal = 2
        self.n_actions = self.action_space.n
        self.discrete = True
        self.name = 'Precipice'


    def is_precipice(self, state):
        if (state[0] == 0 or state[0] == 2) and 2 <= state[1] <= 4:
            return True
        else:
            return False

    def is_wall(self, state):
        return not (
            0 <= state[0] < self.width and
            0 <= state[1] < self.height
        )

    def sample(self):
        not_done = True
        while not_done:
            x, y = self.rng.randint(low=0, high=[self.width, self.height])
            if not self.is_precipice((x, y)):
                not_done = False
        return np.array([x, y])

    def sample_goal_from_state(self, state, horizon, rng):
        used_h = rng.choice(horizon) + 1
        used_x = rng.choice(used_h)
        used_y = used_h - used_x
        used_x = rng.choice([-1, 1.]) * used_x
        used_y = rng.choice([-1, 1.]) * used_y
        sample = np.array([max(min(state[0] + used_x, self.width-1), 0), max(min(state[1] + used_y, self.height-1), 0)])
        return sample

    def uniform_sample_full_space(self, eval=False):
        goal_candidate = np.random.randint(low=0, high=[self.width, self.height])
        while eval and (goal_candidate == self.reset_state).all():
            goal_candidate = np.random.randint(low=0, high=[self.width, self.height])
        return goal_candidate

    def reset(self, eval=False):
        self.state = self.sample() if not eval else self.reset_state
        self.n_steps = 0
        self.goal = self.uniform_sample_full_space(eval=eval)
        return self.state, self.goal

    def _get_direction(self, action):
        if action == 0:
            return 0, 1
        elif action == 1:
            return 1, 0
        elif action == 2:
            return 0, -1
        elif action == 3:
            return -1, 0

    def state_to_goal(self, state):
        return state

    def _move(self, action):
        direction = self._get_direction(action)
        if not self.is_precipice(self.state):
            self.state = [max(min(self.state[0] + direction[0], self.width-1), 0), max(min(self.state[1] + direction[1], self.height-1), 0)]

    def step(self, action):
        assert self.state is not None, 'Cannot call env.step() before calling reset()'
        u = np.random.uniform()
        if u < self.prob:
            self._move(action)
        elif u < (1. + self.prob) / 2.:
            self._move((action + 1) % 4)
        else:
            self._move((action - 1) % 4)
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps) or self.is_precipice(self.state)
        return np.array(self.state), 0, done, {}
    def set_goal(self, goal):
        self.goal = goal