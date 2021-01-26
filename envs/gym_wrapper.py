import gym

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        obs = self.env.reset()
        self.discrete = False
        self.d_obs = obs['observation'].shape[0]
        self.d_actions = env.action_space.shape[0]
        self.d_goal = obs['desired_goal'].shape[0]
        self.action_max = env.action_space.high[0]
        self.name = self.env.unwrapped.spec.id
