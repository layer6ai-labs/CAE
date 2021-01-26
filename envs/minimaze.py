import numpy as np
import gym
import pylab as plt
import os
import seaborn as sns


class MiniMazeEnv:
    width = 24
    height = 24
    walls_semi_thickness = 1
    gap_semi_size = 2

    @staticmethod
    def is_wall(position):
        x, y = position
        # Part of the vertical wall?
        if MiniMazeEnv.walls_semi_thickness - 1 >= x - MiniMazeEnv.width / 2 >= -MiniMazeEnv.walls_semi_thickness:
            # Are we in either of the "gaps"?
            if -MiniMazeEnv.gap_semi_size <= y - MiniMazeEnv.height / 4 <= MiniMazeEnv.gap_semi_size -1\
                    or -MiniMazeEnv.gap_semi_size <= y - 3 * MiniMazeEnv.height / 4 <= MiniMazeEnv.gap_semi_size -1:
                return False
            else:
                return True

        # Same logic for the horizontal wall
        if MiniMazeEnv.walls_semi_thickness - 1 >= y - MiniMazeEnv.height / 2 >= -MiniMazeEnv.walls_semi_thickness:
            # Are we in either of the "gaps"?
            if -MiniMazeEnv.gap_semi_size <= x - MiniMazeEnv.width / 4 <= MiniMazeEnv.gap_semi_size -1\
                    or -MiniMazeEnv.gap_semi_size <= x - 3 * MiniMazeEnv.width / 4 <= MiniMazeEnv.gap_semi_size -1:
                return False
            else:
                return True

    def __init__(self, stochastic_mode = False):
        self._player_pos = None
        # Put these here so that training code can query them like any other env.
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                               high=np.array([MiniMazeEnv.width, MiniMazeEnv.height]))
        self._stochastic_mode = stochastic_mode
        self._rng = np.random.RandomState(12345)
        self.reset_state = np.array([MiniMazeEnv.width//4, MiniMazeEnv.height//4])

        self.all_states = []
        for i in range(self.width):
            for j in range(self.height):
                if not self.is_wall([i,j]):
                    self.all_states.append([i,j])
        self.all_states = np.array(self.all_states)

        self.action_dict = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

        self.norm = 1
        self.d_obs = self.observation_space.shape[0]
        self.d_goal = 2
        self.n_actions = self.action_space.n
        self.discrete = True
        self.name = 'Minimaze'

    def reset(self):
        # Always reset the player to the same position
        self._player_pos = self.reset_state
        self.goal = self.uniform_sample_full_space()
        return self._player_pos, self.goal

    def state_to_goal(self, state):
        return state

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        if self._player_pos is None:
            raise AssertionError("Cannot call env.step() before calling reset()")

        try:
            delta = self.action_dict[action]
        except Exception as e:
            raise ValueError("Invalid action: {}".format(action))
        scale = 1 if not self._stochastic_mode else self._rng.choice([0, 1, 2], p=[0.05, 0.8, 0.15])
        delta = scale*delta
        candidate_pos = self._player_pos + delta
        # Is this outside the map bounds?
        if candidate_pos[0] < 0 or candidate_pos[0] >= MiniMazeEnv.width \
            or candidate_pos[1] < 0 or candidate_pos[1] >= MiniMazeEnv.height:
            # Invalid move, don't take the step
            pass
        elif MiniMazeEnv.is_wall(candidate_pos):
            # Invalid move, don't take the step
            pass
        else:
            # If we get here the move is valid, take the step
            self._player_pos = candidate_pos
        # Return in same format as a gym environment
        return np.array(self._player_pos), 0.0, False, {}

    def sample_goal_from_state(self, state, horizon, rng):
        prob = (np.arange(horizon+1)+1)**2
        cn = 0
        while cn < 15:
            cn += 1
            h1 = rng.choice(range(horizon+1),p=prob/np.sum(prob))
            h2 = rng.randint(low=-h1, high=h1+1)
            h3 = h1 - abs(h2)
            h3 = rng.choice([-1,1]) * h3
            end_s = state + np.array([h2, h3])
            if self.valid_goal(end_s):
                return end_s

        while True:
            search_h = min(horizon, self.width - self.reset_state[0])
            end_s = self.reset_state + rng.randint(low=-search_h, high=search_h+1, size=2)
            if self.valid_goal(end_s):
                return end_s

    def uniform_sample_full_space(self):
        while True:
            goal_candidate = np.random.randint(low=0, high=[MiniMazeEnv.width, MiniMazeEnv.height])
            if self.valid_goal(goal_candidate):
                return goal_candidate

    def valid_goal(self, goal):
        return (
            0 <= goal[0] < MiniMazeEnv.width and
            0 <= goal[1] < MiniMazeEnv.height and
            not MiniMazeEnv.is_wall(goal) and
            (self.reset_state != goal).any()
        )

    def goal_range_sample(self, goal_range, rng):
        cn = 0
        while cn < 30:
            cn += 1
            h1 = rng.randint(low=-goal_range, high=goal_range+1, size=1)
            h2 = (goal_range - abs(h1))*rng.choice([-1,1])
            goal_candidate = self.reset_state + np.array([h1, h2]).squeeze()

            if self.valid_goal(goal_candidate):
                return goal_candidate

        print("&&&&&")
        return self.uniform_sample_full_space(rng)

    def plot_trajectory(self, trajectory, save_as, goal, save_dir):
        pass

    def plot_policy(self, policy, save_as, goal, save_dir):
        maze_array = np.array(
            [
                [255 if MiniMazeEnv.is_wall((x, y)) else 0 for x in range(self.width)]
                for y in range(self.height)
            ]
        )
        plt.imshow(maze_array)
        plt.scatter(goal[0], goal[1], s=100, c='r')
        plt.scatter(self.reset_state[0], self.reset_state[1], s=100, c='g')

        policy_array = []
        for state in self.all_states:
            if (state == goal).all():
                # Don't plot policy arrow on goal
                continue
            action = policy(state, goal)
            policy_array.append(self.action_dict[action])

        x_grid_vals = np.array([s[0] for s in self.all_states if not (s == goal).all()])
        y_grid_vals = np.array([s[1] for s in self.all_states if not (s == goal).all()])
        x_delta = np.array([d[0] for d in policy_array])
        y_delta = np.array([-d[1] for d in policy_array]) # imshow inverts the y axis, thus the negation

        # Adjust the grid vals by one-third of the delta to "centre" the arrows
        plt.quiver(x_grid_vals - x_delta/3, y_grid_vals + y_delta/3, x_delta, y_delta, color='c', headwidth=10)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, save_as))
        plt.close()

    def plot_history_heatmap(self, ep_history, save_dir, burn_in=100, save_as='visitation_heatmap.png'):
        state_counts_array = np.zeros((self.width, self.height))

        for episode in ep_history[burn_in:]:
            for s, _, _, _, _ in episode:
                state_counts_array[s[0], s[1]] += 1
            _, _, s_final, _, _ = episode[-1]
            state_counts_array[s_final[0], s_final[1]] += 1

        sns.heatmap(state_counts_array)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, save_as))
        plt.close()


if __name__ == '__main__':
    # Displays the maze
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from time import sleep

    maze_arr = np.array([[255 if MiniMazeEnv.is_wall((x, y)) else 0for x in range(MiniMazeEnv.width)]
     for y in range(MiniMazeEnv.height)])
    plt.imshow(maze_arr)
    plt.show()