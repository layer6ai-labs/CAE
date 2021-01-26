import cmath
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class DiscreteDubinsCar:
    def __init__(self, turning_angles, max_turns=200, cv_available=True, allow_no_op=True, allow_moving_backwards=True):
        self.turning_angles = turning_angles
        self.cv_available = cv_available
        self.allow_no_op = allow_no_op
        self.allow_moving_backwards = allow_moving_backwards
        self.turning_multipliers = [np.cos(theta * np.pi / 180) + np.sin(theta * np.pi / 180) * 1j for theta in
                                    self.turning_angles]
        self.max_turns = max_turns
        self.n_actions = 7
        self.no_op_action = len(self.turning_angles)
        self.move_backwards = self.no_op_action + 1
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=np.array([-np.inf, -np.inf, -1, -1]),
                                                high=np.array([np.inf, np.inf, 1, 1]))
        self.location = None
        self.velocity = None
        self.trajectory = []
        self.reset_state = self.descartes_coordinates(self.sample_location())
        self.norm = np.inf
        self.d_obs = self.observation_space.shape[0]
        self.d_goal = 2
        self.n_actions = self.action_space.n
        self.discrete = True
        self.name ='DubinsCar'



    @property
    def car_x(self):
        return self.location.real

    @property
    def car_y(self):
        return self.location.imag

    @property
    def car_vx(self):
        return self.velocity.real

    @property
    def car_vy(self):
        return self.velocity.imag

    @property
    def car_angle(self):
        return 180 * cmath.phase(self.velocity) / np.pi

    @property
    def state(self):
        return self.car_x, self.car_y, self.car_vx, self.car_vy

    @staticmethod
    def descartes_coordinates(z):
        return z.real, z.imag

    @staticmethod
    def sample_location(location_ranges=(0, 1, 0, 1)):
        x_min, x_max, y_min, y_max = location_ranges
        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        return x + y * 1j

    @staticmethod
    def sample_velocity(candidate_angles=None):
        if candidate_angles is not None:
            theta = np.random.choice(candidate_angles) * np.pi / 180
        else:
            theta = np.random.randint(0, 2 * np.pi)
        return np.cos(theta) + np.sin(theta) * 1j

    width = 15
    height = 15


    @staticmethod
    def is_wall(position):
        x, y = position

        if 4 <= x <= 6 :
            if -1 <= y <= 6 or 10<=y<=16:
                return True

        if 13 >= x  >= 10:
            if 3 <= y <= 5:
                return True

        if 9 <= x <= 12:
            if 11 <= y <= 13:
                return True

        if x < 0 or x > 15:
            return True

        if y < 0 or y > 15:
            return True

        return False

    def sample(self, initial_pos_range=(0, 1 , 0, 1), candidate_angles=None, return_state=False):
        location = self.sample_location(initial_pos_range)
        velocity = self.sample_velocity(candidate_angles)
        if return_state:
            x, y = self.descartes_coordinates(location)
            vx, vy = self.descartes_coordinates(velocity)
            return x, y, vx, vy
        return location, velocity

    def sample_goal_from_state(self, state, horizon, rng):
        # choose a goal with higher probability toward the boundary of h
        position = state[:2]
        s1 = np.floor(position + [0.5, 0.5])
        prob = (np.arange(horizon + 1) + 1) ** 2
        cn = 0
        while cn < 15:
            cn = cn + 1
            r = rng.choice(range(horizon + 1), p=prob / np.sum(prob))
            theta = rng.uniform(0, 1) * 2 * np.pi
            h1 = np.floor(r * np.cos(theta))
            h2 = np.floor(r * np.sin(theta))
            end_s = s1 + np.array([h1, h2])
            if not DiscreteDubinsCar.is_wall(end_s) and 0 <= end_s[0] < DiscreteDubinsCar.width \
                    and 0 <= end_s[1] < DiscreteDubinsCar.height:
                break
        if cn == 15:
            def goal_sample_eval():
                return rng.randint(low=0, high=[DiscreteDubinsCar.width, DiscreteDubinsCar.height])

            cn = 0
            while cn < 15:
                cn = cn + 1
                end_s = goal_sample_eval()
                if not DiscreteDubinsCar.is_wall(end_s) and 0 <= end_s[0] < DiscreteDubinsCar.width \
                        and 0 <= end_s[1] < DiscreteDubinsCar.height:
                    break

        return end_s

    def uniform_sample_full_space(self):
        while True:
            goal_candidate = np.random.randint(low=0, high=[DiscreteDubinsCar.width, DiscreteDubinsCar.height])
            if self.valid_goal(goal_candidate):
                return goal_candidate

    def goal_range_sample(self, goal_range, rng):
        cn = 0
        while cn < 30:
            cn = cn + 1

            theta = rng.uniform(0, 1) * 2 * np.pi
            h1 = np.floor(goal_range * np.cos(theta))
            h2 = np.floor(goal_range * np.sin(theta))

            goal_candidate = self.reset_state + np.array([h1, h2]).squeeze()

            if self.valid_goal(goal_candidate):
                return goal_candidate

        print("&&&&&")
        return self.uniform_sample_full_space(rng)

    def valid_goal(self, goal):
        return (
            (self.reset_state != goal).any()
            and not DiscreteDubinsCar.is_wall(goal)
            and 0 <= goal[0] < DiscreteDubinsCar.width
            and 0 <= goal[1] < DiscreteDubinsCar.height
        )

    def set_goal(self, goal):
        self.goal = goal

    def reset(self, initial_pos_range=(0, 1, 0, 1), candidate_angles=[0]):
        self.location, self.velocity = self.sample(initial_pos_range, candidate_angles)
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.n_turns = 0
        self.goal = self.uniform_sample_full_space()
        return self.state, self.goal

    def state_to_goal(self, state):
        return state[:2]

    def step(self, action, verbose=False):
        assert self.location is not None, 'Cannot call env.step() before calling reset()'

        previous_location = self.location
        previous_velocity = self.velocity
        candidate_pos = self.location

        if action == 6:
            pass
        elif action//3==0:
            self.velocity *= self.turning_multipliers[action%3]
            candidate_pos -= self.velocity
        elif action//3==1:
            self.velocity *= self.turning_multipliers[action%3]
            candidate_pos += self.velocity
        self.n_turns += 1

        x,y = self.descartes_coordinates(self.location)

        # when the dubins car hit the wall, or out of bound


        x1, y1 = self.descartes_coordinates(candidate_pos)


        if DiscreteDubinsCar.is_wall([x1, y1]):
            n_seg = 5
            x_diff = (x1 - x)/n_seg
            y_diff = (y1 - y)/n_seg
            x_old = x
            y_old = y
            for s in range(n_seg):
                x_new = x_diff + x_old
                y_new = y_diff + y_old
                if DiscreteDubinsCar.is_wall([x_new, y_new]):
                    break
                x_old = x_new
                y_old = y_new
            if not DiscreteDubinsCar.is_wall([x_new, y_old]):
                x_old = x_new
                for s1 in range(s+1,n_seg):
                    x_new = x_old + x_diff
                    if DiscreteDubinsCar.is_wall([x_new, y_old]):
                        break
                    x_old = x_new
                candidate_pos = x_old + y_old * 1j
            elif not DiscreteDubinsCar.is_wall([x_old, y_new]):
                y_old = y_new
                for s1 in range(s+1, n_seg):
                    y_new = y_old + y_diff
                    if DiscreteDubinsCar.is_wall([x_old, y_new]):
                        break
                    y_old = y_new
                candidate_pos = x_old + y_old * 1j
            else:
                candidate_pos = x_old + y_old * 1j



        self.location = candidate_pos

        done = (self.n_turns >= self.max_turns)
        if verbose:
            v_before = self.descartes_coordinates(previous_velocity)
            v_after = self.descartes_coordinates(self.velocity)
            pos_before = self.descartes_coordinates(previous_location)
            pos_after = self.descartes_coordinates(self.location)
            angle = self.turning_angles[action]
            print('angle = {}, velocity: {} -> {}, location: {} -> {}'.format(angle, v_before, v_after,
                                                                              pos_before, pos_after))
        self.trajectory.append(self.state)
        return self.state, 0, done, {}

    def plot_trajectory(self, trajectory, save_as, goal, save_dir):
        pixels_per_unit = 50
        image_padding = 10
        plot_color = (0, 255, 0)  # blue
        height = int(round(pixels_per_unit * 15)) + image_padding * 2
        width = int(round(pixels_per_unit * 15)) + image_padding * 2
        image = 255 * np.ones((height, width, 3), dtype=np.uint8)
        for i in range(len(trajectory) - 1):
            start_x, start_y = trajectory[i]["observation"][:2]
            orient_x, orient_y = trajectory[i]["observation"][2:4]
            end_x, end_y = trajectory[i]["observation_next"][:2]
            start_x = int(round((start_x ) * pixels_per_unit)) + image_padding
            start_y = int(round((start_y ) * pixels_per_unit)) + image_padding
            end_x = int(round((end_x ) * pixels_per_unit)) + image_padding
            end_y = int(round((end_y ) * pixels_per_unit)) + image_padding
            cv2.line(image, (start_x, start_y), (end_x, end_y), color=plot_color, thickness=2)
            color = (255, 0, 0)
            cv2.rectangle(image, (4 * pixels_per_unit + image_padding,  image_padding),
                        (6 * pixels_per_unit + image_padding, 6 * pixels_per_unit + image_padding), color=color,
                        thickness=-1)
            cv2.rectangle(image, (4 * pixels_per_unit + image_padding,  10 * pixels_per_unit +image_padding),
                        (6 * pixels_per_unit + image_padding, 15 * pixels_per_unit + image_padding), color=color,
                        thickness=-1)
            cv2.rectangle(image, (9 * pixels_per_unit + image_padding,  11 * pixels_per_unit +image_padding),
                        (12 * pixels_per_unit + image_padding, 13 * pixels_per_unit + image_padding), color=color,
                        thickness=-1)
            cv2.rectangle(image, (10 * pixels_per_unit + image_padding,  3 * pixels_per_unit +image_padding),
                        ( 13* pixels_per_unit + image_padding, 5 * pixels_per_unit + image_padding), color=color,
                        thickness=-1)

            color = (255, 255, 0)
            cv2.circle(image, (goal[0] * pixels_per_unit + image_padding, goal[1] * pixels_per_unit + image_padding),
                    radius=10, thickness=-1, color=color)

            color = (0, 0, 255)
            thickness = 4
            cv2.arrowedLine(image, (start_x, start_y), (int((start_x + orient_x * 35)), int((start_y + orient_y * 35))),
                            color, thickness)

        cv2.imwrite(os.path.join(save_dir, save_as), image)


def plot_distance(env, save_as, n_episodes=1000, max_dist=100):
    from tqdm import tqdm
    distance = 1000 * np.ones((201, 201))
    distance[100, 100] = 0
    for i in tqdm(range(n_episodes)):
        env.reset( candidate_angles=[0])
        while True:
            action = np.random.choice(range(7))

            state, reward, done, _ = env.step(action)

            state_x, state_y = state[:2]
            x = int(round(state_x)) + 100
            y = int(round(state_y)) + 100
            if 0 <= x < 201 and 0 <= y < 201:
                distance[y, x] = min(distance[y, x], env.n_turns)
            if done:
                break
    img = np.zeros((804, 804, 3))
    for i in range(201):
        for j in range(201):
            d = distance[j, i]
            if d < max_dist:
                value = (max_dist - d) / max_dist
                img[j*4:j*4+4, i*4:i*4+4, :] = 0, value, value
    img[400:404, 400:404, :] = 1, 0, 0
    plt.figure(num=None, figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(save_as)
    plt.close()


def plot_trajectory(env, save_as):
    env.reset()
    action = 0
    for i in range(1000):
        r = np.random.rand()
        if r > 0.5:
            action = np.random.choice(range(env.n_actions))
        env.step(action, verbose=False)

    env.plot_trajectory(save_as=save_as)


if __name__ == '__main__':
    max_dist = 400
    dubin_15 = DiscreteDubinsCar(turning_angles=[-15, 0, 15], max_turns=max_dist)
    plot_trajectory(dubin_15, 'trajectory_15.png')
    plot_distance(dubin_15, 'dist_15.png', n_episodes=1000, max_dist=max_dist)
    dubin_5 = DiscreteDubinsCar(turning_angles=[-5, 0, 5], max_turns=max_dist)
    plot_trajectory(dubin_5, 'trajectory_5.png')
    plot_distance(dubin_5, 'dist_5.png', n_episodes=1000, max_dist=max_dist)

