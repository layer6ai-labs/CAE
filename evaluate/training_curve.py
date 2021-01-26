import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate
import glob


OVERALL_RUNS_DIR = 'runs'
env_name = sys.argv[1]
runs_dir = f'{OVERALL_RUNS_DIR}/{env_name}'
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)
success_list = []
final_step_goal = []
final_success_rate = []
print(glob.glob(f'{runs_dir}/*'))
for run in glob.glob(f'{runs_dir}/*'):
    #loading eval data
    eval = np.load(os.path.join(run, 'eval.npy'), allow_pickle=True)
    success = eval[:,2]
    step_to_goal = eval[:,3]
    env_steps = eval[:,1]

    # calculating final success rate and step to goal based on the last 1000 evals
    final_success_rate.append(np.mean(success[-1000:]))
    final_step_goal.append(np.mean(step_to_goal[-1000:]))
    f = interpolate.interp1d(env_steps, success)
    step_count = np.arange(env_steps[0], env_steps[-1], 1)
    success = f(step_count)
    success_list.append(success)

#trim all curves based on minimum range
n = len(success_list[0])
for i in range(1, len(success_list)):
    n = min(n, len(success_list[i]))

for i in range(len(success_list)):
    success_list[i] = success_list[i][:n]

success_list = np.array(success_list)

#calc mean and std
su_mean = savgol_filter(np.mean(success_list, axis=0), 99999, 1)
su_std = savgol_filter(np.std(success_list, axis=0), 99, 1)
plt.figure()
plt.plot(range(n), su_mean, label="C-learning")
plt.fill_between(range(n), su_mean - su_std, su_mean + su_std, alpha=0.3, facecolor='blue')
plt.show()

#success rate
success_rate = np.array(final_success_rate)
print(np.mean(success_rate))
print(np.std(success_rate))

#path to goal
step_goal=np.array(final_step_goal)
print(np.mean(step_goal))
print(np.std(step_goal))
