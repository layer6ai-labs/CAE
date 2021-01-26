import numpy as np
from typing import List


def sample_long_range_transitions(dataset: List, batch_size, rng: np.random.RandomState, env,
                                  horizon_sampling_probs: np.array = None, use_HER= True, her_fraction=0.8):


    prob = (np.arange(len(dataset)) + 1)

    ep_idx_sample = rng.choice(len(dataset), size=batch_size, replace=True, p=prob / np.sum(prob))
    ep_sample = [dataset[idx] for idx in ep_idx_sample]

    start_idx_sample = [
        rng.choice(len(ep)) for ep
        in ep_sample]

    if horizon_sampling_probs is None:
        h_sample = np.array([rng.choice(len(ep) - start_idx) + 1 for ep, start_idx in zip(ep_sample, start_idx_sample)])
    else:

        max_h = len(horizon_sampling_probs)
        h_sample = np.array([rng.choice(max_h, p=horizon_sampling_probs / horizon_sampling_probs.sum(), size=batch_size)+ 1])

    states_sample = np.array([ep[start_idx]["observation"] for ep, start_idx in zip(ep_sample, start_idx_sample)])
    actions_sample = np.array([ep[start_idx]["action"] for ep, start_idx in zip(ep_sample, start_idx_sample)])
    states_prime_sample = np.array([ep[start_idx]["observation_next"] for ep, start_idx in zip(ep_sample, start_idx_sample)])
    goals_achieved_sample = np.array([ep[start_idx]["achieved_goal"] for ep, start_idx in zip(ep_sample, start_idx_sample)])

    if use_HER:
        goals_achieved_idx_sample = [rng.choice(range(start, len(ep))) for ep, start in zip(ep_sample, start_idx_sample)]
        goals_achieved = np.array([ep[end_idx]["achieved_goal"] for ep, end_idx in zip(ep_sample, goals_achieved_idx_sample)])
        goals_desired = np.array([ep[0]["desired_goal"] for ep in ep_sample])

        s = rng.choice([0, 1], p=[her_fraction, 1 - her_fraction], size=batch_size)
        goals_sample = np.array([g1 if s1 == 0 else g2 for g1, g2, s1 in zip(goals_achieved, goals_desired, s)])
    else:
        end_idx_sample = []
        for i in range(batch_size):
            end_s = env.sample_goal_from_state(states_sample[i,:], h_sample[0][i], rng)
            end_idx_sample.append(end_s)

        goals_sample = np.array(end_idx_sample)

    return states_sample, actions_sample, goals_sample, h_sample, states_prime_sample, goals_achieved_sample
