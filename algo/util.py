import numpy as np
import os
import torch

def updateTargets(target, original):
    """Weighted average update of the target network and original network
        Inputs: target actor(critic) and original actor(critic)"""
    tau = 0.05
    for targetParam, orgParam in zip(target.parameters(), original.parameters()):
        targetParam.data.copy_((1 - tau) * targetParam.data + \
                               tau * orgParam.data)
# clip the input
def preproc_og(o):
    o = np.clip(o, -200, 200)
    return o

# update parameters in the o_norm normalizer
def update_o_norm(o_norm, episode):
    input = []
    ep_steps = len(episode)
    for i in range(ep_steps):
        input.append(np.array(episode[i]["observation"].copy()).squeeze())

    input = preproc_og(np.array(input))
    o_norm.update(input)
    o_norm.recompute_stats()

# update parameters in the g_norm normalizer
def update_g_norm(env, g_norm):
    generated_goals = []
    for _ in range(10000):
        obs = env.reset()
        generated_goals.append(obs['desired_goal'].copy())
    generated_goals_clip = preproc_og(np.array(generated_goals))
    g_norm.update(generated_goals_clip)
    g_norm.recompute_stats()

# save the results
def save_results(run_dir, eval_pts, eval_goal_list, eval_goal_success, eval_goal_step, env_steps, critic1, critic2,
                 actor, o_norm, g_norm):

    eval_array = np.stack((eval_pts, env_steps ,eval_goal_success, eval_goal_step), axis=1)
    np.save(os.path.join(run_dir, 'eval.npy'), eval_array)
    np.save(os.path.join(run_dir, 'goal.npy'), eval_goal_list)

    torch.save(critic1.state_dict(), os.path.join(run_dir, 'critic1.pt'))
    torch.save(critic2.state_dict(), os.path.join(run_dir, 'critic2.pt'))
    torch.save(actor.state_dict(), os.path.join(run_dir, 'actor.pt'))
    if not(o_norm is None):
        np.save(os.path.join(run_dir, 'o_norm.npy'), [o_norm.mean, o_norm.std])
        np.save(os.path.join(run_dir, 'g_norm.npy'), [g_norm.mean, g_norm.std])


