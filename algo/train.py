import torch
import numpy as np
from time import time
import os
import datetime
import json
from algo.C_learning_Discrete import CLearningDiscrete
from algo.C_learning_TD3 import CLearningTD3
from algo.normalizer import normalizer
from algo.util import update_g_norm, update_o_norm, save_results


def train(params, env):

    # folder for this run
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time_s = time()
    torch.manual_seed(params['overall_seed'])

    run_ID = f"{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
    run_dir = f"runs/{env.name} /{run_ID}"
    os.makedirs(run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(json.dumps(params, indent=4))

    # cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define normalizer
    o_norm = normalizer(size=env.d_obs, default_clip_range=5)
    g_norm = normalizer(size=env.d_goal, default_clip_range=5)

    # define the training method (different for discrete and continious environments)
    if env.discrete:
        learning_method_wrapper = CLearningDiscrete(env=env, device=device,**params)
        learning_method_wrapper.set_optimizer(params['learning_rate'], params['use_LR_scheduler'],
                                              milestones=[params['LR_change_episode']],
                                              gamma=1 / params['LR_reduce_factor'])
    else:
        learning_method_wrapper = CLearningTD3(env, device, **params)
        learning_method_wrapper.set_optimizer(learning_rate_critic=params['critic_learning_rate'], learning_rate_actor=params['actor_learning_rate'])
        update_g_norm(env=env, g_norm=g_norm)

    # save dataset gathered by episode runs, goal list generated, mean succes rate, mean steps taken
    dataset = []
    eval_goal_list = []
    eval_mean_success = []
    eval_mean_step = []
    eval_pts = []
    env_steps = []
    total_steps_taken = 0

    # rng
    batch_sampling_rng = np.random.RandomState(params['overall_seed'] + 1)
    exploration_rng = np.random.RandomState(params['overall_seed'] + 2)

    # random exploration
    def random_exploration_policy(states, goal):
        if env.discrete:
            return exploration_rng.choice(env.n_actions)
        else:
            return np.random.uniform(low=-env.action_max, high=env.action_max, size=env.d_actions)

    for _ in range(params['random_exploration_eps']):
        ep_history, ep_success = learning_method_wrapper.run_episode(action_policy=random_exploration_policy,
                                          ep_length=params['max_ep_len'])
        total_steps_taken += len(ep_history)

        # add episode to the dataset when the length is larger than 1
        if len(ep_history) > 1:
            dataset.append(ep_history)

    gde_start_time_s = time()

    for gde_ep in range(params['goal_directed_eps']):

        if env.discrete:

            # epsilon decay
            if params['use_decaying_epsilon']:
                exploration_epsilon = params['exploration_eps'] / (1.0 + (gde_ep / params['epsilon_decay_denominator']))
            else:
                exploration_epsilon = params['exploration_eps']

            policy = learning_method_wrapper.goal_conditioned_c_learning_policy(exploration_rng,
                                            exploration_epsilon)
            ep_history, ep_success = learning_method_wrapper.run_episode(action_policy=policy,
                                     ep_length=params['max_ep_len'])

        else:
            exploration_epsilon = params['exploration_eps']
            noise_epsilon = params['noise_eps']

            policy = learning_method_wrapper.goal_conditioned_c_learning_policy(rng=exploration_rng,
                                                        eval=False, exploration_epsilon=exploration_epsilon,
                                                        noise_epsilon=noise_epsilon, g_norm=g_norm, o_norm=o_norm)

            ep_history, ep_success = learning_method_wrapper.run_episode(action_policy=policy,
                                                 ep_length=params['max_ep_len'])

            # update statistics for o norm using the collected episode
            update_o_norm(o_norm=o_norm, episode=ep_history)

        # add episode to the dataset when the length is larger than 1
        if len(ep_history) > 1:
            dataset.append(ep_history)

        # calc total steps taken
        total_steps_taken += len(ep_history)
        env_steps.append(total_steps_taken)

        # evaluation step
        if gde_ep % params['eval_freq'] == 0:

            eval_goal_step = []
            eval_goal_success = []
            goal_list = []
            for _ in range(params['num_eval_goals']):
                if env.discrete:
                    policy = learning_method_wrapper.goal_conditioned_c_learning_policy(exploration_rng, exploration_epsilon, eval=True)

                    ep_history, ep_success = learning_method_wrapper.run_episode(action_policy=policy,
                                             ep_length=params['max_ep_len'],
                                             use_eval_reset=(env.name == 'Precipice'))
                else:
                    policy = learning_method_wrapper.goal_conditioned_c_learning_policy(rng=exploration_rng,
                                                                                        eval=True,
                                                                                        exploration_epsilon=exploration_epsilon,
                                                                                        noise_epsilon=noise_epsilon,
                                                                                        g_norm=g_norm, o_norm=o_norm)

                    ep_history, ep_success = learning_method_wrapper.run_episode(action_policy=policy,
                                                                             ep_length=params['max_ep_len'])

                eval_goal_step.append(len(ep_history))
                eval_goal_success.append(ep_success)

            gde_elapsed_time_s = time() - gde_start_time_s
            gde_mean_ep_time = gde_elapsed_time_s / (
                        (gde_ep + 1) + (gde_ep // params['eval_freq'] + 1) * (1 + params['num_eval_goals']))

            # print and save the result for the eval
            print(
                f'Mean success rate to goal: {np.mean(eval_goal_success):.2f}\tMean step to goal: {np.mean(eval_goal_step):.1f} ' +
                f' [episode {gde_ep} / {params["goal_directed_eps"]}, {gde_mean_ep_time:.2f} s per episode]')

            eval_goal_list.append(np.array(goal_list))
            eval_mean_success.append(np.mean(eval_goal_success))
            eval_mean_step.append(np.mean(eval_goal_step))
            eval_pts.append(gde_ep)

            # visualization for the trajectories (only minimaze and dubinscar)
            if gde_ep % params['plot_freq'] == 0:
                if env.name == 'DubinsCar':
                    env.plot_trajectory(
                        trajectory=ep_history,
                        save_as=f'plot{gde_ep}.png',
                        goal=ep_history[0]["desired_goal"],
                        save_dir=run_dir
                    )
                elif env.name == 'Minimaze':
                    env.plot_policy(
                        policy=policy,
                        save_as=f'plot{gde_ep}.png',
                        goal=ep_history[0]["desired_goal"],
                        save_dir=run_dir
                    )

        #train the models
        for _ in range(params['train_steps_per_ep']):
            # sample a mini-batch of training
            batch_size = params['batch_size']
            learning_method_wrapper.learn(gde_ep, dataset, batch_size=batch_size, batch_sampling_rng=batch_sampling_rng, o_norm=o_norm, g_norm=g_norm)

        # save the most recent method every 500 episodes
        if gde_ep % 500 == 0:
            if env.discrete:
                save_results(run_dir=run_dir, eval_pts=eval_pts, eval_goal_list=eval_goal_list,
                             eval_goal_step=eval_mean_step,
                             eval_goal_success=eval_mean_success, env_steps=env_steps,
                             critic1=learning_method_wrapper.network,
                             critic2=learning_method_wrapper.network, actor=learning_method_wrapper.network,
                             o_norm=None, g_norm=None)
            else:
                save_results(run_dir=run_dir, eval_pts=eval_pts, eval_goal_list= eval_goal_list, eval_goal_step=eval_mean_step,
                             eval_goal_success=eval_mean_success, env_steps=env_steps, critic1=learning_method_wrapper.critic1,
                             critic2=learning_method_wrapper.critic2, actor=learning_method_wrapper.actor, o_norm=o_norm, g_norm=g_norm)


    elapsed_time_s = time() - start_time_s
    print('Training took {:.0f} seconds.'.format(elapsed_time_s))



