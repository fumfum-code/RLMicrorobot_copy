#!/usr/bin/env python3
import os
import sys
import gym
import swimmer_env
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
def main():
    env = gym.make('SwimmerEnv-v0')
    eval_env = Monitor(env)
    maxscore = 0.0
    
    create_new_model = False
    log_dir = 'rl/test/aaa'
    
    analyse = False
    analyse_file = 'data/swimmer_type_1/action.csv'
    if create_new_model:
        model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.0,
                vf_coef=0.5,
                use_sde=True,
                sde_sample_freq=4,
                target_kl=None,
                tensorboard_log=log_dir,
                create_eval_env=False,
                policy_kwargs=None,
                verbose=0,
                seed=None,
                device='cpu',
                _init_setup_model=True
                )
        

    else:
        model = PPO.load(f'./rl/agent_save/PPO_type120220126_111145/PPO_trained_epoch_0')

    print('simulating ...')
    
    done = False
    obs = env.reset()
    step_counter = 0
    epi_reward = 0 
    while not done: 
        action, _states = model.predict(obs, deterministic=True)
        print(f'step counter : {step_counter}')
        print(f'action {action}')
        dir_exist_flag = os.path.isfile(analyse_file)
        if analyse:
            if dir_exist_flag:
                with open(analyse_file, 'a') as f:
                    np.savetxt(f, action, delimiter=',', fmt='%4f')
            else:
                np.savetxt(analyse_file, action, delimiter=',', fmt='%4f')

        obs, reward, done, info = env.step(action)
        
        print(f'reward : {reward}')
       
        step_counter+=1
        epi_reward += reward        
        if done:
            print('finish')


    print(f'episode reward is {epi_reward}')
    """ 
    print("********** EVALUATING  **********")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    
    
    print(f'mean reward: {mean_reward} +/- {std_reward}')
    """
    
if __name__ == '__main__':
    main()
