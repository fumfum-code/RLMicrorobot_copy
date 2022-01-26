#/usr/bin/env python3
import datetime
import os
import sys
import gym
from tqdm import tqdm
import swimmer_env

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env



is_record = False 
action_interval = 1
swimmer_type_id = 1
reward_gain = 1.0
def main():
    """
    Hyper Parameters
    """
    
    n_envs = 16
    time_steps = int(1e+6)
    epoch = 4
    
 
    """""""""""""""
    " log setting "
    """""""""""""""
    now = datetime.datetime.now()
    model_name =f'PPO_type1'+now.strftime('%Y%m%d_%H%M%S')
    model_save_dir = f'./rl/agent_save/{model_name}/'
    os.makedirs(model_save_dir, exist_ok=True)
    log_dir = f'./log/agent_log/log_type_3/'
    os.makedirs(log_dir, exist_ok=True)
    

    """
    Learning Setting
    """

    multi_process = True
    create_new_model = True
    save_model = True

    
    if multi_process:
        env = SubprocVecEnv(
                [lambda: Monitor(gym.make('SwimmerEnv-v0'),log_dir) for i in range(n_envs)],
                start_method='spawn'
                )        


    else:
        env = make_vec_env('SwimmerEnv-v0',
                n_envs=n_envs,
                monitor_dir=(log_dir+model_name+'_monitor')
                            )
  
    
    #model = PPO('MlpPolicy',env=env,verbose=1)

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
                verbose= 1,
                seed=None,
                device='cpu',
                _init_setup_model=True
                )    
    
    else:
        model=PPO.load('rl/agent_save/PPO_type120220118_162905/PPO_trained_epoch_3', tensorboard_log=log_dir)
        model.set_env(env)
    
    
    eval_env = Monitor(gym.make('SwimmerEnv-v0'))

    testModel(model)
    print('*'*10, 'EVALUATING ','*'*10)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2, deterministic=True)
    print(f'mean reward: {mean_reward} +/- {std_reward}')
    max_score = mean_reward


    """
    TRAINING
    """
    
    print('*'*10, ' START LEARNING ','*'*10)
    for i in tqdm(range(epoch)):
        print('*'*10, 'LEARNING ','*'*10)
        if i == 0:
            model.learn(total_timesteps=int(time_steps),
                        tb_log_name=model_name,
                        reset_num_timesteps=True)
        
        else:
            model.learn(total_timesteps=int(time_steps),
                        tb_log_name=model_name,
                        reset_num_timesteps=False)


        testModel(model)
        print('*'*10, 'EVALUATING ','*'*10)
        mean_reward, std_reward = evaluate_policy(model,
                eval_env, n_eval_episodes=5)
        print(f'mean reward: {mean_reward} +/- {std_reward}')

        if save_model == True: #and mean_reward > max_score:
            print('*'*10, 'SAVING MODEL ','*'*10)
            model.save(model_save_dir + f'PPO_trained_epoch_{i}')
            max_score = mean_reward

    """
    print('start learning ...')
    model.learn(total_timesteps = 10000)

    print('evaluating ...')
    eval_env = Monitor(gym.make('SwimmerEnv-v0'))
    mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10
            )
    print(f'mean reward : {mean_reward}')
    print(f'std reward : {std_reward}')
    """
def testModel(model):
    
    print('*'*10, 'TEST MODEL','*'*10)
    env = gym.make('SwimmerEnv-v0')

    epi_reward = 0    
    
    obs = env.reset()
    for i in tqdm(range(50)):
        action, _states = model.predict(obs,deterministic=True)
        #print(action)
        obs, reward, done, info = env.step(action)
        print(reward)
        epi_reward += reward
        if done == True:
            print(f'epi_reward : {epi_reward}')
            break 



if __name__ == '__main__':
    main()
