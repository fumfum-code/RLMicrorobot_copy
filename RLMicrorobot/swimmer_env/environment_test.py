#!/usr/bin/env python3
import gym
import numpy as np
from swimmer_env.swimmer_module import module as sm


class Environment(gym.Env):
    def __init__(self):
        self.swimmer = sm.Swimmer(swimmer_type_id=1)

        self.observation_space = gym.spaces.Box(
            low =-10,
            high= 10,
            shape = (8, )    
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.step_count = 0

    def reset(self):
        """Return observation"""
        return self.swimmer.reset()

    def step(self,action):
        """
        get action
        return {observation, reward, done, info(dict)}
        """
        return self.swimmer.step(action)
    

    def render(self):
        pass

"""
if __name__ == '__main__':
    env = Environment()
    action = [0,0]
    obs, reward, done, _ =  env.step(action)
    print(f'obs : {obs} , reward : {reward}')
    print("action",env.action_space )
"""
