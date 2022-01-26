#!/usr/bin/env python3

import gym
import numpy as np
from swimmer_module import test as sm


class Environment(gym.Env):
    def __init__(self):
        self.swimmer = sm.Swimmer(swimmer_type_id = 1, action_interval = 1.0, reward_gain = 30.0)

        self.observation_space = gym.spaces.Box(
            low =-100,
            high= 100,
            shape = (4, 3)    
        )
        self.action_space = gym.spaces.Discrete(2)
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


if __name__ == '__main__':
    env = Environment()
    action = [0,0,0,0]
    obs, reward, done, _ =  env.step(action)
    print("action",env.action_space )
