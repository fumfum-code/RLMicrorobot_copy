#!/usr/bin/env python3

import gym
import numpy as np
import swimmer_module as sm


class Environment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low =-100,
            high= 100,
            shape = (2, )    
        )
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0

    def reset(self):
        """Return observation"""
        self.step_count = 0
        return [0.0,0.0]

    def step(self,action):
        """
        get action
        return {observation, reward, done, info(dict)}
        """

        done = False
        reward  = 0
        if action == 0:
            reward = 1
        if self.step_count == 100:
            done = True

        self.step_count += 1

        return [
                [0.0,0.0],
                reward,
                done,
                {}
                ]


