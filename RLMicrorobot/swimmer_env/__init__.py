from gym.envs.registration import register

register(
        id='SwimmerEnv-v0',
        entry_point='swimmer_env.environment_test:Environment'        
            )
