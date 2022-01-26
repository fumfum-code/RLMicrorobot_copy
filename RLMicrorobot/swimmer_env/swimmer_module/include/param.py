import pickle
import numpy as np

DT = 1.0e-2
MAX_TIME = 1000
OUTPUT_ITER = 1.0e-1
action_interval = 1.0
reward_gain = 25


param = dict(DT = DT,
             MAX_TIME = MAX_TIME,
             OUTPUT_ITER = OUTPUT_ITER,
             OUTPUT_PATH = './data/swimmer_type_1/',
             action_interval = action_interval,
             reward_gain = reward_gain
              )       

with open('../../model/param.pickle', 'wb') as f:
    pickle.dump(param,f)


print('parameter updated')
