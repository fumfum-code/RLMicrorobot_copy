import pickle
import numpy as np

x1 = np.array([-1.0, 0.0, 0.0])
x2 = np.array([ 0.0, 0.0, 0.0])
x3 = np.array([ 1.0, 0.0, 0.0])
x4 = np.array([ 2.0, 0.0, 0.0])

Swimmer_type = dict(numParticle = 4,
                    pos = np.stack([x1,x2,x3,x4]),
                    init_pos = np.stack([x1,x2,x3,x4]),
                    radius = 0.1,
                    arm_length = [1.0,1.0,1.0],
                    K_star = 10.0,
                    frec = 0.02
                     )

with open('../../model/swimmer_type1.pickle','wb') as f:
    pickle.dump(Swimmer_type,f)   

print('update ') 
