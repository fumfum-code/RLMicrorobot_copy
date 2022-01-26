import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xlabel = [f'theta_1']
ax.set_ylabel = [f'theta_2']

ax.set_title = ['angle phase']

data = np.loadtxt('./data/swimmer_type_1/phase_angle.csv', delimiter=',')
data = data.reshape(1000,2)
for angle in data:
    ax.scatter(angle[0], angle[1],alpha=0.2, color='b')
ax.plot(angle[0],angle[1],color='r')
fig.savefig('./figure/result/angle_phase_type2.png')
plt.show()
