import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
from tqdm import tqdm


type_name = 'swimmer_type1'
data_name = 'PPO_1_test.csv'
fig_dir   = './figure/manual_test'
movie_name = f'./figure/movie/{type_name}_manual_test.mp4'


data = np.loadtxt('./data/swimmer_type_1/'+data_name, delimiter=',')
for i, pos in tqdm(enumerate(data)):
    fig, ax = plt.subplots()
    ax.set_xlim([-2,5])
    ax.set_ylim([-3,3])
    ax.set_title(f'action1 : {pos[12]} action2 : {pos[13]} step : {i}')
    pos = np.delete(pos, [12,13])
    pos_res = pos.reshape(4,3)
    pos_res = pos_res.T
   
    ax.scatter(pos[0],pos[1])
    ax.scatter(pos[3],pos[4])
    ax.scatter(pos[6],pos[7])
    ax.scatter(pos[9],pos[10])
    ax.plot(pos_res[0],pos_res[1])
    
    fig.savefig(fig_dir + f'/result_{i:04}.png')

img_array = []
for filename in sorted(glob.glob(fig_dir + f'/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

clip_fps = 15.0
out = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*'mp4v'), clip_fps, size)

for i in tqdm(range(len(img_array))):
    out.write(img_array[i])

out.release()



 

