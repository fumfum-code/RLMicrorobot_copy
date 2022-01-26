import glob
import cv2
from tqdm import tqdm

img_array = []
for filename in sorted(glob.glob("swimmer_type1/*.png")):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

clip_fps = 15.0
name = f'movie/swimmer_type1_trained.mp4'
out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'),clip_fps, size)

for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()
