from PIL import Image
import os
from os import path
from matplotlib import pyplot as plt
import cv2
import numpy as np

videodims = (512, 256)
fourcc = cv2.VideoWriter_fourcc(*'I420')
video = cv2.VideoWriter("test.avi", fourcc, 4, videodims)
skeleton_path = '/data/one punch/how people walk skeleton'
image_path = '/data/one punch/how people walk'
skeleton_list = os.listdir(skeleton_path)
skeleton_list.sort()
for file in skeleton_list:
    real_image = Image.open(path.join(image_path, file)).resize([256, 256])
    real_skeleton = Image.open(path.join(skeleton_path, file)).resize([256, 256])
    combime_image = Image.new('RGB', [512, 256])
    combime_image.paste(real_image, [0, 0])
    combime_image.paste(real_skeleton, (256, 0))
    # plt.imshow(combime_image)
    # plt.show()
    video.write(cv2.cvtColor(np.array(combime_image), cv2.COLOR_RGB2BGR))
video.release()
print('sef')