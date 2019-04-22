import numpy as np
import random
import os
from config import batch_size
import numpy as np
from PIL import Image
img_names = os.listdir('./data/faces')
img_num = len(img_names)
random.shuffle(img_names)
batch_num = int(np.ceil(img_num / batch_size))
def next_batch():
    for i in range(batch_num):
        start = i * batch_size
        if i == batch_num - 1:
            end = img_num
        else:
            end = (i + 1) * batch_size
        batch_img_names = img_names[start:end]
        batch_data = []
        for name in batch_img_names:
            img = np.transpose(np.array(Image.open('./data/faces/{}'.format(name))), (2,0,1))
            batch_data.append(img)
        yield np.array(batch_data).astype(np.float32) / 255

def save_img(imgs):
    data = []
    for img in imgs:
        img  =np.transpose(img, (1,2,0))
        data.append(img)