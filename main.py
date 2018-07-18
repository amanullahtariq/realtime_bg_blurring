from matplotlib import  pyplot as plt
import cv2
import numpy as np
from model import Deeplabv3
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# deeplab_model = Deeplabv3()

img = plt.imread("imgs//image4.jpg")
w,h, _ = img.shape
deeplab_model = Deeplabv3()


ratio = 512. / np.max([w,h])

resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
resized = resized / 127.5 - 1
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')

res = deeplab_model.predict(np.expand_dims(resized2,0))
labels = np.argmax(res.squeeze(),-1)
labels = labels[:-pad_x]



mask = labels == 0

resizedFrame = cv2.resize(img, (labels.shape[1],labels.shape[0]))
blur = cv2.GaussianBlur(resizedFrame, (41,41), 0)
resizedFrame[mask] = blur[mask]
plt.imshow(resizedFrame)
plt.waitforbuttonpress()


