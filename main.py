from matplotlib import  pyplot as plt
import cv2
import numpy as np
from model import Deeplabv3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def start_video():
    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def blur_image(image_path):
    img = plt.imread(image_path)
    w, h, _ = img.shape
    deeplab_model = Deeplabv3(backbone='xception', OS=8)

    ratio = 512. / np.max([w, h])

    resized = cv2.resize(img, (int(ratio * h), int(ratio * w)))
    resized = resized / 127.5 - 1
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')

    res = deeplab_model.predict(np.expand_dims(resized2, 0))
    labels = np.argmax(res.squeeze(), -1)
    labels = labels[:-pad_x]

    mask = labels == 0

    resizedFrame = cv2.resize(img, (labels.shape[1], labels.shape[0]))
    blur = cv2.GaussianBlur(resizedFrame, (15, 15), 0)
    resizedFrame[mask] = blur[mask]
    plt.imshow(resizedFrame)
    plt.waitforbuttonpress()


def test_blurring():
    img = cv2.imread("imgs//image1.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow( "HSV Image", hsv)



if __name__== "__main__":
    #blur_image("imgs//image1.jpg")
    test_blurring()







