from matplotlib import  pyplot as plt
import cv2
import numpy as np
from model import Deeplabv3
import os
from imutils.video import WebcamVideoStream

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def start_video():
    # deeplab_model = Deeplabv3(backbone='xception', OS=8)

    deeplab_model = Deeplabv3(OS=8)
    vid = WebcamVideoStream(src=0).start()
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    blurValue = (3, 3)
    blur_bg_value = 81

    while True:
        frame = vid.read()
        if frame is None:
            break
        w, h, _ = frame.shape
        ratio = 512. / np.max([w, h])

        resized = cv2.resize(frame, (int(ratio * h), int(ratio * w)))
        resized = resized / 127.5 - 1.
        pad_x = int(512 - resized.shape[0])
        resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')
        res = deeplab_model.predict(np.expand_dims(resized2, 0))
        labels = np.argmax(res.squeeze(), -1)

        labels = labels[:-pad_x]
        mask = labels == 0
        mask_person = labels != 0

        resizedFrame = cv2.resize(frame, (labels.shape[1], labels.shape[0]))
        blur = cv2.GaussianBlur(resizedFrame, (blur_bg_value,blur_bg_value), 0)

        blur_person = cv2.GaussianBlur(resizedFrame, blurValue, 0)


        resizedFrame[mask] = blur[mask]
        resizedFrame[mask_person] = blur_person[mask_person]

        cv2.imshow("result", resizedFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.stop()
    cv2.destroyAllWindows()

def blur_image(image_path):
    blurValue = (3, 3)
    blur_bg_value = 31

    img = plt.imread(image_path)
    w, h, _ = img.shape
    # deeplab_model = Deeplabv3(OS=8)
    deeplab_model = Deeplabv3(backbone='xception', OS=8)

    ratio = 512. / np.max([w, h])

    resized = cv2.resize(img, (int(ratio * h), int(ratio * w)))
    resized = resized / 127.5 - 1
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')

    res = deeplab_model.predict(np.expand_dims(resized2, 0))
    labels = np.argmax(res.squeeze(), -1)
    labels = labels[:-pad_x ]

    # # print(np.unique(labels))

    mask = labels == 0
    mask_person = labels != 0

    resizedFrame = cv2.resize(img, (labels.shape[1], labels.shape[0]))
    blur_person = cv2.GaussianBlur(resizedFrame, blurValue, 0)

    blur_bg = cv2.medianBlur(resizedFrame,blur_bg_value)

    resizedFrame[mask] = blur_bg[mask]
    resizedFrame[mask_person] = blur_person[mask_person]

    # plt.imshow(resizedFrame)
    # plt.waitforbuttonpress()

    cv2.imshow("result", resizedFrame)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

def test_blurring():
    img = cv2.imread("imgs//image1.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # hsv hue sat value

    lower_red = np.array([150, 150, 50])
    upper_red = np.array([180, 255, 150])


    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask= mask)

    kernel = np.ones((15,15), np.float32)/ 225
    smoothed = cv2.filter2D(res, -1 , kernel)

    #cv2.imshow("res", res)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    cv2.imshow("smoothed", smoothed)


    cv2.waitKey(0)


if __name__== "__main__":

    # blur_image("imgs//image1.jpg")
    #test_blurring()
    start_video()






