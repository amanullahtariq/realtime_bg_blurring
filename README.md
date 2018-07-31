# Real Time Background Blurring using DeeplabV3
This repository contains the code for real time background blurring using deep learning model *[DeeplabV3](https://arxiv.org/abs/1706.05587)*. We used the [Keras implementation of DeeplabV3](https://github.com/bonlime/keras-deeplab-v3-plus) with pretrained weights. 

## Introduction
Few days ago Microsoft introduced Teams with a background blur feature for video calls using AI, so I used AI to build my own


Here's a clip with slightly more blur


You can read the blog post about it [here](http://amanullahtariq.com/blog_posts/realtime_bg_blur.html) 



This is the code to detect the parking space for the car given 2D image from the google maps and 3D point Cloud data of the current enivornment.

## DeepLabV3
 DeeplabV3 is a state-of-the-art model for Semantic Segmentation develop by Google Inc. 



## Results
* Initial data 2D image taking from the google image and plotting the lines using OpenCV


* Applying filters to enhance the lines of the parking spots
![LaneDetection2D](images/2d.png)


## Run
```
sh setup.sh
```


## Requiremnts
For this project to run you need:
* Python 3.5
* Tensorflow 1.4
* OpenCV
* imutils
* Keras


## References

1.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam <br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

2.  **MobileNetV2: Inverted Residuals and Linear Bottlenecks** <br />
    Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen. <br />
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.


### Contact
* Amanullah Tariq 
* Email: amanullah.tariq@gmail.com
* Website: http://amanullahtariq.com/
* Github: http://github.com/amanullahtariq
* Linkedin: https://www.linkedin.com/in/amanullahtariq/
