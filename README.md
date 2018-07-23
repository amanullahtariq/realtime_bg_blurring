# Real Time Background Blurring using DeeplabV3

## Quick summary


2 days ago Microsoft introduced Teams with a background blur feature for video calls using AI, so I used AI to build my own

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/1317442/42730592-ac9c583a-8811-11e8-907d-4c074603708a.gif)

Here's a clip with slightly more blur

![ezgif-3-8d13b793b6](https://user-images.githubusercontent.com/1317442/42804853-9d8cc65a-89c3-11e8-9cfc-e4d1bdd1b57e.gif)

You can read the blog post about it [here](http://zubairahmed.net/2018/07/17/background-blurring-with-semantic-image-segmentation-using-deeplabv3/) 



This is the code to detect the parking space for the car given 2D image from the google maps and 3D point Cloud data of the current enivornment.

## Purpose
This project was made for the purpose of the AIS LAB.

This repository contains:

- 2D Lane Detection
- 3D parking detection

## Results
* Initial data 2D image taking from the google image and plotting the lines using OpenCV

![InitialData](images/initialdata.png)

* Applying filters to enhance the lines of the parking spots
![LaneDetection2D](images/2d.png)

* 3D data of the parking area collected by laser sensors and then used using Point Cloud Library
![3D-Data](images/3D_Data.png)

* Converting 2D image to 3D space so later it can be combine with 3D data 
![2Dto3D](images/2dto3d.png)

* Applying machine learning algorithm, linear regression to detect parking spots in 3D space and combining it with the 2D image data
![FinalResult](images/Final.png)

* Enchancing result by Applying logistic regression
![Result](images/Final2.png)

## Requiremnts
For this project to run you need:
* Visual Studio 2012
* OpenCV 3.0
* Point Cloud Library

### Who do I talk to? ###
I would be happy to talk to you about this project and if you are interested then we can further enhance this project to.

### Contact
* Amanullah Tariq 
* Email: amanullah.tariq@gmail.com
