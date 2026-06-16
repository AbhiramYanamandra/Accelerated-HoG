# Accelerated Histogram of Oriented Gradients
 
In traditional machine learning techniques (non-deep-learning ones), image features extracted from input images replace raw images for more efficient learning and inferencing. As this example demonstrates, Histogram of Oriented Gradients (HoG) descriptors coupled with SVM can be very powerful in image classification tasks. In this project, you'll focus on accelerating the computation of HoG descriptors. For the sake of simplicity, only consider 8-bins HoG in this project. (https://medium.com/@basu369victor/handwritten-digits-recognition-d3d383431845)
 
You'll use your HoG implementation to compute the HoG descriptors for a batch of images. The speedup can be measured by the hardware-accelerated design's latency and/or throughput compared with a software-only implementation on the PS.
 
HoG algorithm background:
   
https://www.youtube.com/watch?v=4ESLTAd3IOM
   
https://lilianweng.github.io/lil-log/2017/10/29/object-recognition-for-dummies-part-1.html
   
https://learnopencv.com/histogram-of-oriented-gradients/
 
Steps to approach this task:
1.	Study the HoG algorithm carefully. Take note of parallel operations.
2.	Develop a software implementation. Test it thoroughly before continuing.
3.	Profile the implementation to identify performance bottlenecks
4.	Ideally, you should be able to create your HLS design using C code for the PS implementation.


# Implementation detail of HoG

While HoG descriptor is provided by OpenCV and scikit-learn in python. Their implementation detail differences causes their output to mismatch. To help you to produce a working C/C++ implementation of HoG, here's an implmentation in python attached (Credits to preethampaul at Github: https://github.com/preethampaul/HOG). You may also develop you own implementation from scratch, as long its output matches with an existing implementation available. The  HoG parameters in this design are: 
    
    cell size: 8x8 pixels
    block size: 2x2 cells per block
    block stride: 8 pixels
    bins: 8 directions

    Therefore, the result length is:
        3x3 blocks, 2x2 cells per block, 8 bins per cell
        9*4*8= 288

input images are provided in cifar_bw folder (credit: https://www.cs.toronto.edu/~kriz/cifar.html)

Execute run.py to generate reference output