# RoboND Deep Learning Project

[//]: # (Image References)

[img_fcn_architecture_1]: imgs/img_fcn_model_1_simple.png
[img_fcn_architecture_2]: imgs/img_fcn_model_2_deeper.png
[img_fcn_architecture_3]: imgs/img_fcn_model_3_vgg.png

[**video 1 - follow me mode - inference**](https://www.youtube.com/watch?v=hCQh8I8g0sg)

[**video 2 - special simulator build for data gathering**](https://www.youtube.com/watch?v=Nq95abB7FiE)

## Introduction

This project consists in the implementation of a Fully Convolutional Neural Network for semantic segmentation, which is used as a component of a perception pipeline that allows a quadrotor to 

## **Rubric points**

### **Problem statement**

<!-- RUBRIC POINT 4 -->
The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

### **Network architecture**

![SIMPLE ARCHITECTURE 1][img_fcn_architecture_1]

![SIMPLE ARCHITECTURE 2][img_fcn_architecture_2]

![SIMPLE ARCHITECTURE 3][img_fcn_architecture_3]

<!-- RUBRIC POINT 1 -->
The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.

<!-- RUBRIC POINT 3 -->
The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

### **Hyperparameters tuning**

<!-- RUBRIC POINT 2 -->
The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

    Epoch
    Learning Rate
    Batch Size
    Etc.

All configurable parameters should be explicitly stated and justified. 

### **Results**

<!-- RUBRIC POINT 5 -->
The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required. 

<!-- RUBRIC POINT 6-->
The file is in the correct format (.h5) and runs without errors.

<!-- RUBRIC POINT 7 -->
The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.