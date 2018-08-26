# RoboND Deep Learning Project

[//]: # (Image References)

[img_fcn_architecture_1]: imgs/img_fcn_model_1_simple.png
[img_fcn_architecture_2]: imgs/img_fcn_model_2_deeper.png
[img_fcn_architecture_3]: imgs/img_fcn_model_3_vgg.png

[gif_follow_me_sample_1]: imgs/gif_follow_me_sample_1.gif
[gif_follow_me_sample_2]: imgs/gif_follow_me_sample_2.gif

[**video 1 - follow me mode - inference**](https://www.youtube.com/watch?v=hCQh8I8g0sg)

[**video 2 - special simulator build for data gathering**](https://www.youtube.com/watch?v=Nq95abB7FiE)

## **Introduction**

This project consists in the implementation of a Fully Convolutional Neural Network for **semantic segmentation**, which is used as a component of a perception pipeline that allows a quadrotor to follow a target person in a simulator.

![SAMPLE FOLLOW ME FOUND TARGET][gif_follow_me_sample_1]

![SAMPLE FOLLOW ME FOLLOWING TARGET][gif_follow_me_sample_2]

To achieve this we followed these steps :

*   Gathered training data from a simulator ( made a special build for easier data gathering, compared to the standard suggested approach ).
*   Designed a implemented a Fully Convolutional Network ( using keras and tensorflow ) for the task of semantic segmentation, based on the lectures given at udacity's RoboND deep learning section, and in [this](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) and [this](https://arxiv.org/pdf/1409.1556.pdf) papers.
*   Made experiments to tune our training hyperparameters ( learning rate, batch size and number of epochs ).
*   Trained the designed model using the gathered training data and tuned hyperparameters and checked the testing accuracy using the Intersection Over Union (IoU) metric.

This work is divided into the following sections :

1.  [Semantic segmentation and FCNs](#semantic_segmentation)
2.  [Data gathering](#data_gathering)
3.  [Network architecture and implementation](#network_architecture)
4.  [Hyperparameters tuning](#hyperparameters_tuning)
5.  [Model training](#model_training)
6.  [Results](#results)
7.  [Conclusions and future work](#conclusions)


## Semantic segmentation and FCNs <id ="semantic_segmentation"></a>

## Data gathering <a id='data_gathering'></a>

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