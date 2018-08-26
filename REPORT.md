# RoboND Deep Learning Project

[//]: # (Image References)

[img_fcn_architecture_1]: imgs/img_fcn_model_1_simple.png
[img_fcn_architecture_2]: imgs/img_fcn_model_2_deeper.png
[img_fcn_architecture_3]: imgs/img_fcn_model_3_vgg.png

[img_IoU_results]: imgs/img_IoU_results.png

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


## Semantic segmentation and FCNs <a id='semantic_segmentation'></a>

TODO

## Data gathering <a id='data_gathering'></a>

TODO

## Network architecture and implementation <a id='network_architecture'></a>

![SIMPLE ARCHITECTURE 1][img_fcn_architecture_1]

![SIMPLE ARCHITECTURE 2][img_fcn_architecture_2]

![SIMPLE ARCHITECTURE 3][img_fcn_architecture_3]

## Hyperparameters tuning <a id='hyperparameters_tuning'></a>

TODO

## Model training <a id='model_training'></a>

TODO

## Results <a id='results'></a>

<!-- RUBRIC POINT 5 -->
The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required. 

<!-- RUBRIC POINT 6 DONE -->
The trained models can be found [**here**](https://github.com/wpumacay/RoboND-DeepLearning-Project/tree/master/data/weights) and include the following trained models :

*   Model 1 ( model_weights_simple_1_full_dataset.h5 ) : 3 convolutional layers and 3 transpose-convolutional layers, trained with batch size of 32.
*   Model 2 ( model_weights_simple_2_full_dataset.h5 ) : Same model as before, but trained with a batch size of 64.
*   Model 3 ( model_weights_simple_3_full_dataset.h5 ) : A one layer deeper model, with 4 convolutional layers and 4 transpose-convolutional layers.
*   Model 4 ( model_weights_vgg.h5 ) : A deeper model with an encoder based on the VGG-13 architecture ( reference [**here**](https://arxiv.org/pdf/1409.1556.pdf) ). The encoder has 8 convolutional layers and 5 max. pooling layers.

<!-- RUBRIC POINT 7 DONE -->
The resulting final score ( IoU based ) for one of our models is 0.465.

![RESULT_IOU_0][img_IoU_results]

All the trained models that we uploaded obtained a score greater than the required score of 0.4, with values oscillating very close to the previously mentioned score.

## Conclusions and future work <a id='conclusions'></a>

TODO




















### **Problem statement**

<!-- RUBRIC POINT 4 -->
The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

### **Network architecture**

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
