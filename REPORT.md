# RoboND Deep Learning Project

[//]: # (Image References)

[img_fcn_architecture_1]: imgs/img_fcn_model_1_simple.png
[img_fcn_architecture_2]: imgs/img_fcn_model_2_deeper.png
[img_fcn_architecture_3]: imgs/img_fcn_model_3_vgg.png

[img_semantic_segmentation_input]: imgs/img_semantic_segmentation_input.jpeg
[img_semantic_segmentation_output]: imgs/img_semantic_segmentation_output.png
[img_semantic_segmentation_definition]: imgs/img_semantic_segmentation.png
[img_vgg_image_classifier]: imgs/img_vgg_image_classifier.png
[img_fcn_paper_1]: imgs/img_fcn_paper_1.png

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


## **Semantic segmentation and FCNs** <a id='semantic_segmentation'></a>

<!-- ### **Problem statement**-->
<!-- RUBRIC POINT 4 -->
<!--The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.-->

### _**Problem definition**_

The problem of **semantic segmentation** consists of doing single-pixel classification for every pixel in an image ( assign a category to each pixel ). The output of semantic segmentation is then an image with pixels values representing the one-hot encoded class assigned.

![SEMANTIC SEGMENTATION DEFINITION][img_semantic_segmentation_definition]

The approach taken in this work is to use Fully Convolutional Networks, which **allow us to obtain this required mapping**.

### _**Fully Convolutional Networks**_

Deep networks give state of the art results in various computer vision tasks ( like [image classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) ). The common approach is to use a network architecture that includes **convolutional layers**, in order to take advantage of the spatial information of the problem ( images, which are 2D arrays ).

In image recognition, an architecture used would be the following ( image based on VGG-B configuration, with 13 layers, from [**this**](https://arxiv.org/pdf/1409.1556.pdf) paper ) :

![VGG IMAGE CLASSIFICATION][img_vgg_image_classifier]

The last layers of this model are Fully Connected layers, which give the final output as a vector of probabilities for each class we have to detect.

To get an output image we instead need to replace the last fully connected layers for some other type of structures that will give us a volume as a result ( width, height, depth ), so we have to use a structure that operates with volumes. 

Because of this, we make use of convolutional layers, as described in [**this**](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) paper. The following image ( from [1] ) shows the general architecture described in the paper.

![FCN from paper 1][img_fcn_paper_1]

The general idea is to replace the fully connected layers for upsampling layers, which avoid flattening and keep working with 4D volumes ( batch, width, height, depth ) instead of flattened values. This resulting architecture is called a **Fully Convolutional Layers** ( all layers operate with volumes ).

### _**Intuition for FCNs usage**_

The resulting architecture has a similar structure as other Deep models used for different tasks, like AutoEncoders, and Sequence-to-Sequence models. Both of these models have 2 specific parts in their structure : an **Encoder** and a **Decoder**.

*   In [**autoencoders**](https://www.youtube.com/watch?v=9zKuYvjFFS8), the encoder tries to reduce the dimensionality of the input image ( similar to generating an embedding ), and the encoder is in charge of taking this reduced representation and generating an image out of it, which should be very similar to the original image.

*   In sequence to sequence models ( like in machine translation [3] ) the encoder reduces the input sequence to a vector representation of this input, and then the decoder generates an output sequence from this vector representation in another language.

The intuition of why this architecture would work is because of the encoding-decoding structures :

*   The encoding structure is in charge of reducing the original input volume to a smaller volume representation, which holds information that describe this image.
*   The decoding structure is in charge of taking this volume representation and generating an output image that solves the task at hand ( in our case, generate the pixel-wise classification of the input image in an output volume ).



## **Data gathering** <a id='data_gathering'></a>

TODO

## **Network architecture and implementation** <a id='network_architecture'></a>

<!-- RUBRIC POINT 1 -->
<!--The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.-->

<!-- RUBRIC POINT 3 -->
<!--The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.-->

![SIMPLE ARCHITECTURE 1][img_fcn_architecture_1]

![SIMPLE ARCHITECTURE 2][img_fcn_architecture_2]

![SIMPLE ARCHITECTURE 3][img_fcn_architecture_3]

## **Hyperparameters tuning** <a id='hyperparameters_tuning'></a>

TODO

## **Model training** <a id='model_training'></a>

TODO

## **Results** <a id='results'></a>

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

## **Conclusions and future work** <a id='conclusions'></a>

TODO




## **References** <a id='references'></a>

*   [1] Jonathan Long, Evan Shelhamer, Trevor Darrel. _**Fully Convolutional Networks for Semantic Segmentation**_ in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 4, pp. 640-651, 1 April 2017.
*   [2] Karen Simonyan, Andrew Zisserman. _**Very Deep Convolutional Networks for Large-Scale Image Recognition**_. ArXiv 2014.
*   [3] Ilya Sutskever, Oriol Vinyals, Quoc V. Le. _**Sequence to Sequence Learning with Neural Networks**_ Proc. NIPS, Montreal, Canada, 2014.

## **Other resources**

*   [**Variational AutoEncoders explanation**](https://www.youtube.com/watch?v=9zKuYvjFFS8)
*   [**Stanford cs231n lecture on detection and segmentation**](https://youtu.be/nDPWywWRIRo?t=9m18s)



### **Network architecture**

### **Hyperparameters tuning**

<!-- RUBRIC POINT 2 -->
The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

    Epoch
    Learning Rate
    Batch Size
    Etc.

All configurable parameters should be explicitly stated and justified. 
