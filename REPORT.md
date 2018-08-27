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

[img_quadsim_1]: imgs/img_quadsim_1.png
[img_quadsim_data_gathering_1]: imgs/img_quadsim_data_gathering_1.png

[img_cs231n_cnn_basics_1]: imgs/img_cs231n_cnn_basics_1.png
[img_cs231n_cnn_basics_2]: imgs/img_cs231n_basics_output_volume.png
[img_1x1_convolutions]: imgs/img_1x1_convolutions.png
[img_cs231n_unpooling]: imgs/img_unpooling.png
[img_cs231n_transpose_convolutions]: imgs/img_transpose_convolutions.png
[img_cs231n_max_pooling]: imgs/img_max_pooling.png

[img_fcn_paper_skip_connections]: imgs/img_skip_connections.png
[img_fcn_paper_skip_connections_importance]: imgs/img_skip_connections_importance.png

[img_IoU_results]: imgs/img_IoU_results.png

[gif_follow_me_sample_1]: imgs/gif_follow_me_sample_1.gif
[gif_follow_me_sample_2]: imgs/gif_follow_me_sample_2.gif

[gif_quadsim_tools]: imgs/gif_quadsim_tools.gif

[**video 1 - follow me mode - inference**](https://www.youtube.com/watch?v=hCQh8I8g0sg)

[**video 2 - special simulator build for data gathering**](https://www.youtube.com/watch?v=Nq95abB7FiE)

## **Introduction**

This project consists in the implementation of a Fully Convolutional Neural Network for **semantic segmentation**, which is used as a component of a perception pipeline that allows a quadrotor to follow a target person in a simulated environment.

![SAMPLE FOLLOW ME FOUND TARGET][gif_follow_me_sample_1]

![SAMPLE FOLLOW ME FOLLOWING TARGET][gif_follow_me_sample_2]

To achieve this we followed these steps :

*   Gathered training data from a simulator ( made a special build for easier data gathering, compared to the standard suggested approach ).
*   Designed a implemented a Fully Convolutional Network ( using keras and tensorflow ) for the task of semantic segmentation, based on the lectures given at udacity's RoboND deep learning section, and in [this](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) and [this](https://arxiv.org/pdf/1409.1556.pdf) papers.
*   Made experiments to tune our training hyperparameters ( learning rate, batch size and number of epochs ).
*   Trained the designed model using the gathered training data, tuned the model hyperparameters, and checked the testing accuracy using the Intersection Over Union (IoU) metric.

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

The problem of **semantic segmentation** consists of doing single-pixel classification for every pixel in an image ( assign a class label to each pixel ). The desired output of semantic segmentation is then an image with pixels values representing the one-hot encoded class assigned.

![SEMANTIC SEGMENTATION DEFINITION][img_semantic_segmentation_definition]

The approach taken in this work is to use Fully Convolutional Networks, which **allow us to obtain this required mapping**.

### _**Fully Convolutional Networks**_

Deep networks give state of the art results in various computer vision tasks ( like [image classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) ). The common approach is to use network architectures that include **convolutional layers**, in order to take advantage of the spatial information of the problem ( images, which are 2D arrays ).

In image recognition, an architecture used would be the following ( image based on VGG-B configuration, with 13 layers, from [**this**](https://arxiv.org/pdf/1409.1556.pdf) paper ) :

![VGG IMAGE CLASSIFICATION][img_vgg_image_classifier]

The last layers of this model are Fully Connected layers, which give the final output as a vector of probabilities for each class we have to detect.

To get an output image we instead need to replace the last fully connected layers for some other type of structures that will give us a volume as a result ( width, height, depth ), so we have to use a structure that operates with volumes. 

Because of this, we make use of convolutional layers, as described in [**this**](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) paper. The following image ( from [1] ) shows the general architecture described in the paper.

![FCN from paper 1][img_fcn_paper_1]

The general idea is to replace the fully connected layers for upsampling layers, which avoid flattening and keep working with 4D volumes ( batch, width, height, depth ) instead of flattened values. This resulting architecture is called a **Fully Convolutional Layers** ( all layers operate with volumes ).

### _**Intuition for FCNs usage**_

The resulting architecture has a similar structure as other Deep models used for different tasks, like AutoEncoders, and Sequence-to-Sequence models. Both of these models have 2 specific parts in their structure : an **Encoder** and a **Decoder**.

*   In [**autoencoders**](https://www.youtube.com/watch?v=9zKuYvjFFS8), the encoder tries to reduce the dimensionality of the input image ( similar to generating an embedding ), and the decoder is in charge of taking this reduced representation and generate an image out of it, which should be very similar to the original image.

*   In sequence to sequence models ( like in machine translation [3] ) the encoder reduces the input sequence to a vector representation of this input, and then the decoder generates an output sequence in another language based in this intermediate vector representation.

The intuition of why this architecture would work is because of the encoding-decoding structures :

*   The encoding structure is in charge of reducing the original input volume to a smaller volume representation, which holds information that describe this image.
*   The decoding structure is in charge of taking this volume representation and generating an output image that solves the task at hand ( in our case, generate the pixel-wise classification of the input image in an output volume ).


## **Data gathering** <a id='data_gathering'></a>

Our benchmark for testing our architecture is a simulated environment made in [**Unity**](https://unity3d.com/). The simulator is a build from [**this**](https://github.com/udacity/RoboND-QuadRotor-Unity-Simulator) project made by Udacity.

![QUADSIM snapshot][img_quadsim_1]

The process to follow is to use the simulator to generate image data from the GimbalCamera in the Quadrotor, and then preprocess it to get the training data ( input images and output masks ).

![QUADSIM data gathering 1][img_quadsim_data_gathering_1]

The main bottleneck is to make the paths that the quadrotor and target person should follow, and also the spawning points for the other people. At first the functionality provided is enough to get some batches of data, but after some digging into an estimate of the amount of data required and some initial results we choose the options of taking large batches of data, mostly because we expected at first that our agent should be able to navigate the whole environment.

Based on some intuition from [**here**](https://youtu.be/C_LGsoe36I8?t=18m56s), in which they explain how imitation learning works for a self-driving car made by Nvidia, we got the conclussion that our dataset should be expressive enough, so large batches of data should be taken from the simulator, in various situations as explained in the lectures from Udacity ( tips about data gathering ).

The current approach proposed, while possible, is a bit impractical if various scenarios are needed. Based on this issue, we decided to change the simulator in order to implement extra data-gathering tools for this purpose ( a quick video of the tools is found [**here**](https://www.youtube.com/watch?v=Nq95abB7FiE) ). The implementation I made to add these tools can be found in [**this**](https://github.com/udacity/RoboND-QuadRotor-Unity-Simulator) forked repo ( I will make a pull request, once I get to make a summary of the new options available and how to use them ).

![QUADSIM TOOLS][gif_quadsim_tools]

We abstracted the data recording step into **schedules**, which are formed by the following :

*   A patrol path for the quadrotor
*   A hero path for the target
*   A group of spawn points
*   A mode of operation : follow-target, follow-target far, patrol.

We added several options to allow the edition and saving/loading of this schedules into **.json** files. The schedules we created for our data recording can be found [**here**](https://github.com/wpumacay/RoboND-DeepLearning-Project/tree/master/data/schedules). After loading the schedules, we can request a full recording of all of them, and wait for some hours to get our large batches of data.

Using this approach, we gathered a training set of **150000** training examples, which we preprocessed with a modified version of the preprocessing script provided ( link [**here**](https://github.com/wpumacay/RoboND-QuadRotor-Unity-Simulator/blob/master/preprocess_ims.py) ), which gives us a resulting dataset of **154131** ( including the provided initial training-only dataset ).

We chose this amount of data because we wanted to try a bigger endoder architecture, which is based on the VGG-B configuration ( 13 layers ). Some colleagues in the lab have worked with these architectures, and as they suggested, some bigger dataset would be required because of the deeper architecture. 

To make sure that more data would be needed I ran some initial tests using the 3 architectures I will show in the next sections, and could not achieve the required score by a small margin ( 0.35 final score in the initial experiments ). Based on the false positive and false negatives returned we came to the conclussion that we needed more data, specially more data for 3 scenarios ( already suggested in the lectures ) :

*   Data with the target visible, and with a big crowd.
*   Data with the target visible but very far.
*   Data while in standard patrol ( mostly target not visible ).

With the new bigger training dataset we could train our 3 network architectures and all of them got a passing final score ( around 0.47 for each one ).

## **Network architecture and implementation** <a id='network_architecture'></a>

<!-- RUBRIC POINT 1 -->
<!--The student clearly explains each layer of the network architecture and the role that it plays in the overall network. The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data. Any choice of configurable parameters should also be explained in the network architecture.

The student shall also provide a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.-->

<!-- RUBRIC POINT 3 -->
<!--The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.-->

We implemented three different FCN architectures using **convolutional layers**, **1x1 convolutions**, **upsampling**, **skip connections** and **max-pooling layers** ( this last one in the VGG based model ). Next, we explain each of these components :

### _**Convolutional layers**_

Convolutional layers are a special type of layers that operates by convolving filters over an input volume. This convolution operator is basically a dot-product of these filters over a portion of the input volume, and then sliding this **receptive field** over the entire input volume to get the resulting output volume of the operation.

The following image ( from Stanford's cs231n great [**lecture**](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf) on convolutional networks ) shows the overview of this process.

![CONVOLUTIONAL LAYER OVERVIEW 1][img_cs231n_cnn_basics_1]

The resulting output volume is generated by applying all the filters in the convolutional layer and stacking the resulting activation maps into a single output volume, process that is shown in the following image ( again, from Stanford's cs231n [**lecture 5**](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf) )

![CONVOLUTIONAL LAYER OVERVIEW 2][img_cs231n_cnn_basics_2]

### _**1x1 Convolutions**_

1x1 convolutions are convolution layers with filters of kernel 1x1 and strides of 1. The importance can be a bit counter intuitive, but as explained in the lectures, and in [**this**](https://www.coursera.org/lecture/convolutional-neural-networks/networks-in-networks-and-1x1-convolutions-ZTb8x) video ( from Andrew Ng's course in deep learning ), there are some key aspects that make the use 1x1 convolutions a good resource to use in network architectures :

*   It's essentially the same as a fully connected layer, but it keeps the spatil dimensions in the output volume by not flattenig. 
*   They provide a way add non-linearity without adding many parameters. We have to remember that a convolutional layer is followed by an activation function ( ReLU, for example ), which is the same case for 1x1 convolutional layers.
*   They can help increse or decrease the dimensionality of our working volumes by just setting the required number of filters.

The following figure shows an example of dimensionality reduction of our volumes by using 1x1 convolutions ( image from the slides of [**this**](https://www.coursera.org/lecture/convolutional-neural-networks/networks-in-networks-and-1x1-convolutions-ZTb8x) by Andrew Ng ).

![1x1 CONVOLUTIONS][img_1x1_convolutions]


### _**Upsampling**_

Upsampling consists on scaling an output volume to a bigger size. In the context of our volumes, we scale the size of the width and height by an upsample factor, effectively increasing the size.

These are some methods we can use to upsample a volume :

*   Unpooling : basically apply a reverse operation to the classic pooling operation, like in the following image ( from [**this**](https://youtu.be/nDPWywWRIRo?t=18m44s) cs231n lecture )

    ![img_cs231n_unpooling]

*   Transpose convolutions : this is a 'transpose' operation to the convolution operation. Basically, the kernels are scaled by the factors given by the input volume to upsample, instead of the other way around. Then, the results are combined in an output volume that is upscaled according to the stride used. The following image depicts this operation ( agains, from [**this**](https://youtu.be/nDPWywWRIRo?t=25m29s) cs231n lecture )

    ![img_cs231n_transpose_convolutions]

*   Resampling + Interpolation : which is to resize the original input volume to the required size, and interpolate the values according to some interpolation method, like bilinear interpolation. This is commonly used when scaling an image in an image edition software. The [**bilinear upsampling**] method is the one used in the [**utils**](https://github.com/wpumacay/RoboND-DeepLearning-Project/blob/master/code/utils/separable_conv2d.py) provided.

### _**Skip connections**_

Skip connections consist on combining the volumes from previous layers ( like in the encoder's layers ) to the volumes in the decoder. This allows to include finner details to from previous layers into the last layers of our model. As described in [1], they make use of skip connections in their models, as shown in the following figure ( taken from the paper ).

![img_fcn_paper_skip_connections]

The results of applying skip connections is that finner details are obtained in the output volume, as shown in [1], which is shown in the following figure ( from the paper ).

![img_fcn_paper_skip_connections_importance]

We make use of skip connections in our model with this property in mind.

### _**Max pooling**_

Pooling consists of downsampling a volume ( reducing the dimensionality of the volume ) by means of combining the elements of the volume into a single number over a certain receptive field ( a region of certain size ). This is similar as convolution, but instead of making use of a linear operation by matrix multiplies, we are sliding and taking a single value from the sliding region, which effectively reduces the size of the volume.

Pooling can be achieved by Average-pooling ( the operation over the receptive field is an **average** operation ) or by Max-pooling ( the operation over the receptive field is a **max** ). Max-pooling is depicted in the following figure ( from [**this**](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf) cs231n lecture ) :

![img_cs231n_max_pooling]

We use max-pooling in our VGG based architecture by using some pooling layers in between convolutional layers.

### _**FCNs models**_

Finally, the model architectures created are based on the previous described operations, and they are :

#### _**Model 1**_

![SIMPLE ARCHITECTURE 1][img_fcn_architecture_1]

#### _**Model 2**_

![SIMPLE ARCHITECTURE 2][img_fcn_architecture_2]

#### _**Model 3**_

![SIMPLE ARCHITECTURE 3][img_fcn_architecture_3]

## **Hyperparameters tuning** <a id='hyperparameters_tuning'></a>

<!-- RUBRIC POINT 2 -->
<!--The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

    Epoch
    Learning Rate
    Batch Size
    Etc.

All configurable parameters should be explicitly stated and justified. -->

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
*   [**Stanford cs231n notes on convolutional networks**](https://cs231n.github.io/convolutional-networks/)
*   [**Stanford cs231n lecture 5 notes**](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf)
*   [**Andrew Ng on 1x1 convolutions - coursera deeplearning.ai course**](https://www.coursera.org/lecture/convolutional-neural-networks/networks-in-networks-and-1x1-convolutions-ZTb8x)
