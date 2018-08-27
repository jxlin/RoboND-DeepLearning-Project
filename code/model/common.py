
import os
import glob
import sys

from scipy import misc
import numpy as np

import tensorflow as tf
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models
from tensorflow import image


# import utils from other folder
sys.path.insert( 0, '../' )
from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools

# Convolutional Layers ###################################################

"""
Makes a separable convolution layer with batch normalization
"""
def separableConv2dBatchnorm( input_layer, filters, strides=1 ) :
    output_layer = SeparableConv2DKeras( filters = filters, kernel_size = 3, 
                                         strides = strides,
                                         padding='same', activation='relu' )( input_layer )
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

"""
Makes a normal convolution layer with batch normalization ( used for 1x1 convolutions )
"""
def conv2dBatchnorm( input_layer, filters, kernel_size = 3, strides = 1 ) :
    output_layer = layers.Conv2D( filters = filters, kernel_size = kernel_size, 
                                  strides = strides, padding='same', 
                                  activation='relu')( input_layer )
    
    output_layer = layers.BatchNormalization()( output_layer ) 
    return output_layer

# Upsampling #############################################################
def bilinearUpsample( input_layer ):
    output_layer = BilinearUpSampling2D( ( 2, 2 ) )( input_layer )
    return output_layer

# Max pooling ############################################################
def maxPoolingLayer( inputs, pSize = ( 2, 2 ), pStrides = ( 2, 2 ) ) :
    _layer = layers.MaxPooling2D( pool_size = pSize, strides = pStrides )( inputs )
    return _layer

# Encoder-Decoder creation ###############################################

def encoderBlock( input_layer, filters, strides ) :
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separableConv2dBatchnorm( input_layer, filters, strides )
    return output_layer

def decoderBlock( small_ip_layer, large_ip_layer, filters ) :
    # Upsample the small input layer using the bilinear_upsample() function.
    _upsampled_ip_layer = bilinearUpsample( small_ip_layer )
    # Concatenate the upsampled and large input layers using layers.concatenate
    _concat_layer = layers.concatenate( [ _upsampled_ip_layer, large_ip_layer ] )
    # Add some number of separable convolution layers
    output_layer = separableConv2dBatchnorm( _concat_layer, filters )
    
    return output_layer

# Extra utils ############################################################
def showShape( mlayer, name = '', show = True ) :
    _str = name + ' : ' + str( mlayer.get_shape().as_list() )
    if show :
        print( _str )
    return _str
