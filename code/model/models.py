
import sys
sys.path.insert( 0, '../' )

# import common helper functions
from model.common import *
# import drawing utils for convnet ( just for viz. )
from ext.draw_convnet import *

INDEX_LAYER_NAME = 0
INDEX_LAYER_OBJ = 1
INDEX_LAYER_VOLUME_TYPE = 2
INDEX_LAYER_LAYER_TYPE = 3
INDEX_LAYER_KERNEL_SIZE = 4

INDEX_BATCH = 0
INDEX_WIDTH = 1
INDEX_HEIGHT = 2
INDEX_DEPTH = 3

LAYER_WIDTH = 275
MAX_SHOW_LAYERS = 64

LAYER_TYPE_INPUT = 'INPUT'
LAYER_TYPE_CONVOLUTION = 'CONVOLUTION'
LAYER_TYPE_1X1_CONVOLUTION = '1x1 CONVOLUTION'
LAYER_TYPE_BILINEAR_UPSAMPLING = 'UPSAMPLING'
LAYER_TYPE_MAXPOOLING = 'MAXPOOLING'
LAYER_TYPE_FLATTEN = 'FLATTEN'
LAYER_TYPE_FULLYCONNECTED = 'FULLYCONNECTED'
LAYER_TYPE_SOFTMAX = 'SOFTMAX'

VOLUME_TYPE_INPUT = 'Input'
VOLUME_TYPE_FEATURE_MAPS = 'Feature\nmaps'
VOLUME_TYPE_HIDDEN_UNITS = 'Hidden\nunits'
VOLUME_TYPE_OUTPUT = "Output"

class RModel( object ) :

    def __init__( self, inWidth, inHeight, inDepth, numclasses, lrate = 0.001, savedmodel = None ) :
        super( RModel, self ).__init__()
        # Define some model internals
        self.m_inputs = layers.Input( ( inWidth, inHeight, inDepth ) )
        self.m_numClasses = numclasses
        self.m_layers = []
        self.m_model = None
        self.m_learningRate = lrate
        
        # Drawing internals
        self.m_drawPatches = []
        self.m_drawColors = []
        self.m_drawXdiff = []
        self.m_drawTextList = []
        self.m_drawLocDiff = []
        self.m_drawTopLeftList = []
        self.m_drawFig = None
        self.m_drawAx = None

        # some utils
        self.m_logger = plotting_tools.LoggerPlotter()

        # create the architecture
        if savedmodel :
            self._loadModel( savedmodel )
        else :
            self._buildArchitecture()
            self._buildModel()
    
    def _loadModel( self, savedmodel ) : 
        self.m_model = model_tools.load_network( savedmodel )

    def _saveModel( self, savepath ) :
        model_tools.save_network( self.m_model, savepath )

    def _buildArchitecture( self ) :
        # Override this method
        pass

    def _buildModel( self ) :
        self.m_model = models.Model( inputs = self.m_inputs,
                                     outputs = self.m_layers[-1][INDEX_LAYER_OBJ] )
        self.m_model.compile( optimizer = keras.optimizers.Adam( self.m_learningRate ),
                              loss = 'categorical_crossentropy' )

    def model( self ) :
        return self.m_model

    def layers( self ) :
        return self.m_layers

    def __str__( self ) :
        # print the layers from encoder-decoder structure of our model
        _str = ''
        for i in range( len( self.m_layers ) ) :
            _name  = self.m_layers[i][0]
            _layer = self.m_layers[i][1]
            _str += showShape( _layer, _name, False ) + '\n'
            
        return _str

    def train( self, 
               numEpochs, 
               trainingIterator, 
               validationIterator,
               numTrainingBatchesPerEpoch,
               numValidationBatchesPerEpoch,
               numWorkers ) :

        self.m_model.fit_generator( trainingIterator,
                                    steps_per_epoch = numTrainingBatchesPerEpoch,
                                    epochs = numEpochs,
                                    validation_data = validationIterator,
                                    validation_steps = numValidationBatchesPerEpoch,
                                    callbacks = [ self.m_logger ],
                                    workers = numWorkers )

    def getTrainHistory( self ) :
        return self.m_logger.hist_dict

    def show( self ) :
        # create figure
        self.m_drawFig, self.m_drawAx = plt.subplots()
        # check if already has data
        if len( self.m_drawPatches ) == 0 or len( self.m_drawColors ) == 0 :
            # initialize the drawing layout
            self.m_drawXdiff = []
            self.m_drawLocDiff = []
            self.m_drawTopLeftList = []
            self.m_drawTextList = []
            for i in range( len( self.m_layers ) ) :
                self.m_drawXdiff.append( LAYER_WIDTH if i != 0 else 0 )
                self.m_drawTextList.append( self.m_layers[i][INDEX_LAYER_VOLUME_TYPE] )
                self.m_drawLocDiff.append( [3, -3] )
            
            self.m_drawTopLeftList = np.c_[np.cumsum( self.m_drawXdiff ), np.zeros( len( self.m_layers ) )]
            # make the patches
            for i in range( len( self.m_layers ) ) :
                # add layer to patches
                self._addPatchForLayer( i, self.m_layers[i] )

        self._drawPatches()
        self._drawBottomLabels()

        self.m_drawFig.set_size_inches( 20, 5 )

        plt.tight_layout()
        plt.axis( 'equal' )
        plt.axis( 'off' )
        plt.show()
    
    def _addPatchForLayer( self, layerIndx, layer ) :
        _name = layer[INDEX_LAYER_NAME]
        _layer = layer[INDEX_LAYER_OBJ]
        _type = layer[INDEX_LAYER_VOLUME_TYPE]

        _size = _layer.get_shape().as_list()
        _width = _size[INDEX_WIDTH]
        _height = _size[INDEX_HEIGHT]
        _depth = _size[INDEX_DEPTH]

        add_layer_with_omission( self.m_drawPatches, self.m_drawColors,
                                 size = ( _width, _height ), num = _depth,
                                 num_max = MAX_SHOW_LAYERS,
                                 num_dots = 3,
                                 top_left = self.m_drawTopLeftList[ layerIndx ] + 0 * layerIndx,
                                 loc_diff = self.m_drawLocDiff[ layerIndx ] )
        label( ( self.m_drawTopLeftList[ layerIndx ][0],
                 self.m_drawTopLeftList[ layerIndx ][1] ),
               _type + '\n{}@{}x{}'.format( _depth, _width, _height ) )

    def _drawPatches( self ) :
        for _patch, _color in zip( self.m_drawPatches, self.m_drawColors ) :
            _patch.set_color( _color * np.ones( 3 ) )
            if isinstance( _patch, Line2D ) :
                self.m_drawAx.add_line( _patch )
            else :
                _patch.set_edgecolor( 0.0 * np.ones( 3 ) )
                self.m_drawAx.add_patch( _patch )

    def _drawBottomLabels( self ) :
        for i in range( len( self.m_layers ) ) :
            if i == 0 :
                continue
            _txt = self.m_layers[i][INDEX_LAYER_LAYER_TYPE]
            _size = self.m_layers[i][INDEX_LAYER_OBJ].get_shape().as_list()
            _ksize = self.m_layers[i][INDEX_LAYER_KERNEL_SIZE]
            _w = _size[INDEX_WIDTH]
            _h = _size[INDEX_HEIGHT]

            if ( _txt == LAYER_TYPE_CONVOLUTION or 
                 _txt == LAYER_TYPE_1X1_CONVOLUTION ) :
                _txt += '\n{}x{} kernel'.format( _ksize, _ksize )

            label( [ self.m_drawTopLeftList[i - 1][0] + 100,
                     self.m_drawTopLeftList[i - 1][1] ],
                   _txt, xy_off = [120, -350] )

class RFCNsimpleModel( RModel ) :

    def __init__( self, inWidth, inHeight, inDepth, numclasses, lrate = 0.001 ) :
        super( RFCNsimpleModel, self ).__init__( inWidth, inHeight, inDepth, numclasses, lrate )

    def _buildArchitecture( self ) :
        # ENCODER
        _conv1 = encoderBlock( self.m_inputs, 32, 2 )
        _conv2 = encoderBlock( _conv1, 64, 2 )
        _conv3 = encoderBlock( _conv2, 128, 2 )
        # MIDDLE 1X1 CONVOLUTION
        _mid = conv2dBatchnorm( _conv3, 256, 1 )
        # DECODER
        _tconv1 = decoderBlock( _mid, _conv2, 128 )
        _tconv2 = decoderBlock( _tconv1, _conv1, 64 )
        _z = decoderBlock( _tconv2, self.m_inputs, 32 )
        # OUTPUT LAYER
        _y = layers.Conv2D( self.m_numClasses, 3, activation = 'softmax', padding = 'same' )( _z )

        self.m_layers.append( [ 'input', self.m_inputs, VOLUME_TYPE_INPUT, LAYER_TYPE_INPUT, -1 ] )
        self.m_layers.append( [ 'conv1', _conv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv2', _conv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv3', _conv3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'mid', _mid, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_1X1_CONVOLUTION, 1 ] )
        self.m_layers.append( [ 'tconv1', _tconv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv2', _tconv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'z', _z, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'y', _y, VOLUME_TYPE_OUTPUT, LAYER_TYPE_SOFTMAX, -1 ] )

class RFCNdeeperModel( RModel ) :

    def __init__( self, inWidth, inHeight, inDepth, numclasses, lrate = 0.001 ) :
        super( RFCNdeeperModel, self ).__init__( inWidth, inHeight, inDepth, numclasses, lrate )

    def _buildArchitecture( self ) :
        # ENCODER
        _conv1 = encoderBlock( self.m_inputs, 32, 2 )
        _conv2 = encoderBlock( _conv1, 64, 2 )
        _conv3 = encoderBlock( _conv2, 128, 2 )
        _conv4 = encoderBlock( _conv3, 256, 2 )
        # MIDDLE 1X1 CONVOLUTION
        _mid = conv2dBatchnorm( _conv4, 512, 1 )
        # DECODER
        _tconv1 = decoderBlock( _mid, _conv3, 256 )
        _tconv2 = decoderBlock( _tconv1, _conv2, 128 )
        _tconv3 = decoderBlock( _tconv2, _conv1, 64 )
        _z = decoderBlock( _tconv3, self.m_inputs, 32 )
        # OUTPUT LAYER
        _y = layers.Conv2D( self.m_numClasses, 3, activation = 'softmax', padding = 'same' )( _z )

        self.m_layers.append( [ 'input', self.m_inputs, VOLUME_TYPE_INPUT, LAYER_TYPE_INPUT, -1 ] )
        self.m_layers.append( [ 'conv1', _conv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv2', _conv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv3', _conv3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv4', _conv4, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'mid', _mid, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_1X1_CONVOLUTION, 1 ] )
        self.m_layers.append( [ 'tconv1', _tconv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv2', _tconv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv3', _tconv3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'z', _z, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'y', _y, VOLUME_TYPE_OUTPUT, LAYER_TYPE_SOFTMAX, -1 ] )

class RFCNvggModel( RModel ) :

    def __init__( self, inWidth, inHeight, inDepth, numclasses, lrate = 0.001 ) :
        super( RFCNvggModel, self ).__init__( inWidth, inHeight, inDepth, numclasses, lrate )

    def _buildArchitecture( self ) :
        # ENCODER
        _conv1 = encoderBlock( self.m_inputs, 32, 1 )
        _pool1 = maxPoolingLayer( _conv1 )
        _conv2 = encoderBlock( _pool1, 64, 1 )
        _pool2 = maxPoolingLayer( _conv2 )
        _conv3 = encoderBlock( _pool2, 128, 1 )
        _conv4 = encoderBlock( _conv3, 128, 1 )
        _pool3 = maxPoolingLayer( _conv4 )
        _conv5 = encoderBlock( _pool3, 256, 1 )
        _conv6 = encoderBlock( _conv5, 256, 1 )
        _pool4 = maxPoolingLayer( _conv6 )
        _conv7 = encoderBlock( _pool4, 256, 1 )
        _conv8 = encoderBlock( _conv7, 256, 1 )
        _pool5 = maxPoolingLayer( _conv8 )
        # MIDDLE 1X1 CONVOLUTION
        _mid = conv2dBatchnorm( _pool5, 512, 1 )
        # DECODER
        _tconv1 = decoderBlock( _mid, _pool4, 256 )
        _tconv2 = decoderBlock( _tconv1, _pool3, 256 )
        _tconv3 = decoderBlock( _tconv2, _pool2, 128 )
        _tconv4 = decoderBlock( _tconv3, _pool1, 64 )
        _z = decoderBlock( _tconv4, self.m_inputs, 32 )
        # OUTPUT LAYER
        _y = layers.Conv2D( self.m_numClasses, 3, activation = 'softmax', padding = 'same' )( _z )

        self.m_layers.append( [ 'input', self.m_inputs, VOLUME_TYPE_INPUT, LAYER_TYPE_INPUT, -1 ] )
        self.m_layers.append( [ 'conv1', _conv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'pool1', _pool1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_MAXPOOLING, -1 ] )
        self.m_layers.append( [ 'conv2', _conv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'pool2', _pool2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_MAXPOOLING, -1 ] )
        self.m_layers.append( [ 'conv3', _conv3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv4', _conv4, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'pool3', _pool3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_MAXPOOLING, -1 ] )
        self.m_layers.append( [ 'conv5', _conv5, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv6', _conv6, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'pool4', _pool4, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_MAXPOOLING, -1 ] )
        self.m_layers.append( [ 'conv7', _conv7, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'conv8', _conv8, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_CONVOLUTION, 3 ] )
        self.m_layers.append( [ 'pool5', _pool5, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_MAXPOOLING, -1 ] )
        self.m_layers.append( [ 'mid', _mid, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_1X1_CONVOLUTION, 1 ] )
        self.m_layers.append( [ 'tconv1', _tconv1, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv2', _tconv2, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv3', _tconv3, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'tconv4', _tconv4, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'z', _z, VOLUME_TYPE_FEATURE_MAPS, LAYER_TYPE_BILINEAR_UPSAMPLING, -1 ] )
        self.m_layers.append( [ 'y', _y, VOLUME_TYPE_OUTPUT, LAYER_TYPE_SOFTMAX, -1 ] )