import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from config import config

# (16, 139, 64)
config_model = config['model_name']


@keras.saving.register_keras_serializable()
class TCSConv(layers.Layer):
    """
    An implementation of Time-channel Seperable Convolution
    
    **Arguments**
    out_channels : int
        The requested number of output channels of the layers
    kernel_size : int
        The size of the convolution kernel
    """
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        self.depthwise_conv = layers.DepthwiseConv1D(kernel_size, padding='same') # 16, 65, 128
        self.pointwise_conv = layers.Conv1D(out_channels, kernel_size=1) # 16, 65, 128 -> 16, 65, 64

    def call(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
    

@keras.saving.register_keras_serializable()
class SubBlock(layers.Layer):

    """
    An implementation of a sub-block that is repeated R times

    **Arguments**
    out_channels : int
        The requested number of output channels of the layers
    kernel_size : int
        The size of the convolution kernel

    residual : None or torch.Tensor
    Only applicable for the final sub-block. If not None, will add 'residual' after batchnorm layer
    """
    def __init__(self, out_channels, kernel_size):
        super().__init__()
        self.tcs_conv = TCSConv(out_channels, kernel_size)
        self.bnorm = layers.BatchNormalization()
        self.dropout = layers.Dropout(rate=0.5)

    def call(self, x, residual=None):
        x = self.tcs_conv(x)
        x = self.bnorm(x)

        # apply the residual if passed
        if residual is not None:
            x = x + residual
        
        x = layers.ReLU()(x)
        x = self.dropout(x)

        return x
 

@keras.saving.register_keras_serializable()
class MainBlock(layers.Layer):
    """
    An implementation of the residual block containing R repeating sub-blocks

    **Arguments**
    out_channels : int
        The requested number of output channels of the sub-blocks
    kernel_size : int
        The size of the convolution kernel
    R : int
        The number of repeating sub-blocks contained within this residual block

    residual : None or torch.Tensor
        Only applicable for the final sub-block. If not None, will add 'residual' after batchnorm layer
    """

    def __init__(self, out_channels, kernel_size, R):
        super().__init__()
        self.residual_pointwise = layers.Conv1D(out_channels, kernel_size=1) # 16, 65, 128 -> 16, 65, 64
        self.residual_batchnorm = layers.BatchNormalization()

        self.sub_blocks = []

        for i in range(R):
            self.sub_blocks.append(SubBlock(out_channels, kernel_size))
    
    def call(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)

        for i, layer in enumerate(self.sub_blocks):
            if (i+1) == len(self.sub_blocks): # compute the residual in the final sub-block
                x = layer(x, residual)
                # print(f'SubBlock {i}: ', x.shape)
            else:
                x = layer(x)
                # print(f'SubBlock {i}: ', x.shape)
        return x
        

@keras.saving.register_keras_serializable()
class MatchboxNet(keras.Model):
    """
    An implementation of MatchboxNet

    **Arguments**
    B : int
        The number of residual blocks in the model
    R : int
        The number of sub-blocks within each residual block
    C : int
        The size of the output channels within a sub-block
    kernel_sizes : None or list
        If None, kernel sizes will be assigned to values used in the paper. Otherwise kernel_sizes will be used
        len(kernel_sizes) must equal the number of blocks (B)
    NUM_CLASSES : int
        The number of classes in the dataset (i.e. number of keywords.)
    """

    def __init__(self, B, R, C, kernel_sizes=None, NUM_CLASSES=25, name="MatchboxNet"):
        super().__init__(name=name)
        if not kernel_sizes:
            kernel_sizes = [11+i*2 for i in range(1, B+1)]

        self.prologue_conv1 = layers.Conv1D(128, kernel_size=11, strides=2, input_shape=(139, 64))
        self.prologue_bnorm1 = layers.BatchNormalization()

        self.blocks = []
        for i in range(B):
            self.blocks.append(MainBlock(C, kernel_size=kernel_sizes[i], R=R))

        # the epilogue layers
        self.epilogue_conv1 = layers.Conv1D(128, kernel_size=29, dilation_rate=2)
        self.epilogue_bnorm1 = layers.BatchNormalization()
        
        self.epilogue_conv2 = layers.Conv1D(128, kernel_size=1)
        self.epilogue_bnorm2 = layers.BatchNormalization()

        self.epilogue_conv3 = layers.Conv1D(NUM_CLASSES, kernel_size=1)
        self.epilogue_globalavgepool = layers.GlobalAveragePooling1D()

    def call(self, x):
        # prologue block
        # print('Input Layer: ', x.shape)
        x = self.prologue_conv1(x) 
        x = self.prologue_bnorm1(x)
        x = layers.ReLU()(x)
        # print('prologue_conv1:', x.shape)

        # intermediate block
        for layer in self.blocks:
            x = layer(x)
        # print('MainBlock: ', x.shape)
        # epilogue blocks
        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)
        x = layers.ReLU()(x)
        # print('epilogue block 1: ', x.shape)

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)
        x = layers.ReLU()(x)
        # print('epilogue block 2: ', x.shape)

        x = self.epilogue_conv3(x)
        # print('epilogue block 3: ', x.shape)
        x = self.epilogue_globalavgepool(x)
        # print('epilogue glavgpool: ', x.shape)

        x = tf.keras.activations.softmax(x)
        # print(x.shape)
        return x

