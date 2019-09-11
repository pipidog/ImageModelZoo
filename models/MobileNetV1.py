from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a MobileNetV2 from scratch, reference https://arxiv.org/abs/1704.04861

{ Arguments }

    input_shape: 
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: 
        number of classes in dataset

    first_block_filters:
        In MobileNet V1, there is initial Conv2D layer before entering the main body.
        The original paper recommend 32.
    
    alpha: 0~1
        MobileNet use a parameter alpha to control the model size. The number of channel
        on all Conv2D will multiple this parameter so the model size will be reduced if
        alpha < 1.0.  
    
    n_downsampling: 0~6
        The original MobileNet contains 6 Strides =2 Conv2D layers. So the spatial resolutions
        will be 64 times smaller on its row and columns. However, for low resolution dataset
        such as cifar10, you don't want the downsampling too aggressive. If you set 
        this value to, say 3, only the last 3 downsampling will be performed. All the others
        will use stride = 1 instead.

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    * for cifar10, cifar100, try:
    MobileNetV1(input_shape = (32,32,3), n_classes = 10, first_block_filters = 32, 
                    alpha = 0.25, n_downsampling = 3,  l2_weight =1e-4)
    * for Tiny ImageNet, try:
    MobileNetV1(input_shape = (64,64,3), n_classes = 200, first_block_filters = 32, 
                    alpha = 0.5, n_downsampling = 4,  l2_weight =1e-4)
    * for ImageNet, try:
    MobileNetV1(input_shape = (224,224,3), n_classes = 1000, first_block_filters = 32, 
                    alpha = 1.0, n_downsampling = 6,  l2_weight =1e-4)
'''


# make sure the number of filters is a mulipler of divisor (usually pow(2,n), such as 8)
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBlocks:
    @classmethod  # Conv + BN
    def BNConv(cls, x_in, filters, kernel_size = (1,1), strides = (1,1), l2_weight = 1e-4):
        x = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
            padding = 'same',  kernel_initializer='he_normal', kernel_regularizer = l2(l2_weight))(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)        
        return x

    @classmethod # Depthwise Conv + BN
    def DWBNConv(cls, x_in, depth_multiplier = 1, kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4):
        x = layers.DepthwiseConv2D(kernel_size = (3,3), strides= strides, depth_multiplier=1, 
            padding='same',  kernel_initializer='he_normal', kernel_regularizer= l2(l2_weight))(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)       
        return x

    @classmethod # Depthwise Separable
    def DSBNConv(cls, x_in, filters, strides = (1,1), l2_weight = 1e-4):
        x = cls.DWBNConv(x_in, depth_multiplier=1, kernel_size = (3,3), strides = strides, l2_weight = l2_weight)
        x = cls.BNConv(x, filters = filters, kernel_size = (1,1), strides = (1,1), l2_weight = l2_weight)
        return x

def MobileNetV1(input_shape, n_classes, first_block_filters = 32, alpha = 1.0, n_downsampling = 6,  l2_weight =1e-4):
    assert 0 <= n_downsampling <= 6
    dim_reduction = [1]*(6-n_downsampling)+[2]*(n_downsampling)
    first_block_filters = make_divisible(32 * alpha, 8)

    # initial layer
    x_in = layers.Input(shape = input_shape)
    x = ConvBlocks.BNConv(x_in, filters = first_block_filters, kernel_size=(3,3), 
            strides = (dim_reduction[0],)*2, l2_weight = l2_weight)

    # Depthwise Separable part
    x = ConvBlocks.DSBNConv(x, filters = int(64*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(128*alpha), strides = (dim_reduction[1],)*2, l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(128*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(256*alpha), strides = (dim_reduction[2],)*2, l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(256*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (dim_reduction[3],)*2, l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(512*alpha), strides = (1,1), l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(1024*alpha), strides = (dim_reduction[4],)*2, l2_weight = l2_weight)
    x = ConvBlocks.DSBNConv(x, filters = int(1024*alpha), strides = (dim_reduction[5],)*2, l2_weight = l2_weight)

    # output for classification
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(n_classes, activation='softmax', kernel_regularizer = l2(l2_weight))(x)

    model = Model(inputs = x_in, outputs = x_out)
    return model


if __name__ == '__main__':
    # e.g. for cifar10, an appropriate choice could be:
    model = MobileNetV1(input_shape = (32,32,3), n_classes = 10, alpha = 0.25, n_downsampling = 3)
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)







