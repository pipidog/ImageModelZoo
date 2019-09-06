from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a MobileNetV2 from scratch, reference https://arxiv.org/abs/1801.04381

{ Arguments }

    input_shape: 
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: 
        number of classes in dataset

    first_block_filters:
        In MobileNet V2, there is initial Conv2D layer before entering the main body.
        The original paper recommend 32.

    last_block_filters:
        In MobileNet V2, there is a 1x1 Conv2D before entering the final classifier 
        layer (you can consider it as the units of a Dense layer). The original paper
        use 1280 for ImageNet. For dataset with less classes, you should use a small 
        value, such as 128 for cifar10.  
    
    alpha: 0~1
        MobileNet use a parameter alpha to control the model size. The number of channels
        on all Conv2D will multiple this parameter so the model size will be reduced if
        alpha < 1.0.  
    
    n_downsampling: 0~5
        The original MobileNet contains 5 Strides =2 Conv2D layers. So the spatial resolutions
        will be reduced 32 times on its row and columns. However, for low resolution dataset
        such as cifar10, you don't want the downsampling too aggressive. If you set 
        this value to, say 3, only the last 3 downsampling will be performed. All the others
        will use stride = 1 instead.


{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    * for cifar10, cifar100, try:
    MobileNetV2(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_block_filters = 128)
    * for Tiny ImageNet, try:
    MobileNetV2(input_shape = (64,64,3), n_classes = 200, alpha = 0.5, n_downsampling = 4, last_block_filters = 256)
    * for ImageNet, try:
    MobileNetV2(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_block_filters = 1280)

'''

# make sure the number of filters is mulipler of divisor
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
        x = layers.ReLU(max_value = 6.0)(x)        
        return x

    @classmethod # Depthwise Conv + BN
    def DWBNConv(cls, x_in, depth_multiplier = 1, kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4):
        x = layers.DepthwiseConv2D(kernel_size = (3,3), strides= strides, depth_multiplier=1, 
            padding='same',  kernel_initializer='he_normal', kernel_regularizer= l2(l2_weight))(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value = 6.0)(x)       
        return x

    @classmethod # Bottleneck block 
    def Bottleneck(cls, x_in, expansion_factor, output_channel, strides = (1,1), l2_weight = 1e-4):
        input_channel = K.int_shape(x_in)[-1]
        x = cls.BNConv(x_in, input_channel*expansion_factor, l2_weight = l2_weight)
        x = cls.DWBNConv(x, strides = strides, l2_weight = l2_weight)
        x = cls.BNConv(x, output_channel, l2_weight = l2_weight)
        if K.int_shape(x_in) == K.int_shape(x):
            x = layers.Add()([x_in, x])
        return x

def MobileNetV2(input_shape = (224,224,3), n_classes = 1000, first_block_filters = 32, 
                last_block_filters = 1280, alpha = 1.0, n_downsampling = 5, l2_weight = 1e-4):
    
    assert 0 <= n_downsampling <= 5
    dim_reduction = [1]*(5-n_downsampling)+[2]*(n_downsampling)

    if first_block_filters is None:
        first_block_filters = make_divisible(32 * alpha, 8)
    if last_block_filters is None:
        if alpha > 1.0:
            last_block_filters = make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
    
    # initial layer
    x_in = layers.Input(shape = input_shape)
    x = ConvBlocks.BNConv(x_in, first_block_filters, kernel_size = (3,3), 
            strides = (dim_reduction[0],)*2, l2_weight = l2_weight)

    # Bottleneck layers
    x = ConvBlocks.Bottleneck(x, expansion_factor = 1, output_channel = int(16*alpha),  
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(24*alpha),  
            strides=(dim_reduction[1],)*2, l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(24*alpha),  
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(32*alpha),  
            strides=(dim_reduction[2],)*2, l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(32*alpha),  
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(32*alpha),  
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(64*alpha),  
            strides=(dim_reduction[3],)*2, l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(64*alpha),  
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(64*alpha),  
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(64*alpha),  
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(96*alpha),  
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(96*alpha),  
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(96*alpha),  
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(160*alpha), 
            strides=(dim_reduction[4],)*2, l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(160*alpha), 
            strides=(1,1), l2_weight = l2_weight)
    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(160*alpha), 
            strides=(1,1), l2_weight = l2_weight)

    x = ConvBlocks.Bottleneck(x, expansion_factor = 6, output_channel = int(320*alpha), 
            strides=(1,1), l2_weight = l2_weight)
   
    # output for classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1,1,K.int_shape(x)[-1]))(x)
    x = ConvBlocks.BNConv(x, last_block_filters)
    x = layers.Flatten()(x)
    x_out = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs = x_in, outputs = x_out)
    return model


if __name__ == '__main__':
    model = MobileNetV2(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_block_filters = 100)
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)







