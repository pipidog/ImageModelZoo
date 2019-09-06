from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a Xception Net from scratch, reference https://arxiv.org/abs/1610.02357

    Xception Net has the following structure:
    part 1: two conv, the first conv performs a stride for downsampling (optional)
    part 2: several sconv-sconv-maxpooling block for downsamping
    part 3: repeat xception blocks for several times 
            (number filter is fixed to the last layer in maxpool blocks )
    part 4: another sconv-sconv-maxpooling block for downsamping
    part 5: two sconv-sconv
    part 6: global average pooling-dense-softmax

    Therefore, to specify a Xception Net, you need to input their corresponding parameters

    part 1: input the number of filter of the first two conv and whether to perform downsampling
    part 2: number of sconv-sconv-maxpooling blocks (at least 1 block) and their corresponding filters
    part 3: the number of repeats
    part 4 & part 5: 4 number filters of each sconv layer

    So it is not difficult to define a Xception Net
    
{ Arguments }
    input_shape: a tuple w/ 3 elements
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: an integer
        number of classes in dataset

    first_two_conv: (int, int, boolen)
        number of filters of the first two conv layer and whether to perform a stride = 2
        downsampling in the first layer.

    maxpool_block:
        each maxpool-block is constituted by sconv-sconv-maxpool where the number of filters
        are the same for both sconv. Therefore, you only need one parameter, i.e. the number
        of filters, to specify a maxpool_block. 
        
        e.g. (64, 128)

        means there are two maxpool_block with filters = 64 and 128 respectively.

    middle_flow_repeat: int
        the middle flow part is a series of xception blocks (sconv-sconv-sconv) with 
        filter size identical to the last layer in maxpool_block. therefore, you only 
        need a single parameter to define this part, e.g. 8, means respeat 8 times 
        of xception blocks.

    exit_flow_filters:
        the exit flow of Xception is sconv-sconv-maxpool-sconv-sconv. Therefore, you 
        need 4 filter numbers for each sconv.

    l2_weight: float, default = 1e-4
        l2 penality add to all Conv or SConv layers.

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    * The original paper of Xception doesn't offer any suggestion on model parameters
    other than ImageNet. Therefore, I suggest the following parameters which will give
    you a model that roughly the same number of parameters as ResNet. 

    Given the fact that Xception outperforms ResNet on ImageNet slightly when model 
    complexity is similar, hopefully you can still get similar results when using 
    the following parameters.
    
    # For cifar10 or cifar100
    model = XceptionNet(input_shape = (32,32,3), n_classes = 10, 
            first_two_conv = (32, 64, False), maxpool_block = (64, 64),
            middle_flow_repeat = 8,
            exit_flow = (128, 128, 256, 256), l2_weight = 1e-4)
    (it will give you roughly the same number of parameters as ResNet 20, 0.27M v.s. 0.30M)

    # For cifar10 or cifar100
    model = XceptionNet(input_shape = (32,32,3), n_classes = 10, 
            first_two_conv = (32, 64, False), maxpool_block = (64, 128),
            middle_flow_repeat = 6,
            exit_flow = (128, 128, 128, 128), l2_weight = 1e-4)
    (it will give you roughly the same number of parameters as ResNet 32, 0.46M v.s 0.48M)

    * For larger datasets, e.g. ImageNet, try (obtained from original paper)
    model = XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
            first_two_conv = (32, 64, True), maxpool_block = (128, 256, 728),
            middle_flow_repeat = 8,
            exit_flow = (728, 1024, 1536, 2048), l2_weight = 1e-4)    
'''

class ConvBlocks:
    @staticmethod
    def BNConv(x, filters, kernel_size, strides, l2_weight = 1e-4, has_act = True):
        x = layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, 
                padding = 'same', kernel_regularizer=l2(l2_weight))(x)
        x = layers.BatchNormalization()(x)
        if has_act:
            x = layers.ReLU()(x)          
        return x

    @staticmethod
    def SepConv(x, filters, kernel_size, strides = (1,1), l2_weight = 1e-4, relu = 'front'):
        assert (relu == 'front') or (relu == 'back') or (relu == None) 
        if relu == 'front':
            x = layers.ReLU()(x)
        x = layers.SeparableConv2D(filters, kernel_size = kernel_size, strides = strides, 
                padding = 'same', kernel_regularizer=l2(l2_weight))(x)
        x = layers.BatchNormalization()(x)
        if relu == 'back':
            x = layers.ReLU()(x)
        return x
    
    @classmethod
    def SepConvMaxPoolBlock(cls, x_in, filters, l2_weight = 1e-4, front_relu = True):
        assert type(front_relu) == type(True) 
        assert len(filters) == 2
        x = cls.SepConv(x_in, filters[0], (3,3), l2_weight = l2_weight, relu = 'front' if front_relu else None)
        x = cls.SepConv(x, filters[1], (3,3), l2_weight = l2_weight, relu = 'front')
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding = 'same')(x)
        x_skip = cls.BNConv(x_in, filters[1], kernel_size = (1,1), strides = (2,2), l2_weight = l2_weight, has_act = False)
        x = layers.Add()([x_skip, x])
        return x
    
    @classmethod
    def SepConvBlock(cls, x_in, filters, repeats, l2_weight = 1e-4):
        for n in range(repeats):
            x = cls.SepConv(x_in, filters, kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4, relu = 'front')
            x = cls.SepConv(x, filters, kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4, relu = 'front')
            x = cls.SepConv(x, filters, kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4, relu = 'front')
            x_in = layers.Add()([x_in, x])
        return x_in

def XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
            first_two_conv = (32, 64, True), maxpool_block = (128, 256, 728),
            middle_flow_repeat = 8,
            exit_flow = (728, 1024, 1536, 2048), l2_weight = 1e-4):
    x_in = layers.Input(shape = input_shape)
    # Entry Flow
    x = ConvBlocks.BNConv(x_in, first_two_conv[0], kernel_size = (3,3), 
            strides = (2,2) if first_two_conv else (1,1), 
            l2_weight = l2_weight, has_act = True)
    x = ConvBlocks.BNConv(x, first_two_conv[1], kernel_size = (3,3), 
            strides = (1,1), l2_weight = l2_weight, has_act = True)
    for n, filters in enumerate(maxpool_block):
        x = ConvBlocks.SepConvMaxPoolBlock(x, (filters, filters), l2_weight = l2_weight, front_relu=(n!=0))

    # middle flow 
    x = ConvBlocks.SepConvBlock(x, maxpool_block[-1], repeats = middle_flow_repeat, l2_weight = 1e-4)

    # Exit Flow
    x = ConvBlocks.SepConvMaxPoolBlock(x, exit_flow[0:2], l2_weight = l2_weight, front_relu= True)
    x = ConvBlocks.SepConv(x, exit_flow[2], kernel_size = (3,3), strides = (1,1), l2_weight = l2_weight, relu = 'back')
    x = ConvBlocks.SepConv(x, exit_flow[3], kernel_size = (3,3), strides = (1,1), l2_weight = l2_weight, relu = 'back')

    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

if __name__ == '__main__':
    model = XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
            first_two_conv = (32, 64, True), maxpool_block = (128, 256, 728),
            middle_flow_repeat = 8,
            exit_flow = (728, 1024, 1536, 2048), l2_weight = 1e-4)   
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)

    



