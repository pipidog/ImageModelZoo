from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a Xception from scratch, reference https://arxiv.org/abs/1610.02357

    * Note1:
    To use the original version of Xception, just use the function: XceptionNet().
    Since Xception Net is a fixed structure, you don't need to specified any parameter
    except the input_shape, n_classes and l2_weight, e.g.:
        model = XceptionNet(input_shape = (299,299,2), n_classes = 1000, l2_weight = 1e-4)

    * Note2:
    The original paper of Xception Net is only designed for ImageNet and doesn't 
    provide any recommendation for other possible architectures for smaller datasets
    such as cifar10 or Tiny ImageNet. 

    The original Xception is constituted via four parts: Enter Flow, Middle Flow, 
    Exit Flow and finally a global average pooling for regression. The Enter flow
    and Exit flow are somewhat artificial and have so simple rules to extend. An important
    feature of the enter and exit part is that all the resolution downsampling 
    (using Maxpooling) appear here. The middle flow are just repeat of XceptionBlocks 
    with the filter number identical to the last filter number of enter flow. 

    In order to make Xception Net compatible to other conditions, I provide other
    version of ModifyXception Net. The architecture is bascially identical to the
    original Xception except the enter flow. 

    Instead of using maxpooling for downsampling, I replace it with Conv with stride = 2.
    Therefore You just need to specify the filter numbers and whether to perform
    downsampling using stride 2 of each layer in the Enter flows. In most cases, such 
    modification will not harm your performance but make Xception Net more flexible
    and easy to define. 

    In the following, we will introduce the arguments for ModifyXceptionNet
    
{ Arguments }
    input_shape: a tuple w/ 3 elements
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: an integer
        number of classes in dataset

    enter_flow: tuple of tuple
        Specify the enter flow layers. Each layer is constituted by a Conv2D-BN-Relu
        with kernel size (3x3). Therefore, you only need to specify its filter number
        and whether to perform stride 2 downsampling. e.g.
        
        enter_flow = ((32, False), (64, False), (128, True))

        means there are three convolutions with filter 32, 64 and 128 respectively.
        The last tuple (128, True) comes with a True means this layer will use stride = 
        for downsampling.

     
     middle_flow_repeat: integer
        This part is exactly identical to the middle flow part of original paper. The
        filter size is by default identical to the last layer in enter flow. Therefore
        you don't have to specify any informaiton except the number of repeats of the 
        Xception blocks. 

    exit_flow: tuple of tuple
        The exit flow is identical to the original paper. The first block performs a
        maxpooling for downsampling and the next block are two SeparableConv2D. Therefore
        there are four filter numbers you need to specify.

    l2_weight: float, default = 1e-4
        l2 penality add to all Conv layers.

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    * For small datasets, e.g. cifar10, cifar100, try: 
    model = ModifyXceptionNet(input_shape = (32,32,3), n_classes = 10, 
            enter_flow = ((32, False), (64, True)), middle_flow_repeat = 8,
            exit_flow = (128, 128, 256, 256), l2_weight = 1e-4)
    (it will give you roughly the same number of parameters as ResNet 20)
    
    model = ModifyXceptionNet(input_shape = (32,32,3), n_classes = 10, 
            enter_flow = ((32, False), (64, True), (128, True)), middle_flow_repeat = 6,
            exit_flow = (128, 128, 128, 128), l2_weight = 1e-4)
    (it will give you roughly the same number of parameters as ResNet 34)

    * For larger datasets, e.g. ImageNet, try
    model = XceptionNet()  (the original version of Xception)
    
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

def XceptionNet(input_shape = (299,299,3), n_classes = 1000, l2_weight = 1e-4):
    x_in = layers.Input(shape = input_shape)
    # Entry Flow
    x = ConvBlocks.BNConv(x_in, 32, kernel_size = (3,3), strides=(2,2), l2_weight = l2_weight, has_act = True)
    x = ConvBlocks.BNConv(x, 64, kernel_size = (3,3), strides=(1,1), l2_weight = l2_weight, has_act = True)
    x = ConvBlocks.SepConvMaxPoolBlock(x, (128,128), l2_weight = l2_weight, front_relu= False)
    x = ConvBlocks.SepConvMaxPoolBlock(x, (256,256), l2_weight = l2_weight, front_relu= True)
    x = ConvBlocks.SepConvMaxPoolBlock(x, (728,728), l2_weight = l2_weight, front_relu= True)
    # Middle Flow
    x = ConvBlocks.SepConvBlock(x, 728, repeats = 8, l2_weight = 1e-4)
    # Exit Flow
    x = ConvBlocks.SepConvMaxPoolBlock(x, (728, 1024), l2_weight = l2_weight, front_relu= True)
    x = ConvBlocks.SepConv(x, 1536, kernel_size = (3,3), strides = (1,1), l2_weight = l2_weight, relu = 'back')
    x = ConvBlocks.SepConv(x, 2048, kernel_size = (3,3), strides = (1,1), l2_weight = l2_weight, relu = 'back')
    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

def ModifyXceptionNet(input_shape = (299,299,3), n_classes = 1000, 
            enter_flow = ((32, False), (64, False), (128, True)), middle_flow_repeat = 8,
            exit_flow = (128, 128, 128, 128), l2_weight = 1e-4):
    x_in = layers.Input(shape = input_shape)
    # Entry Flow
    x = x_in
    for n, (filters, downsampling) in enumerate(enter_flow):
        strides = (2,2) if downsampling else (1,1)
        x = ConvBlocks.BNConv(x, filters, kernel_size = (3,3), strides=strides, l2_weight = l2_weight, has_act = True)
    # Middle Flow
    x = ConvBlocks.SepConvBlock(x, filters, repeats = middle_flow_repeat, l2_weight = 1e-4)

    # Exit flow
    x = ConvBlocks.SepConvMaxPoolBlock(x, (exit_flow[0],exit_flow[1]), l2_weight = l2_weight, front_relu= True)
    x = ConvBlocks.SepConv(x, exit_flow[2], kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4, relu = 'back')
    x = ConvBlocks.SepConv(x, exit_flow[3], kernel_size = (3,3), strides = (1,1), l2_weight = 1e-4, relu = 'back')

    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

if __name__ == '__main__':
    model = ModifyXceptionNet(input_shape = (32,32,3), n_classes = 10, 
            enter_flow = ((32, False), (64, True), (128, True)), middle_flow_repeat = 6,
            exit_flow = (128, 128, 128, 128), l2_weight = 1e-4)
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)

    



