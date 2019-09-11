from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a Xception Net from scratch, reference https://arxiv.org/abs/1610.02357

    The original Xception is only designed for ImageNet. There is no any suggested
    structure for other datasets such as cifar10, cifar100, Tiny ImageNet, etc, which 
    comes with much lower resolutions. 

    Xception Net has the following structure: (see Fig.5 in original paper)
    Entry Flow:     
        Two Conv (first one with stride = 2) and a series of sconv-sconv-maxpooling 
        blocks for downsampling
    Middle Flow: 
        Repeat sconv-sconv-sconv blocks where the filters size is identical to the 
        last layer in Entry flow.
    Exit Flow:
        a sconv-sconv-maxpooling block, two sconv layers and finally a global average
        pooling. 

    Apprently, the entry and exit flows are somewhat arbitary and has no simple 
    perodic structure and all the downsamplings occur here.

    In order to define a Xception-like Network with best flexibility, we design the 
    following API:
    
    => First, you need you define your own entry flow, the entry flow should first built
    by a few conventional layers and followed by a few sconv-sconv-maxpooling blocks.

    It can be expressed by this way, e.g:
    (('conv',32,True), ('sconv_pool',128,128))
    
    ('conv',32,True) means a conventional layer with 32 filters (3x3), True means use
    a stride = 2 for downsampling

    ('sconv_pool', 128,128) means it a scconv-sconv-maxpool blocks, with 128 and 128 
    filters (kernel = 3x3, stride = 1) respectively for each sconv layer  

    => second, you need to define the repeats of the middle flow. since it is simply
    a stack of sconv-sconv-socnv layers with same number of filters (the last layer
    in entry flow), kernel size (3x3) and strides (1,1), it only requires a single
    parameter, i.e. the number of repeats, to define. 

    => thrid, the exit flow. It is built by a few sconv-pool blocks and sconv layers. 
    Therefore, it ca be define by, e.g.:
    (('sconv_pool',728,1024),('sconv',1536)) 

    the definition of 'sconv_pool' is the same with the entry flow. As for the ('sconv', 1536)
    it means a sconv layer with 1536 filters (kernel size = 3x3, stride = 1)

    If you follow the above the API to define a network, you can consider it a Xception-like
    network. 


{ Arguments }
    input_shape: a tuple w/ 3 elements
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: an integer
        number of classes in dataset


    entry_flow: tuple of tuple 
        must defined by 'conv' and 'sconv' e.g.: (('conv',32,True),('sconv_pool',128,128))
        (see summary for its meaning). The first layer must be a 'conv'

    middle_flow_repeat: int
         number of repeats of the sconv-sconv-sconv blocks in the middle flow

    exit_flow: tpule of tupel
        must define 'sconv' and 'sconv_pool', e.g. (('sconv_pool',728,1024),('sconv',1536))
        (see summary for its meaning)

    dropout_rate: float 
        the original paper use 0.5 at the final dense layer 

    l2_weight: float 
        l2 weight for each conv layer

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
    
    * For larger datasets, e.g. ImageNet, try (obtained from original paper)
    model = XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
    entry_flow = (('conv',32,True),('conv',64,False),('sconv_pool',128,128),('sconv_pool',256,256),('sconv_pool',728,728)),
    middle_flow_repeat = 9,
    exit_flow = (('sconv_pool',728,1024),('sconv',1536),('sconv',2048)),
    dropout_rate = 0.5, l2_weight = 1e-4)
'''

class ConvBlocks:
    @staticmethod
    def BNConv(x, filters, kernel_size, strides, l2_weight = 1e-4, has_act = True):
        x = layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, 
                padding = 'same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_weight))(x)
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
                padding = 'same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_weight))(x)
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
    entry_flow = (('conv',32,True),('conv',64,False),('sconv_pool',128,128),('sconv_pool',256,256),('sconv_pool',728,728)),
    middle_flow_repeat = 9,
    exit_flow = (('sconv_pool',728,1024),('sconv',1536),('sconv',2048)),
    dropout_rate = 0.5, l2_weight = 1e-4):

    x_in = layers.Input(shape = input_shape)
    x = x_in
    # Entry flow
    for n, (block_type, attr1, attr2) in enumerate(entry_flow):
        if n == 0 and block_type != 'conv':
            raise Exception('entry flow must begin with a "conv" !')
        if block_type == 'conv':
            x = ConvBlocks.BNConv(x, attr1, (3,3), 2 if attr2 else 1, l2_weight, True)
        elif block_type == 'sconv_pool':
            x = ConvBlocks.SepConvMaxPoolBlock(x, (attr1,attr2), l2_weight, front_relu=(entry_flow[n-1][0]!='conv'))
        else: 
            raise Exception('entry flow should be built via "conv" and "sconv_pool" only')

    # middle flow
    x = ConvBlocks.SepConvBlock(x, entry_flow[-1][-1], repeats = middle_flow_repeat, l2_weight = 1e-4)

    # exit flow
    for block in exit_flow:
        if block[0] == 'sconv_pool':
            x = ConvBlocks.SepConvMaxPoolBlock(x, (block[1],block[2]), l2_weight, front_relu=True)
        elif block[0] == 'sconv':
            x = ConvBlocks.SepConv(x, block[1], 3, 1, l2_weight, 'back')
        else:
             raise Exception('exit flow should be built via "sconv" and "sconv_pool" only')
    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

if __name__ == '__main__':
    model = XceptionNet()   
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)

    



