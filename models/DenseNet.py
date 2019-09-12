from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a DensetNet from scratch, reference https://arxiv.org/pdf/1608.06993.pdf

{ Arguments }

    input_shape: 
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: 
        number of classes in dataset

    first_layer: 
        DenseNet has a (Conv2D +Maxpooling) layers before entering the main body
        of DenseNet. You should specify their properties using first_layer and
        first_pooling arguments. Properties of first Conv2D layer are specified 
        using three parameters: (filter, kernel_size, if stride = 2). 
        e.g. (24, 3, True) means a convlutional layer with filter = 24, 
        kernel_size = (3,3) and stride = 2 (if False, use stride = 1). 
        
        DenseNet's original paper suggests:
        (16, 3, True) or (2*growth rate, 3, True) for ImageNet.

        However, for datasets with low spatial resolutions, e.g. cifar10, you don't 
        want the model to downsample the resolution so aggresively. If you, you should 
        set the last parameter to False, e.g. (16, 3, False) to keep its resolution. 

    first_pooling: 
        DenseNet has a (Conv2D + Maxpooling) layers before entering the main body
        of DenseNet. You should specify the pooling size here (the stride is always
        set as (2,2)). In DenseNet's original paper, it uses (3,3) for ImageNet.

        However, for datasets with low spatial resolutions, e.g. cifar10, you don't 
        want the model to downsample the resolution so aggresively. If you, you should 
        set it to None, if so the maxpooling layer will not be skipped.

    dense_layers:
        DenseNet is made by several dense block. Each dense block is consisted of several
        dense layers. You can specify it by, say (12,12,12), which means you want three
        dense blocks with 12, 12, and 12 layers respectively. For small dataset, you can 
        use DenseNet-40, which is (12,12,12). For larger dateset such as ImageNet, please
        consult Table.1 in the original paper, e.g. (6,12,24,16) refers to DenseNet-121.
        * Note: DenseNet connects each dense block using a transition block and each 
            transition block contains a maxpooling (2,2) for spatial resolution downsampling.
            Therefore, if you specify four dense blocks, there will be three downsampling
            in between (the last block connects to a classifier not a transition block).
            and the final resolution of your feature map will become 8 times smaller on 
            its rows and columns.  

    growth_rate: 
        Define the growth rate of each dense layer. This is a key component of DenseNet, 
        If you don't understand what is it, read the original paper. For small dataset
        such as cifar10, it is recommended to use 12. For larger dataset such as ImageNet, 
        you should use 32. 

    bottleneck: 
        Whether to use Bottleneck structure for each dense layer to reduce computational
        cost. In this impelmentation, we use the recommended bottleneck width = 4, i.e.
        the channels of all input feature maps is resize to  4*growth_rate using a 1x1
        convolution layer. If set to True, it is called DenseNet-B. Experiment results
        show that bottleneck can make DenseNet more efficient. Therefore, it is recommended
        to set to True.

    compression_rate:
        compression rate in the transition block. The original paper recommends 0.5.

    dropout_rate:
        dropout rate to be used right after each Conv2D. The original paper recommend 0.2
        
    l2_weight:
        l2 penality used for each Conv2D. suggested value: 1e-4

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }

    Each DenseNet can be speficied as DenseNet-BC{L, k}, where B means if bottleneck is on, C means
    if compression is on. L is the total layers (Conv2D+Dense, exclude pooling). k is growth rate.
    In the following, I show how to implement the models discussed in the orignal paper use the API.

    >> for cifar 10 or cifar100, try: 
    PS: if Tiny ImageNet (64x64x3), try (16, 3, True) for first_layer, so the problem become cifar.

    * DenseNet {L=40, k=12}
    model = DenseNet(input_shape = (32, 32, 3), n_classes = 10, first_layer = (16, 3, False), 
                first_pooling = None,  dense_layers = [12,12,12], growth_rate = 12, 
                bottleneck = False, compression_rate = 1.0, dropout_rate = 0.2, l2_weight = 1e-4)
    * DenseNet-BC{L=100, k = 12}
    model = DenseNet(input_shape = (32, 32, 3), n_classes = 10, first_layer = (16, 3, False), 
            first_pooling = None,  dense_layers = [16,16,16], growth_rate = 12, 
            bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)

    * DenseNet-BC{L=100, k = 24}
    model = DenseNet(input_shape = (32, 32, 3), n_classes = 10, first_layer = (16, 3, False), 
            first_pooling = None,  dense_layers = [16,16,16], growth_rate = 24, 
            bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)

    >> For ImageNet, try:

    * DenseNet-BC {L=121, k=32}
    model = DenseNet(input_shape = (224, 224, 3), n_classes = 1000, first_layer = (64, 7, True), 
                first_pooling = (7,7),  dense_layers = [6,12,24,16], growth_rate = 32, 
                bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)

    * DenseNet-BC {L=169, k=32}
    model = DenseNet(input_shape = (224, 224, 3), n_classes = 1000, first_layer = (64, 7, True), 
                first_pooling = (7,7),  dense_layers = [6,12,32,32], growth_rate = 32, 
                bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)

    * DenseNet-BC {L=201, k=32}
    model = DenseNet(input_shape = (224, 224, 3), n_classes = 1000, first_layer = (64, 7, True), 
                first_pooling = (7,7),  dense_layers = [6,12,48,32], growth_rate = 32, 
                bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)

    * DenseNet-BC {L=264, k=32}
    model = DenseNet(input_shape = (224, 224, 3), n_classes = 1000, first_layer = (64, 7, True), 
                first_pooling = (7,7),  dense_layers = [6,12,64,48], growth_rate = 32, 
                bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)
'''

class ConvBlocks:
    @staticmethod
    def BNConv(x_in, filters, kernel_size, l2_weight, dropout_rate):
        x = layers.BatchNormalization()(x_in)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size = kernel_size, kernel_initializer='he_uniform',
                padding = 'same', use_bias=False, kernel_regularizer=l2(l2_weight))(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)            
        return x

    @classmethod
    def DenseLayer(cls, x, n_channels, dropout_rate=None, bottleneck=False, l2_weight=1e-4):
        if bottleneck:
            bottleneck_width = 4 # recommend by the paper
            x = cls.BNConv(x, n_channels*bottleneck_width, (1,1), l2_weight, dropout_rate)
        x = cls.BNConv(x, n_channels, (3,3), l2_weight, dropout_rate)        
        return x

    @classmethod
    def TransBlock(cls, x, dropout_rate = None, compression_rate = 1.0, l2_weight = 1e-4, downsampling = True):
        x = cls.BNConv(x, int(K.int_shape(x)[-1]*compression_rate), (1,1), l2_weight ,dropout_rate)
        if downsampling:
            x = layers.AveragePooling2D((2, 2), padding = 'same')(x)
        return x        

    @classmethod
    def DenseBlock(cls, x, n_layers, growth_rate, dropout_rate = None, bottleneck = False, l2_weight = 1e-4):
        for n in range(n_layers):
            x_in = x
            x = cls.DenseLayer(x_in, growth_rate, dropout_rate, bottleneck, l2_weight)
            x = layers.Concatenate()([x_in, x])
        return x   

def DenseNet(input_shape = (224,224,4), n_classes = 1000, 
            first_layer = (16, 7, True), first_pooling = (3,3),  
            dense_layers = (6,12,24,16), growth_rate = 32, 
            bottleneck = True, compression_rate = 0.5, 
            dropout_rate = 0.2, l2_weight = 1e-4):
    # initializal layers
    x_in = layers.Input(shape = input_shape)
    first_strides = (2,2) if first_layer[-1] else (1,1)
    x = layers.Conv2D(filters = first_layer[0], kernel_size = (first_layer[1],)*2, 
                    strides = first_strides, padding = 'same', kernel_initializer='he_uniform',
                    use_bias=False, kernel_regularizer=l2(l2_weight))(x_in)
    if first_pooling is not None:
        x = layers.MaxPooling2D(pool_size=first_pooling, strides=(2,2), padding = 'same')(x) 

    # dense blocks
    for n in range(len(dense_layers)):
        x = ConvBlocks.DenseBlock(x, dense_layers[n], growth_rate, dropout_rate, bottleneck, l2_weight)
        if n != len(dense_layers)-1:
            x = ConvBlocks.TransBlock(x, dropout_rate, compression_rate, l2_weight)
        else:
            x = layers.GlobalAveragePooling2D()(x)
    # classifier
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

if __name__ == '__main__':
    # DenseNet-BC {L=100, k=12}
    model = DenseNet(input_shape = (32, 32, 3), n_classes = 10, first_layer = (16, 3, False), 
                first_pooling = None,  dense_layers = [16,16,16], growth_rate = 12, 
                bottleneck = True, compression_rate = 0.5, dropout_rate = 0.2, l2_weight = 1e-4)
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)