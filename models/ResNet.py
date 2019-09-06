from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

'''
{ Summary }

    Build a ResNet from scratch, reference https://arxiv.org/abs/1512.03385

{ Arguments }
    input_shape: a tuple w/ 3 elements
        hape of input image; e.g.: (32,32,3) for cifar10

    n_classes: an integer
        number of classes in dataset

    first_layer_kernel: an integer, e.g. 3 means (3,3)
        ResNet has a (Conv2D +Maxpooling) layers before entering the main body
        of ResNet. The number of filters of the Conv2D must consistent with the
        filter of the first Residual block, therefore, you don't need to assign
        it. It will be calculated automatically. However, you will need to specify
        the kernel size. For large dataset such as image, a 7x7 is suggested. 
        For small dataset, use 3x3. 

    first_layer_downsampling: True / False
        The first Conv2D in ResNet uses a stride = 2 to reduce spatial resolution
        before entering the main body. For dataset with low resolution such as 
        cifar10, you should set it to False so stride = 1 for the first layer

    first_pooling: a tuple, e.g (3,3) or None 
        pool size of the first pooling layer. for dataset with low resolution such
        as cifar10, you should use None so that the spatial feature maps will not 
        be reduced.

    residual_blocks:
        ResNet is constituted by repeating a residual sub-blocks to form a larger
        residual block (see Table.1 in original paper). Since the kernel size of
        each subblock is always fixed to (3x3), you only need to specify the number
        of output channels of each sub-block and number of repeats. e.g. ResNet-101
        in Table.1 of the original Table can be expressed as  
        ((256,2),(512,4),(1024,23),(2048,3)) with bottlenect set to True.
        * Note: there is a downsampling among two residual blocks, therefore, the
        above setting will reduce spatial resolution 2^8 = 8 times smaller. 

    bottleneck: True / False 
        If True, use bottleneck residual sub-locks as shown in ResNet-50, -101, -152
        in Table.1 of the original paper. If False, use conventional sub-blocks 
        as shown in ResNet-18, -34.

    l2_weight: float, default = 1e-4
        l2 penality add to all Conv layers.

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    >> for cifar 10 or cifar100, try (based on the original paper): 
    * Template: (for residual_block see below)
    ResNet(input_shape = (32,32,3), n_classes = 10, first_layer_kernel = 3,
            first_layer_downsampling = False, first_pooling = None, 
            residual_blocks = residual_block,
            bottleneck = False, l2_weight = 1e-4) 
    , where 
    ResNet-20: residual_block = ((16,3), (32,3), (64,3)) (error ~ 8.75%)
    ResNet-32: residual_block = ((16,5), (32,5), (64,5)) (error ~ 7.51%)
    ResNet-44: residual_block = ((16,7), (32,7), (64,7)) (error ~ 7.17%)
    ResNet-56: residual_block = ((16,9), (32,9), (64,9)) (error ~ 6.97%)
    ResNet-110: residual_block = ((16,18), (32,18), (64,18)) (error ~ 6.43%)

    >> for ImageNet, try (based on original paper):
    * Template: Non-Bottleneck (for residual_block see below)
    ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
            first_layer_downsampling = True, first_pooling = (3,3), 
            residual_blocks = residual_block,
            bottleneck = False, l2_weight = 1e-4) 
    , where
    ResNet-18: residual_block = ((64,2),(128,2),(256,2),(512,2))
    ResNet-34: residual_block = ((64,3),(128,4),(256,6),(512,4))

    * Template: Bottleneck (for residual_block see below)
    ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
            first_layer_downsampling = True, first_pooling = (3,3), 
            residual_blocks = residual_block,
            bottleneck = True, l2_weight = 1e-4) 
    , where
    ResNet-50: residual_block = ((256,3),(512,4),(1024,6),(2048,4))
    ResNet-101: residual_block = ((256,3),(512,4),(1024,23),(2048,3))
    ResNet-152: residual_block = ((256,3),(512,8),(1024,36),(2048,3))
'''

class ConvBlocks:
    @staticmethod
    def BNConv(x_in, filters, kernel_size, strides, l2_weight = 1e-4, has_act = True):
        x = layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, 
                padding = 'same', kernel_regularizer=l2(l2_weight))(x_in)
        x = layers.BatchNormalization()(x)
        if has_act:
            x = layers.ReLU()(x)          
        return x

    @classmethod
    def SimpleLayer(cls, x_in, n_channels, downsampling = False, l2_weight = 1e-4):
        if downsampling:
            x = cls.BNConv(x_in, n_channels, (3,3), strides = (2,2), l2_weight = l2_weight)
            x = cls.BNConv(x, n_channels, (3,3), strides = (1,1), l2_weight = l2_weight, has_act = False)
            x_tmp = layers.Conv2D(n_channels, kernel_size = (1,1), strides = (2,2), 
                padding = 'same', kernel_regularizer=l2(l2_weight))(x_in)
            x = layers.Add()([x, x_tmp])
            x = layers.ReLU()(x)
        else:
            x = cls.BNConv(x_in, n_channels, (3,3), strides = (1,1), l2_weight = l2_weight)
            x = cls.BNConv(x, n_channels, (3,3), strides = (1,1), l2_weight = l2_weight, has_act= False)
            x = layers.Add()([x, x_in])
            x = layers.ReLU()(x)
        return x
    
    @classmethod
    def BottleneckLayer(cls, x_in, n_channels, downsampling = False, l2_weight = 1e-4):
        if downsampling: # ResNet use a quarter of output channel as bottleneck
            x = cls.BNConv(x_in, int(n_channels/4), (1,1), strides = (1,1), l2_weight = l2_weight)
            x = cls.BNConv(x, int(n_channels/4), (3,3), strides = (2,2), l2_weight = l2_weight)
            x = cls.BNConv(x, n_channels, (1,1), strides = (1,1), l2_weight = l2_weight, has_act = False)
            x_tmp = layers.Conv2D(n_channels, kernel_size = (1,1), strides = (2,2), 
                padding = 'same', kernel_regularizer=l2(l2_weight))(x_in)
            x = layers.Add()([x, x_tmp])
            x = layers.ReLU()(x)
        else:
            x = cls.BNConv(x_in, int(n_channels/4), (1,1), strides = (1,1), l2_weight = l2_weight)
            x = cls.BNConv(x, int(n_channels/4), (3,3), strides = (1,1), l2_weight = l2_weight)
            x = cls.BNConv(x, n_channels, (1,1), strides = (1,1), l2_weight = l2_weight, has_act=False)
            x = layers.Add()([x, x_in])
            x = layers.ReLU()(x)
        return x    

    @classmethod
    def ResidualBlock(cls, x, n_channels, repeats, bottleneck = False, downsampling = True, l2_weights = 1e-4):
        # downsampling only happens in the beginning of each block
        if bottleneck:
            for r in range(repeats):
                if r == 0:
                    x = cls.BottleneckLayer(x, n_channels, downsampling = downsampling, l2_weight = l2_weights)
                else:
                    x = cls.BottleneckLayer(x, n_channels, downsampling = False, l2_weight = l2_weights)
        else:
            for r in range(repeats):
                if r == 0:
                    x = cls.SimpleLayer(x, n_channels, downsampling = downsampling, l2_weight = l2_weights)
                else:
                    x = cls.SimpleLayer(x, n_channels, downsampling = False, l2_weight = l2_weights)
        return x             

def ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
            first_layer_downsampling = True, first_pooling = (3,3), 
            residual_blocks = ((256,3),(512,4),(1024,6),(2048,3)),
            bottleneck = True, l2_weight = 1e-4):

    # initial Conv2D
    first_filter = residual_blocks[0][0] 
    first_stride = 2 if first_layer_downsampling else 1

    x_in = layers.Input(shape = input_shape)
    x = ConvBlocks.BNConv(x_in, first_filter, first_layer_kernel, first_stride, l2_weight)
    if first_pooling is not None:
        x = layers.MaxPool2D(first_pooling, strides = (2,2), padding = 'same')(x)
    
    # Residual blocks
    for n, (n_channels, repeats) in enumerate(residual_blocks):
        x = ConvBlocks.ResidualBlock(x, n_channels, repeats, bottleneck, n!=0, l2_weight)

    # classifier
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)
    model = Model(inputs = x_in, outputs = x_out)
    return model

if __name__ == '__main__':
    # ResNet-32 for cifar10 in page.7 of original paper
    model = ResNet(input_shape=(32,32,3), n_classes = 10, first_layer_kernel = 3,
                first_layer_downsampling = True, first_pooling = None, 
                residual_blocks=((16,5), (32,5), (64,5)), bottleneck=False)
    model.summary()
    plot_model(model, 'model.png', show_shapes = True)
    
    # ResNet-152 for ImageNet in Table.1 of original paper
    # model = ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
    #         first_layer_downsampling = True, first_pooling = (3,3), 
    #         residual_blocks = ((256,3),(512,8),(1024,36),(2048,3)),
    #         bottleneck = True, l2_weight = 1e-4) 
    # model.summary()
    # plot_model(model, 'model.png', show_shapes = True)