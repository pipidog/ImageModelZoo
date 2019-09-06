from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
'''
{ Summary }
    Build a VGG16, VGG19 from scratch

{ Arguments }
    input_shape:
        input shape of input image, e.g. (32,32,3) for cifar10

    n_classes:
        number of classes in your dataset

    alpha: 0.0~1.0
        this parameter was not in the original paper. I intentionally introduce this parameter
        for you to control model size conveniently. If you use 0.5, the filters of each Conv2D
        layers will be halved.

    n_downsampling: 0~5 
        In original VGG, there are 5 maxpooling layers with pool size (2,2). Therefore you 
        feature map will be 32 times smaller than your original data before entering the final
        dense layers. However, for images with smaller spatial resolutions, say cifar10, you 
        don't want downsample it too aggresively. You can control how many downsampling will 
        be performed in the model. If you set it to 3, the first 2 maxpooling will be skipped. 

    last_dense_units:
        There are two dense layers in final part of VGG. The default 4096 is designed for ImageNet
        which contains 1000 classes. For dataset with classes, e.g. cifar10 with only 10 classes, 
        you should consider use a smaller value, e.g. 128. 

{ Returns }
    Model: 
        A Keras model instance

{ Suggested Models }
    for smaller dataset, such as cifar10, cifar100, try reduced vgg16 or vgg19:
        model = vgg16(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_dense_units = 128)
        model = vgg19(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_dense_units = 128)
    for Tiny ImageNet, (64x64x4), try:
        model = vgg16(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 4, last_dense_units = 512)
        model = vgg19(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 4, last_dense_units = 512)
    for ImageNet, (224x224x4), try original VGG16 or VGG19:
        model = vgg16(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096)
        model = vgg19(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096)
'''

class ConvBlock:
    @staticmethod
    def conv(x_in, filters):
        x = layers.Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), 
            padding = 'same',  kernel_initializer='he_normal', kernel_regularizer = l2(4e-5))(x_in)
        return x
    @staticmethod
    def maxpool(x_in):
        x = layers.MaxPooling2D(pool_size=(2, 2), padding= 'same')(x_in)
        return x

def vgg16(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096):
    assert 0 <= n_downsampling <= 5
    dim_reduction = [False]*(5-n_downsampling)+[True]*(n_downsampling)
    x_in = layers.Input(shape = input_shape)
    x = ConvBlock.conv(x_in, int(64*alpha))
    x = ConvBlock.conv(x, int(64*alpha))
    if dim_reduction[0]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(128*alpha))
    x = ConvBlock.conv(x, int(128*alpha))
    if dim_reduction[1]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(256*alpha))
    x = ConvBlock.conv(x, int(256*alpha))
    x = ConvBlock.conv(x, int(256*alpha))
    if dim_reduction[2]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    if dim_reduction[3]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    if dim_reduction[4]:
        x = ConvBlock.maxpool(x)
    x = layers.Flatten()(x)
    x = layers.Dense(last_dense_units, activation = 'relu')(x)
    x = layers.Dense(last_dense_units, activation = 'relu')(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)

    model = Model(inputs = x_in, outputs = x_out)
    plot_model(model, 'model.png',show_shapes = True)
    print(model.summary())
    return model

def vgg19(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096):
    assert 0 <= n_downsampling <= 5
    dim_reduction = [False]*(5-n_downsampling)+[True]*(n_downsampling)
    x_in = layers.Input(shape = input_shape)
    x = ConvBlock.conv(x_in, int(64*alpha))
    x = ConvBlock.conv(x, int(64*alpha))
    if dim_reduction[0]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(128*alpha))
    x = ConvBlock.conv(x, int(128*alpha))
    if dim_reduction[1]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(256*alpha))
    x = ConvBlock.conv(x, int(256*alpha))
    x = ConvBlock.conv(x, int(256*alpha))
    x = ConvBlock.conv(x, int(256*alpha))
    if dim_reduction[2]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    if dim_reduction[3]:
        x = ConvBlock.maxpool(x)
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    x = ConvBlock.conv(x, int(512*alpha))
    if dim_reduction[4]:
        x = ConvBlock.maxpool(x)
    x = layers.Flatten()(x)
    x = layers.Dense(last_dense_units, activation = 'relu')(x)
    x = layers.Dense(last_dense_units, activation = 'relu')(x)
    x_out = layers.Dense(n_classes, activation = 'softmax')(x)

    model = Model(inputs = x_in, outputs = x_out)

    return model

if __name__ =='__main__':
    # test for cifar10
    model = vgg16(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_dense_units = 100)
    model.summary()
    plot_model(model, 'model.png',show_shapes = True)
    