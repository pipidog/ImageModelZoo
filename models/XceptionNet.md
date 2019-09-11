## Summary

Build a Xception Net from scratch, reference https://arxiv.org/abs/1610.02357

Xception Net has the following structure:
* part 1: two conv, the first conv performs a stride for downsampling (optional)
* part 2: several sconv-sconv-maxpooling block for downsamping
* part 3: repeat xception blocks for several times 
        (number filter is fixed to the last layer in maxpool blocks )
* part 4: another sconv-sconv-maxpooling block for downsamping
* part 5: two sconv-sconv
* part 6: global average pooling-dense-softmax

Therefore, to specify a Xception Net, you need to input their corresponding parameters

* part 1: input the number of filter of the first two conv and whether to perform downsampling
* part 2: number of sconv-sconv-maxpooling blocks (at least 1 block) and their corresponding filters
* part 3: the number of repeats
* part 4 & part 5: 4 number filters of each sconv layer
* part 6: no paramaters needed

So it does not need complicated parameters to define a Xception Net.
    
## Arguments }
**input_shape**, a tuple w/ 3 elements   
hape of input image; e.g.: (32,32,3) for cifar10

**n_classes**, an integer   
number of classes in dataset

**first_two_conv**, (int, int, boolen)   
number of filters of the first two conv layer and whether to perform a stride = 2
downsampling in the first layer.

**maxpool_block**, (int, int, ...)      
each maxpool-block is constituted by sconv-sconv-maxpool where the number of filters
are the same for both sconv. Therefore, you only need one parameter, i.e. the number
of filters, to specify a maxpool_block.    

e.g. (64, 128)

means there are two maxpool_block with filters = 64 and 128 respectively.

**middle_flow_repeat**, int     
the middle flow part is a series of xception blocks (sconv-sconv-sconv) with 
filter size identical to the last layer in maxpool_block. therefore, you only 
need a single parameter to define this part, e.g. 8, means respeat 8 times 
of xception blocks.

**exit_flow_filters**, (int, int, int, int)     
the exit flow of Xception is sconv-sconv-maxpool-sconv-sconv. Therefore, you 
need 4 filter numbers for each sconv.

**dropout_rate**, float or None     
the original paper use a dropout layer right before final regression layer. 
the suggested dropout_rate is 0.5 for ImageNet. Use None if you don't want it.

**l2_weight**, float, default = 1e-4        
l2 penality add to all Conv or SConv layers.

## Returns
**Model**        
A Keras model instance

## Suggested Models
```python
'''
The original paper of Xception doesn't offer any suggestion on model parameters
other than ImageNet. Therefore, I suggest the following parameters which will give
you a model that roughly the same number of parameters as ResNet. 

Given the fact that Xception outperforms ResNet on ImageNet slightly when model 
complexity is similar, hopefully you can still get similar results when using 
the following parameters.
'''
# For cifar10, cifar100 
# the following parameters will give you roughly the same number of parameters as ResNet 20, 0.27M v.s. 0.30M
model = XceptionNet(input_shape = (32,32,3), n_classes = 10, 
        first_two_conv = (32, 64, False), maxpool_block = (64, 64),
        middle_flow_repeat = 8,
        exit_flow = (128, 128, 256, 256), dropout_rate = 0.5, l2_weight = 1e-4)


# For cifar10, cifar100
# the following parameters will give you roughly the same number of parameters as ResNet 32, 0.46M v.s 0.48M    
model = XceptionNet(input_shape = (32,32,3), n_classes = 10, 
        first_two_conv = (32, 64, False), maxpool_block = (64, 128),
        middle_flow_repeat = 6,
        exit_flow = (128, 128, 128, 128), dropout_rate = 0.5, l2_weight = 1e-4)


# For larger datasets, e.g. ImageNet, use the parameters suggested by the original paper)
model = XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
        first_two_conv = (32, 64, True), maxpool_block = (128, 256, 728),
        middle_flow_repeat = 8,
        exit_flow = (728, 1024, 1536, 2048), dropout_rate = 0.5, l2_weight = 1e-4)    
```