## Summary

Build a Xception Net from scratch, reference https://arxiv.org/abs/1610.02357

The original Xception is only designed for ImageNet. There is no any suggested
structure for other datasets such as cifar10, cifar100, Tiny ImageNet, etc, which 
comes with much lower resolutions. 

Xception Net has the following structure: (see Fig.5 in original paper)       
**Entry Flow**      
Two Conv (first one with stride = 2) and a series of sconv-sconv-maxpooling 
blocks for downsampling

**Middle Flow**     
Repeat sconv-sconv-sconv blocks where the filters size is identical to the 
last layer in Entry flow.

**Exit Flow**       
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
    
## Arguments }
**input_shape**, a tuple w/ 3 elements      
shape of input image; e.g.: (32,32,3) for cifar10

**n_classes**, an integer       
number of classes in dataset

**entry_flow**, tuple of tuple      
must defined by 'conv' and 'sconv' e.g.: (('conv',32,True),('sconv_pool',128,128))
(see summary for its meaning). The first layer must be a 'conv'

**middle_flow_repeat**, int     
number of repeats of the sconv-sconv-sconv blocks in the middle flow

**exit_flow**, tuple of tuple       
must define 'sconv' and 'sconv_pool', e.g. (('sconv_pool',728,1024),('sconv',1536))
(see summary for its meaning)

**dropout_rate**, float         
the original paper use 0.5 at the final dense layer 

**l2_weight**, float         
l2 weight for each conv layer

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

# For larger datasets, e.g. ImageNet, use the parameters suggested by the original paper)
model = XceptionNet(input_shape = (299,299,3), n_classes = 1000, 
        first_two_conv = (32, 64, True), maxpool_block = (128, 256, 728),
        middle_flow_repeat = 8,
        exit_flow = (728, 1024, 1536, 2048), dropout_rate = 0.5, l2_weight = 1e-4)    
```