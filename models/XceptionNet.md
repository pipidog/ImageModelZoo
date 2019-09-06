## Summary

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

## Arguments
**input_shape**, a tuple w/ 3 elements    
hape of input image; e.g.: (32,32,3) for cifar10

**n_classes**, an integer    
number of classes in dataset

**enter_flow**, tuple of tuple    
Specify the enter flow layers. Each layer is constituted by a Conv2D-BN-Relu
with kernel size (3x3). Therefore, you only need to specify its filter number
and whether to perform stride 2 downsampling. e.g.

enter_flow = ((32, False), (64, False), (128, True))

means there are three convolutions with filter 32, 64 and 128 respectively.
The last tuple (128, True) comes with a True means this layer will use stride = 
for downsampling.


**middle_flow_repeat**, integer    
This part is exactly identical to the middle flow part of original paper. The
filter size is by default identical to the last layer in enter flow. Therefore
you don't have to specify any informaiton except the number of repeats of the 
Xception blocks. 

**exit_flow**, tuple of tuple     
The exit flow is identical to the original paper. The first block performs a
maxpooling for downsampling and the next block are two SeparableConv2D. Therefore
there are four filter numbers you need to specify.

**l2_weight**, float, default = 1e-4    
l2 penality add to all Conv layers.

## Returns
**Model**    
A Keras model instance

## Suggested Models
```python
# For small datasets, e.g. cifar10, cifar100, try: 
model = ModifyXceptionNet(input_shape = (32,32,3), n_classes = 10, 
    enter_flow = ((32, False), (64, True)), middle_flow_repeat = 8,
    exit_flow = (128, 128, 256, 256), l2_weight = 1e-4)
#(it will give you roughly the same number of parameters as ResNet 20)

model = ModifyXceptionNet(input_shape = (32,32,3), n_classes = 10, 
    enter_flow = ((32, False), (64, True), (128, True)), middle_flow_repeat = 6,
    exit_flow = (128, 128, 128, 128), l2_weight = 1e-4)
#(it will give you roughly the same number of parameters as ResNet 34)

# For larger datasets, e.g. ImageNet, try
model = XceptionNet()  (the original version of Xception)
```