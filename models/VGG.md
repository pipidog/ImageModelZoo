
## Summary
    Build a VGG16, VGG19 from scratch

## Arguments
**input_shape**   tuple   
        input shape of input image, e.g. (32,32,3) for cifar10

**n_classes** integer   
        number of classes in your dataset

**alpha** float, 0.0~1.0   
        this parameter was not in the original paper. I intentionally introduce this parameter
        for you to control model size conveniently. If you use 0.5, the filters of each Conv2D
        layers will be halved.

**n_downsampling** integer, 0~5    
        In original VGG, there are 5 maxpooling layers with pool size (2,2). Therefore you 
        feature map will be 32 times smaller than your original data before entering the final
        dense layers. However, for images with smaller spatial resolutions, say cifar10, you 
        don't want downsample it too aggresively. You can control how many downsampling will 
        be performed in the model. If you set it to 3, the first 2 maxpooling will be skipped. 

**last_dense_units** integer   
        There are two dense layers in final part of VGG. The default 4096 is designed for ImageNet
        which contains 1000 classes. For dataset with classes, e.g. cifar10 with only 10 classes, 
        you should consider use a smaller value, e.g. 128. 

## Returns
**Model**    
        A Keras model instance

## Suggested Models
```python
# for smaller dataset, such as cifar10, cifar100, try reduced vgg16 or vgg19:
model = vgg16(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_dense_units = 128)
model = vgg19(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_dense_units = 128)
# for Tiny ImageNet, (64x64x4), try:
model = vgg16(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 4, last_dense_units = 512)
model = vgg19(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 4, last_dense_units = 512)
# for ImageNet, (224x224x4), try original VGG16 or VGG19:
model = vgg16(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096)
model = vgg19(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_dense_units = 4096)
```