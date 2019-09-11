## Summary

    Build a MobileNetV2 from scratch, reference https://arxiv.org/abs/1801.04381

## Arguments

**input_shape**, tuple   
        hape of input image; e.g.: (32,32,3) for cifar10

**n_classes**, integer   
        number of classes in dataset

**first_block_filters**, integer   
        In MobileNet V2, there is initial Conv2D layer before entering the main body.
        The original paper recommend 32.

**last_block_filters**, integer   
        In MobileNet V2, there is a 1x1 Conv2D before entering the final classifier 
        layer (you can consider it as the units of a Dense layer). The original paper
        use 1280 for ImageNet. For dataset with less classes, you should use a small 
        value, such as 128 for cifar10.  
    
**alpha**, float, 0.0~1.0   
        MobileNet use a parameter alpha to control the model size. The number of channels
        on all Conv2D will multiple this parameter so the model size will be reduced if
        alpha < 1.0.  
    
**n_downsampling**, integer 0~5   
        The original MobileNet contains 5 Strides =2 Conv2D layers. So the spatial resolutions
        will be reduced 32 times on its row and columns. However, for low resolution dataset
        such as cifar10, you don't want the downsampling too aggressive. If you set 
        this value to, say 3, only the last 3 downsampling will be performed. All the others
        will use stride = 1 instead.


## Returns
**Model**    
        A Keras model instance

## Suggested Models
```python
# for cifar10, cifar100, try:
model = MobileNetV2(input_shape = (32,32,3), n_classes = 10, alpha = 0.5, n_downsampling = 3, last_block_filters = 128)
# for Tiny ImageNet, try:
model = MobileNetV2(input_shape = (64,64,3), n_classes = 200, alpha = 0.5, n_downsampling = 4, last_block_filters = 256)
# for ImageNet, try:
model = MobileNetV2(input_shape = (224,224,3), n_classes = 1000, alpha = 1.0, n_downsampling = 5, last_block_filters = 1280)
```