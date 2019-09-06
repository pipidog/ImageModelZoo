## Summary 

    Build a MobileNetV2 from scratch, reference https://arxiv.org/abs/1704.04861

## Arguments

**input_shape:** tuple   
        hape of input image; e.g.: (32,32,3) for cifar10

**n_classes:**  integer     
        number of classes in dataset

**first_block_filters**: integer   
        In MobileNet V1, there is initial Conv2D layer before entering the main body.
        The original paper recommend 32.
    
**alpha** float, 0.0~1.0   
        MobileNet use a parameter alpha to control the model size. The number of channel
        on all Conv2D will multiple this parameter so the model size will be reduced if
        alpha < 1.0.  
    
**n_downsampling** integer, 0~6   
        The original MobileNet contains 6 Strides =2 Conv2D layers. So the spatial resolutions
        will be 64 times smaller on its row and columns. However, for low resolution dataset
        such as cifar10, you don't want the downsampling too aggressive. If you set 
        this value to, say 3, only the last 3 downsampling will be performed. All the others
        will use stride = 1 instead.

## Returns
**Model**    
        A Keras model instance

## Suggested Models
    ```python
    # for cifar10, cifar100, try:
    MobileNetV1(input_shape = (32,32,3), n_classes = 10, first_block_filters = 32, 
                    alpha = 0.25, n_downsampling = 3,  l2_weight =1e-4)
    # for Tiny ImageNet, try:
    MobileNetV1(input_shape = (64,64,3), n_classes = 200, first_block_filters = 32, 
                    alpha = 0.5, n_downsampling = 4,  l2_weight =1e-4)
    # for ImageNet, try:
    MobileNetV1(input_shape = (224,224,3), n_classes = 1000, first_block_filters = 32, 
                    alpha = 1.0, n_downsampling = 6,  l2_weight =1e-4)
    ```