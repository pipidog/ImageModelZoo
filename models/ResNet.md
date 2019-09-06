## Summary 

    Build a ResNet from scratch, reference https://arxiv.org/abs/1512.03385

## Arguments 
**input_shape**, tuple w/ 3 elements    
        hape of input image; e.g.: (32,32,3) for cifar10

**n_classes**, integer   
        number of classes in dataset

**first_layer_kernel** integer, e.g. 3 means (3,3)   
        ResNet has a (Conv2D +Maxpooling) layers before entering the main body
        of ResNet. The number of filters of the Conv2D must consistent with the
        filter of the first Residual block, therefore, you don't need to assign
        it. It will be calculated automatically. However, you will need to specify
        the kernel size. For large dataset such as image, a 7x7 is suggested. 
        For small dataset, use 3x3. 

**first_layer_downsampling**,  True / False   
        The first Conv2D in ResNet uses a stride = 2 to reduce spatial resolution
        before entering the main body. For dataset with low resolution such as 
        cifar10, you should set it to False so stride = 1 for the first layer

**first_pooling** tuple, e.g (3,3) or None    
        pool size of the first pooling layer. for dataset with low resolution such
        as cifar10, you should use None so the first pooling layer will be skipped.

**residual_blocks** tuple of tuple   
        ResNet is constituted by repeating a residual sub-blocks to form a larger
        residual block (see Table.1 in original paper). Since the kernel size of
        each subblock is always fixed to (3x3), you only need to specify the number
        of output channels of each sub-block and number of repeats. e.g. ResNet-101
        in Table.1 of the original Table can be expressed as 
        ((256,2),(512,4),(1024,23),(2048,3)), i.e. first block is constituted by two sub-block with output channel = 256 for each and so on. 

**bottleneck**, True / False    
        If True, use bottleneck residual sub-locks as shown in ResNet-50, -101, -152
        in Table.1 of the original paper. If False, use conventional sub-blocks 
        as shown in ResNet-18, -34.

**l2_weight**, float, default = 1e-4   
        l2 penality add to all Conv layers.

## Returns
**Model**   
        A Keras model instance

## Suggested Models
    # for cifar 10 or cifar100, try (based on the original paper): 
    # Template: (for residual_block see below)

    # ResNet-20: (error ~ 8.75%)
    residual_block = ((16,3), (32,3), (64,3)) 
    # ResNet-32: (error ~ 7.51%)
    residual_block = ((16,5), (32,5), (64,5)) 
    # ResNet-44: (error ~ 7.17%)
    residual_block = ((16,5), (32,5), (64,5)) 
    # ResNet-56: (error ~ 6.97%)
    residual_block = ((16,9), (32,9), (64,9)) 
    # ResNet-110: (error ~ 6.43%)
    residual_block = ((16,18), (32,18), (64,18)) 

    ResNet(input_shape = (32,32,3), n_classes = 10, first_layer_kernel = 3,
            first_layer_downsampling = False, first_pooling = None, 
            residual_blocks = residual_block,
            bottleneck = False, l2_weight = 1e-4) 


    # for ImageNet, try (based on original paper):

    # Template: Non-Bottleneck (for residual_block see below)
    # ResNet-18: 
    residual_block = ((64,2),(128,2),(256,2),(512,2))
    # ResNet-34: 
    residual_block = ((64,3),(128,4),(256,6),(512,4))

    ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
            first_layer_downsampling = True, first_pooling = (3,3), 
            residual_blocks = residual_block,
            bottleneck = False, l2_weight = 1e-4) 
    
    # Template: Bottleneck (for residual_block see below)
    #ResNet-50: 
    residual_block = ((256,3),(512,4),(1024,6),(2048,4))
    #ResNet-101: 
    residual_block = ((256,3),(512,4),(1024,23),(2048,3))
    #ResNet-152: 
    residual_block = ((256,3),(512,8),(1024,36),(2048,3))

    ResNet(input_shape = (224,224,3), n_classes = 1000, first_layer_kernel = 7,
            first_layer_downsampling = True, first_pooling = (3,3), 
            residual_blocks = residual_block,
            bottleneck = True, l2_weight = 1e-4) 
    ```
