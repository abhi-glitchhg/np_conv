# check all the implementations by comparing them to mainstream libraries. 
#Also need to do time profiling and memory connsumption against normal convolutions



#check against pytorch

def pytorch_reference(img, weight,S =(1,1), D=(1,1)):
    """
    img: ndarray with nchw shape
    weight:iokk shape
    """
    try:
        import torch
    except:
        raise("To run this pytorch is required. Please visit pytorch.org to for installation related information.")
    
    return torch.nn.conv2d(img, weight,stride=S, dilation=D)


def keras_reference(img, weight, S=(1,1), D= (1,1)):
    try:
        import tensorflow as tf
    except:
        raise("To run this, tensorflow is required. please visit tensorflow.org for installation related information.")
    
    return tf.keras.backend.conv2d(img, weight,strides=S, dilation_rate = D)
     
