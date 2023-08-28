# check all the implementations by comparing them to mainstream libraries. 
#Also need to do time profiling and memory connsumption against normal convolutions

import numpy as np
from conv2d import conv2d_nhwc
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
    
    return torch.nn.functional.conv2d(img, weight,stride=S, dilation=D)


def keras_reference(img, weight, S=(1,1), D= (1,1)):
    try:
        import tensorflow as tf
    except:
        raise("To run this, tensorflow is required. please visit tensorflow.org for installation related information.")
    
    return tf.keras.backend.conv2d(img, weight,strides=S, dilation_rate = D)
     
def basic_test():
    """
    basic test suit, once we have enough functions; we will make a test folder and handle all test relted matters there only 
    """
    print("hi")
    import torch
    for i in range(20):
        Z = np.random.randn(5,224,224,3)

        W = np.random.randn(4,4,3,6)
    
    
        Z_p = torch.tensor(Z).permute(0,3,1,2)
        W_p = torch.tensor(W).permute(3,2,0,1)
        S =(1,1)
        D=(1,1)

        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))
        print('we done 1')

        S =(1,1)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))
        print('we done 2')
        S =(3,3)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))
        print('we done 3')
        S =(3,2)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))
        print('we done 4')
        S =(3,6)
        D=(2,1)
        print(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy().shape)
        print(conv2d_nhwc(Z,W,S,D).shape)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D)), 'dont match :( '
        print('we done 5')

        print("No errors found :) ")


if __name__ == "__main__":
    basic_test()