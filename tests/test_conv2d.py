# check all the implementations by comparing them to mainstream libraries. 
#Also need to do time profiling and memory connsumption against normal convolutions

import numpy as np
from np_conv import conv2d_nchw, conv2d_nhwc


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



def test_pytorch():
    """
    basic test suit, once we have enough functions; we will make a test folder and handle all test relted matters there only 
    """
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

def test_raise_error():
    print("hi")
    raise NotImplementedError
