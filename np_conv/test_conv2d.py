# check all the implementations by comparing them to mainstream libraries. 
#Also need to do time profiling and memory connsumption against normal convolutions

import numpy as np
import torch
from conv2d import conv2d_nchw, conv2d_nhwc
from functools import partial
from numpy.testing import assert_allclose as assert_np

assert_np = partial(assert_np, rtol=1e-05)

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
    i = 0
    import torch
    for i in range(5):
        for height in [224,384,1024,822]:
            for width in [224,384,1024,822]:
                for kernel1 in [3,7,14,22]:
                    for kernel2 in [3,7,14,22]:
                        for ic in [1,3,7]:
                            for oc in [4,7,1]:
                                #print(f'{i} th iteration', end=" \t")
                                
                                Z1 = np.random.randn(5,height,width,ic)
                                W1 = np.random.randn(kernel1,kernel2, ic, oc)
                            
                                Z_p_1 = torch.tensor(Z1).permute(0,3,1,2)
                                W_p_1 = torch.tensor(W1).permute(3,2,0,1)

                                Z2 = np.random.randn(5,ic,height,width)
                                W2 = np.random.randn(oc,ic,kernel1,kernel2)

                                Z_p_2 = torch.tensor(Z2)
                                W_p_2 = torch.tensor(W2)


                                S =(1,1)
                                D=(1,1)

                                assert_np( conv2d_nhwc(Z1,W1,S,D), pytorch_reference(Z_p_1,W_p_1,S,D).permute(0,2,3,1).contiguous().numpy())
                                #assert_np( conv2d_nchw(Z2,W2,S,D), pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                #print(np.max(conv2d_nchw(Z2,W2,S,D)- pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy()))
                                #diff =  conv2d_nchw(Z2,W2,S,D) - pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy()
                                #np.save( f"{i}th_diff.npz",diff)
                                #assert_np( conv2d_nchw(Z2,W2,S,D), pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                
                                S =(1,1)
                                D=(2,2)
                                assert_np( conv2d_nhwc(Z1,W1,S,D),pytorch_reference(Z_p_1,W_p_1,S,D).permute(0,2,3,1).contiguous().numpy() )
                                #assert_np( conv2d_nchw(Z2,W2,S,D), pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() )

                                S =(3,3)
                                D=(2,2)
                                assert_np( conv2d_nhwc(Z1,W1,S,D), pytorch_reference(Z_p_1,W_p_1,S,D).permute(0,2,3,1).contiguous().numpy()) 
                                #assert_np( conv2d_nchw(Z2,W2,S,D), pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                
                                S =(3,2)
                                D=(2,1)
                                assert_np( conv2d_nhwc(Z1,W1,S,D), pytorch_reference(Z_p_1,W_p_1,S,D).permute(0,2,3,1).contiguous().numpy()) 
                                #assert_np( conv2d_nchw(Z2,W2,S,D),pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() ) 

                                S =(3,6)
                                D=(2,1)
                                assert_np(conv2d_nhwc(Z1,W1,S,D), pytorch_reference(Z_p_1,W_p_1,S,D).permute(0,2,3,1).contiguous().numpy())
                                #assert_np(conv2d_nchw(Z2,W2,S,D),pytorch_reference(Z_p_2,W_p_2,S,D).contiguous().numpy() )

                                del Z1, W1, Z2, W2, Z_p_1, Z_p_2, W_p_1, W_p_2

                                #print(f'{i} th done')
                                i+=1
                                
if __name__ == "__main__":
    test_pytorch()