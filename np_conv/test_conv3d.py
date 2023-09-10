print("hello")
import numpy as np
import torch
from conv3d import conv3d_nchw
from functools import partial
from numpy.testing import assert_allclose as assert_np

assert_np = partial(assert_np, rtol=1e-07)

def pytorch_reference_conv3d(img, weight,S =(1,1,1), D=(1,1,1)):
    """
    img: ndarray with nchw shape
    weight:iokk shape
    """
    try:
        import torch
    except:
        raise("To run this pytorch is required. Please visit pytorch.org to for installation related information.")
    return torch.nn.functional.conv3d(img, weight,stride=S, dilation=D)



def test_pytorch():
    """
    basic test suit, once we have enough functions; we will make a test folder and handle all test relted matters there only 
    """
    it = 0
    import torch

    for i in range(1):
        for height in [32,64,128,224,384]:
            for width in [32,64,128,224,384]:
                for depth in [32,64,128,224,384]:
                    for kernel1 in [3,7,5,1]:
                        for kernel2 in [3,7,5,2]:
                            for kernel3 in [3,7,5,2]:
                                for ic in [1,3,7]:
                                    for oc in [4,7,3]:
                                        print(f"{it} th iteration:   {height} {width} {kernel1} {kernel2} {ic} {oc}")

                                        
                        
                                        Z2 = np.random.randn(5,ic,depth, height,width)
                                        W2 = np.random.randn(oc,ic,kernel1,kernel2, kernel1)

                                        Z_p_2 = torch.tensor(Z2)
                                        W_p_2 = torch.tensor(W2)


                                        S =(1,1,1)
                                        D=(1,1,1)


                                        assert_np( conv3d_nchw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        S =(1,1,2)
                                        D=(2,2,1)
                                        
                                        assert_np( conv3d_nchw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )

                                        S =(1,1,4)
                                        D=(3,3,2)

                                        assert_np( conv3d_nchw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        

                                        S =(1,1,2)
                                        D=(5,5,1)
                                        
                                        assert_np( conv3d_nchw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        

                                        S =(3,2,3)
                                        D=(2,1,3)
                                        
                                        assert_np( conv3d_nchw(Z2,W2,S,D),pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() ) 

                                        S =(3,6,2)
                                        D=(2,1,5)
                                        
                                        assert_np(conv3d_nchw(Z2,W2,S,D),pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        it+=1


if __name__ == "__main__":
  test_pytorch()
