print("hello")
import numpy as np
import torch
from conv3d import conv3d_ncdhw, conv3d_ndhwc
from functools import partial
from numpy.testing import assert_allclose as assert_np

assert_np = partial(assert_np, rtol=1e-05)

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
        for height in [32,64,127]:
            for width in [32,64]:
                for depth in [16, 20]:
                    for kernel1 in [2,1]:
                        for kernel2 in [3,7,5]:
                            for kernel3 in [3,7,5,2]:
                                for ic in [1,3,7]:
                                    for oc in [4,7,3]:
                                        print(f"{it} th iteration:  {depth} {height} {width} {kernel1} {kernel2} {ic} {oc}")

                                        
                                
                                        Z1 = np.random.randn(5,depth,height,width,ic)
                                        W1 = np.random.randn(kernel1,kernel2, kernel3,ic, oc)
                                    
                                        Z_p_1 = torch.tensor(Z1).permute(0,4,1,2,3) #ncdhw
                                        W_p_1 = torch.tensor(W1).permute(4,3,0,1,2) #oi
                                
                                        Z2 = np.random.randn(5,ic,depth, height,width)
                                        W2 = np.random.randn(oc,ic,kernel1,kernel2, kernel3)

                                        Z_p_2 = torch.tensor(Z2)
                                        W_p_2 = torch.tensor(W2)


                                        S =(1,1,1)
                                        D=(1,1,1)
                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())

                                        assert_np( conv3d_ncdhw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        
                                        
                                        S =(1,1,2)
                                        D=(2,2,1)
                                        
                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())

                                        assert_np( conv3d_ncdhw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )

                                        S =(1,1,4)
                                        D=(3,3,2)

                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())

                                        assert_np( conv3d_ncdhw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        

                                        S =(1,1,2)
                                        D=(5,5,1)

                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())
                                        
                                        assert_np( conv3d_ncdhw(Z2,W2,S,D), pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        

                                        S =(3,2,3)
                                        D=(2,1,3)
                                        

                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())
                                        assert_np( conv3d_ncdhw(Z2,W2,S,D),pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() ) 

                                        S =(3,6,2)
                                        D=(2,1,5)

                                        assert_np( conv3d_ndhwc(Z1,W1,S,D), pytorch_reference_conv3d(Z_p_1,W_p_1,S,D).permute(0,2,3,4,1).contiguous().numpy())
                                        
                                        assert_np(conv3d_ncdhw(Z2,W2,S,D),pytorch_reference_conv3d(Z_p_2,W_p_2,S,D).contiguous().numpy() )
                                        it+=1


if __name__ == "__main__":
  test_pytorch()
