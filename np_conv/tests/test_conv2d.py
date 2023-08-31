# check all the implementations by comparing them to mainstream libraries. 
#Also need to do time profiling and memory connsumption against normal convolutions

import numpy as np
import torch
#from ..conv2d import conv2d_nchw, conv2d_nhwc


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
from numpy.lib.stride_tricks import as_strided


def conv2d_nchw(img, weight, S=(1,1), D=(1,1)):
  """
  img: nchw ndarray

  weight: oikk shaped ndarray 

  """


  N,CI, HI, WI = img.shape
  Ns,Cs, Hs, Ws = img.strides
  K = weight.shape[-2:]

  CO  = weight.shape[0]

  HO = (HI -1 -D[0]*(K[0]-1))//S[0] +1
  WO = (WI -1 -D[1]*(K[1]-1))//S[1] +1

  temp = as_strided(img,
                    
                    shape = (N, HO, WO, CI, K[0], K[1]),

                    strides = (Ns, Hs*S[1], Ws+(S[0]-1)*img.itemsize*CI, Cs, Hs*D[0], Ws+(D[1]-1)*img.itemsize)
                    )
  
  
  print(temp.shape)
  temp =temp.reshape(-1, K[0]*K[1]*CI)
  print(temp.shape)
  print(weight.shape)
  weight = weight.reshape(CO, -1) 
  print(weight.shape)

  out = weight @ temp.T
  return out.reshape(N,CO, HO,WO)



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

        S =(1,1)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))

        S =(3,3)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))

        S =(3,2)
        D=(2,2)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D))

        S =(3,6)
        D=(2,1)
        print(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy().shape)
        print(conv2d_nhwc(Z,W,S,D).shape)
        assert np.linalg.norm(pytorch_reference(Z_p,W_p,S,D).permute(0,2,3,1).contiguous().numpy() == conv2d_nhwc(Z,W,S,D)), 'dont match :( '

if __name__ == "__main__":
    #test_pytorch()

    Z = np.random.randn(1,3,6,6)
    W = np.random.randn(1,3,3,3)
    np_out = conv2d_nchw(Z,W)
    
    Z_p = torch.tensor(Z)
    W_p = torch.tensor(W)

    out = pytorch_reference(Z_p, W_p)
    print(out, '\n',np_out,'\n', np_out.dtype)
    print(np.linalg.norm(pytorch_reference(Z_p,W_p).contiguous().numpy() == conv2d_nchw(Z,W)), 'dont match :( '
    )