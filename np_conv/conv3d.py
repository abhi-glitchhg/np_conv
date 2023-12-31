from numpy.lib.stride_tricks import as_strided
import numpy as np


def conv3d_ncdhw(img, weight, stride=(1,1,1), dilation=(1,1,1)):
  """
  img: ncdhw ndarray
  weight: oikkk shaped ndarray 
  """
  assert img.flags['C_CONTIGUOUS'], "array should be c contiguous array, "
  S, D = stride, dilation

  N,CI,DI, HI, WI = img.shape
  Ns,Cs,DS, Hs, Ws = img.strides
  K = weight.shape[-3:]

  CO  = weight.shape[0]

  DO = (DI -1 -D[0]*(K[0]-1))//S[0] +1
  HO = (HI -1 -D[1]*(K[1]-1))//S[1] +1
  WO = (WI -1 -D[2]*(K[2]-1))//S[2] +1
  temp = as_strided(img,
                    shape = (N,DO, HO, WO, CI, K[0], K[1], K[2]),
                    strides = (Ns,DS*S[0], Hs*S[1], Ws + (S[2]-1)*img.itemsize, Cs,DS*D[0], Hs*D[1], Ws+(D[2]-1)*img.itemsize)
                    ).reshape(-1, K[0]*K[1]*K[2]*CI)
  
  out = (temp @ weight.reshape(CO, -1).T).reshape(N,DO,HO,WO,CO) # (N*DO,HO*WO, C)
  return out.transpose((0,4,1,2,3))

def conv3d_ndhwc(img, weight, stride=(1,1,1), dilation = (1,1,1)):
  """
  img : ndarray with ndhwc shape 
  weight: ndarray with KKKIO shape
  S: stride
  D: Dilation
  """
  assert img.flags['C_CONTIGUOUS'], "array should be c contiguous array, "
  S, D = stride, dilation
  N, DI, HI, WI, CI = img.shape
  Ns, Ds , Hs, Ws, Cs = img.strides
  K = weight.shape[0:3]
  CO = weight.shape[-1]
  DO, HO, WO = (DI -1 -D[0]*(K[0]-1))//S[0] +1 ,(HI -1 -D[1]*(K[1]-1))//S[1] +1 ,(WI -1 -D[2]*(K[2]-1))//S[2] +1

  temp = as_strided(img ,        
          shape = (N, DO, HO, WO, K[0], K[1],K[2], CI),
           strides = (Ns, Ds*S[0], Hs*S[1], Ws+(S[2]-1)*img.itemsize*CI,Ds*D[0], Hs*D[1], Ws+(D[2]-1)*img.itemsize*CI , Cs ) 

           ).reshape(-1, K[0]*K[1]*K[2]*CI)
  
  out = temp @ weight.reshape(-1, CO) # N,DO,HO,WO,CO

  return out.reshape(N,DO, HO, WO, CO)
  
