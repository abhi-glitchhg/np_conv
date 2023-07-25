from numpy.lib.stride_tricks import as_strided
import numpy as np



def conv2d(img, weight, S=(1,1), D= (1,1)):
  """
  implementation of convolutions with Numpy. Hopefully optimized
  """
  N, HI, WI, CI = img.shape
  Ns, Hs, Ws, Cs = img.strides
  K = weight.shape[0], weight.shape[1]

  HO = (HI -1 -D[0]*(K[0]-1))//S[0] +1
  WO = (WI -1 -D[1]*(K[1]-1))//S[1] +1

  temp = as_strided(f ,
           
          shape = (N, HO, WO, K[0], K[1], CI),

           strides = (Ns, Hs*S[1], Ws+(S[0]-1)*img.itemsize*CI, Hs*D[1], Ws+CI*(D[0]-1)*img.itemsize, Cs ) 

           ).reshape(-1, K[0]*K[1]*CI)
  
  print(temp)
