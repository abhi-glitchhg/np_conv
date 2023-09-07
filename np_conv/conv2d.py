from numpy.lib.stride_tricks import as_strided
import numpy as np



def conv2d_nhwc(img, weight, S=(1,1), D= (1,1)):
  """
  implementation of convolutions with Numpy. Hopefully optimized


  img : ndarray with nhwc shape 

  weight: ndarray with KKIO shape

  """
  assert img.flags['C_CONTIGUOUS'], "array should be c contiguous array, "

  try: 
    assert len(img.shape) == 4
  except:
    assert len(img.shape) == 3, "expected array is of improper dimensions, fix it :) "
    img = np.expand_dims(img, axis=0)

  N, HI, WI, CI = img.shape
  Ns, Hs, Ws, Cs = img.strides
  K = weight.shape[0], weight.shape[1]
  CO = weight.shape[-1]

  HO = (HI -1 -D[0]*(K[0]-1))//S[0] +1
  WO = (WI -1 -D[1]*(K[1]-1))//S[1] +1
  
  temp = as_strided(img ,
           
          shape = (N, HO, WO, K[0], K[1], CI),

           strides = (Ns, Hs*S[0], Ws+(S[1]-1)*img.itemsize*CI, Hs*D[0], Ws+CI*(D[1]-1)*img.itemsize, Cs ) 

           ).reshape(-1, K[0]*K[1]*CI)
  
  out = temp @ weight.reshape(-1, CO)

  return out.reshape(N, HO, WO, CO)


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

                    strides = (Ns, Hs*S[0], Ws+(S[1]-1)*img.itemsize*CI, Cs, Hs*D[0], Ws+(D[1]-1)*img.itemsize)
                    ).reshape(-1, K[0]*K[1]*CI)

  
  out = temp @ weight.transpose((1,2,3,0)).reshape(-1,CO)
  #out =  weight.reshape(CO, -1) @ temp.T
  return out.reshape(N,HO,WO, CO).transpose((0,3,1,2))