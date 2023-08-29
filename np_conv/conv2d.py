from numpy.lib.stride_tricks import as_strided
import numpy as np



def conv2d_nhwc(img, weight, S=(1,1), D= (1,1)):
  """
  implementation of convolutions with Numpy. Hopefully optimized


  img : ndarray with nhwc shape 

  """
  # TODO 
  # add checks for contiguous ndarray 
  # it might be an issue if the array is not contiguous as we are directly picking values from the memory. (this assumes array is contiguous)
  # again question arise that whether shoud we raise the issue or make the array contiguous ?
  # -> on first thought we should raise the issue to the user. Let him handle the rough edges 
  # -> in the error message add guide on how to make the array contigous. 
  # also the array might be contiguous in fortran style; ahhhhhhhhhh
  #  
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

           strides = (Ns, Hs*S[1], Ws+(S[0]-1)*img.itemsize*CI, Hs*D[0], Ws+CI*(D[1]-1)*img.itemsize, Cs ) 

           ).reshape(-1, K[0]*K[1]*CI)

  print(temp.shape)
  out = temp @ weight.reshape(-1, CO)

  return out.reshape(N, HO, WO, CO)



def conv2d_nchw(img, weight, S, D):
  N,CI, HI, WI = img.shape
  Ns,Cs, Ws, Cs = img.strides
  K = weight.shape[-2], weight.shape[-1]

  CO  = weight.shape[1]

  HO = (HI -1 -D[0]*(K[0]-1))//S[0] +1
  WO = (WI -1 -D[1]*(K[1]-1))//S[1] +1

  temp = 0
  #TODO Add some meaningful code here.
  
  out= temp@ weight.reshape(...)

  return out.reshape(N,CO, HO,WO)
