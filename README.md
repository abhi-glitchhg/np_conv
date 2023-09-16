# np_conv

Implementations of convolutions in numpy without for loops.

# Why this project? 


![image](https://github.com/abhi-glitchhg/np_conv/assets/72816663/6f60d5a3-cbdb-47a9-9bf3-e4f74e2741e9)


# Warning 

Here, we are using [ `numpy.lib.stride_tricks.as_strided` ](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html) function to get the desired views of the tensors. The documentation of the above function warns to use this function with care. 
> Warning
>This function has to be used with extreme care, see notes. 

and 

> For these reasons it is advisable to avoid `as_strided` when possible.

Though, I have checked with different permutations and combinations, please first check the outputs yourself for your use case. 


# Material I found useful while implementing this:

1) CMU's Deep Learning Systems Course: [website](https://dlsyscourse.org/), [github](https://github.com/dlsyscourse/public_notebooks) 
