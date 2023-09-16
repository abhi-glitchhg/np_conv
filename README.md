# np_conv

Implementations of convolutions in numpy without for loops.

# Why this project? 


![image](https://github.com/abhi-glitchhg/np_conv/assets/72816663/6f60d5a3-cbdb-47a9-9bf3-e4f74e2741e9)


# Warning 

Here, we are using [ `numpy.lib.stride_tricks.as_strided` ](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html) function to get the desired views of the numpy array. The documentation of the above function warns to use this function with care. 
> Warning

>This function has to be used with extreme care, see notes. 
as_strided creates a view into the array given the exact strides and shape. This means it manipulates the internal data structure of ndarray and, if done incorrectly, the array elements can point to invalid memory and can corrupt results or crash your program. It is advisable to always use the original x.strides when calculating new strides to avoid reliance on a contiguous memory layout.

>Furthermore, arrays created with this function often contain self overlapping memory, so that two elements are identical. Vectorized write operations on such arrays will typically be unpredictable. They may even give different results for small, large, or transposed arrays.

>Since writing to these arrays has to be tested and done with great care, you may want to use writeable=False to avoid accidental write operations.

> For these reasons it is advisable to avoid `as_strided` when possible.

Though, I have checked with different permutations and combinations, please first check the outputs yourself for your use case. 


# Material I found useful while implementing this:

1) CMU's Deep Learning Systems Course: [website](https://dlsyscourse.org/), [github](https://github.com/dlsyscourse/public_notebooks) 
