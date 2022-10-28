import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            ##kernel axis
            addition = 0.0
            for i in range(Hk):
                for j in range(Wk):
                    if (m + 1 - i) < 0 or (n + 1 - j) < 0 or (
                            m + 1 - i) >= Hi or (n + 1 - j) >= Wi:
                        addition += 0
                    else:
                        addition += kernel[i][j] * image[m + 1 - i][n + 1 - j]
            out[m][n] = addition
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, [(pad_height, ), (pad_width, )], mode='constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernelflip = np.flipud(np.fliplr(kernel))  #easy way to flip
    imagePadded = zero_pad(image, Hk // 2, Wk // 2)
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            out[m][n] = np.sum(imagePadded[m:m + Hk, n:n + Wk] *
                               kernelflip)  #applying convolution
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    h = np.delete(kernel,0,axis=0)
    x = Hk//2
    y = Wk//2
    out = np.copy(image)
    f = zero_pad(image, x, y)
    ### YOUR CODE HERE
    for i in range(x, Hi + x):
        for j in range(y, Wi + y):
            sum1 = f[i-x:i+x-1,j-y:j+y+1]
            out[i-x,j-y] = (sum1 * h).sum()
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    kernel = np.flipud(np.fliplr(g))
    out = conv_fast(f, kernel)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pic_mean = np.mean(g, axis=(0, 1))
    gZeroMean = g - pic_mean

    out = cross_correlation(f, gZeroMean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    #padding image
    imagePadded = zero_pad(f, Hk // 2, Wk // 2)
    #buffering output
    out = np.zeros((Hi, Wi))
    #calculating template normalize values
    pic_mean = np.mean(g)
    pict_std = np.std(g)
    gout = (g - pic_mean) / pict_std
    #cross correlation
    for i in range(Hi):
        for n in range(Wi):
            #normalizing subimage of f
            subimage = imagePadded[i:i + Hk, n:n + Wk]
            pic_mean = np.mean(subimage)
            fstd = np.std(subimage)
            fout = (subimage - pic_mean) / fstd
            #weighted sum
            out[i][n] = np.sum(fout * gout)
    ### END YOUR CODE

    return out
