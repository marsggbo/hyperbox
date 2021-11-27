import numpy as np


def twonn_dimension(data, return_xy=False):
    """
    https://github.com/fmottes/TWO-NN/blob/master/TwoNN/twonn_dimension.py
    Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.
    
    -----------
    Parameters:
    
    data : 2d array-like
        2d data matrix. Samples on rows and features on columns.
    return_xy : bool (default=False)
        Whether to return also the coordinate vectors used for the linear fit.
        
    -----------
    Returns:
    
    d : int
        Intrinsic dimension of the dataset according to TWO-NN.
    x : 1d array (optional)
        Array with the -log(mu) values.
    y : 1d array (optional)
        Array with the -log(F(mu_{sigma(i)})) values.
        
    -----------
    References:
    
    [1] E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
    
    
    """
    
    
    data = np.array(data)
    
    N = len(data)
    
    #mu = r2/r1 for each data point
    mu = []
    for i,x in enumerate(data):
        
        dist = np.sort(np.sqrt(np.sum((x-data)**2, axis=1)))
        r1, r2 = dist[dist>0][:2]

        mu.append((i+1,r2/r1))
        

    #permutation function
    sigma_i = dict(zip(range(1,len(mu)+1), np.array(sorted(mu, key=lambda x: x[1]))[:,0].astype(int)))

    mu = dict(mu)

    #cdf F(mu_{sigma(i)})
    F_i = {}
    for i in mu:
        F_i[sigma_i[i]] = i/N

    #fitting coordinates
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1-F_i[i] for i in sorted(mu.keys())])

    #avoid having log(0)
    x = x[y>0]
    y = y[y>0]

    y = -1*np.log(y)

    #fit line through origin to get the dimension
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
        
    if return_xy:
        return d, x, y
    else: 
        return d
