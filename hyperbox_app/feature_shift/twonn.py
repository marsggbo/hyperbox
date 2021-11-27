import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression

# TWO-NN METHOD FOR ESTIMATING INTRINSIC DIMENSIONALITY
# Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017).
# Estimating the intrinsic dimension of datasets by a minimal neighborhood information.
# Scientific reports, 7(1), 12140.

# https://github.com/jmmanley/two-nn-dimensionality-estimator/blob/master/twonn.py

# Implementation by Jason M. Manley, jmanley@rockefeller.edu
# June 2019


def estimate_id(X, plot=False, X_is_dist=False):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #   plot = Boolean flag of whether to plot fit
    #   X_is_dist = Boolean flag of whether X is an NxN distance metric instead
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality

    N = X.shape[0]

    if X_is_dist:
        dist = X
    else:
        # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric='euclidean'))

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i,:])
        mu[i] = dist[i,sort_idx[2]] / dist[i,sort_idx[1]]

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))

    d = lr.coef_[0][0] # extract slope

    if plot:
        # PLOT FIT THAT ESTIMATES INTRINSIC DIMENSION
        s=plt.scatter(np.log(mu[sort_idx]), -np.log(1-Femp), c='r', label='data')
        p=plt.plot(np.log(mu[sort_idx]), lr.predict(np.log(mu[sort_idx]).reshape(-1,1)), c='k', label='linear fit')
        plt.xlabel('$\log(\mu_i)$'); plt.ylabel('$-\log(1-F_{emp}(\mu_i))$')
        plt.title('ID = ' + str(np.round(d, 3)))
        plt.legend()

    return d
