""" Problem Set 04 starter code

Please make sure your code runs on Python version 3.5.0

Due date: 2016-03-04 13:00
"""

import numpy as np
import scipy.optimize

def my_dual_svm(X, y, C=1):
    """ Support vector machine - Dual problem

    SVM classification for a numeric test matrix. The
    returned result is the vector of coefficients from
    the support vector machine (beta, *not* alpha!).

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      C: positive numeric value giving the cost parameter in the SVM

    Returns:
      a 1d numpy array of length p giving the coefficients of beta in
      the SVM model
    """
    import random
    n = y.size
    p = X.shape[1]
    bounds = [(0, C)] * n
    init_guess = []
    for i in range(0, n):
      init_guess.append(random.uniform(0, 1))
    def func(alpha, sign = -1.0):
      """ Maximize the function value

      """
      n = alpha.size
      sum_alpha = sum(alpha)
      second = 0
      for i in range(0, n):
        for j in range(0, n):
          second += alpha[i] * alpha[j] * y[i] * y[j] * sum(X[i,] * X[j,])
      second *= -0.5
      re = sum_alpha + second
      return sign * re
    res = scipy.optimize.minimize(func, init_guess, method='SLSQP', bounds = bounds, options={'disp': True})
    alpha = res.x
    beta = np.dot(alpha * y, X)
    return beta # correct dimension



def my_primal_svm(X, y, lam=1, k=5, T=100):
    """ Support vector machine - Dual problem

    SVM classification for a numeric test matrix. The
    returned result is the vector of coefficients from
    the support vector machine (beta, *not* alpha!).

    Args:
      X: an n by p numpy array; the data matrix of predictors
      y: a length n numpy array; the observed response
      lam: positive numeric value giving the tuning parameter
        in the (primal, penalized format) of the support vector machine
      k: positive integer giving the number of samples selected in
        each iteration of the algorithm
      T: positive integer giving the total number of iteration to run

    Returns:
      a 1d numpy array of length p giving the coefficients of beta in
      the SVM model
    """
    import math
    n = X.shape[0]
    p = X.shape[1]
    w = np.random.rand(p) # float
    # scale w to 1/sqrt(lam)
    w = w / sum(np.square(w)) / math.sqrt(lam)

    for t in range(1, T + 1):
      A_t_order = np.random.permutation(n)[:k]
      A_t_x = X[A_t_order,] # m * p matrix
      A_t_y = y[A_t_order] # lenth m row vector
      inner_product = np.dot(w, np.transpose(A_t_x)) * A_t_y # to be debugged
      A_t_plus_order = np.where(inner_product < 1)
      eta = 1.0 / (lam * t)
      # print(A_t_y[A_t_plus_order])
      # print(A_t_x[A_t_plus_order,:])
      # print(np.dot(A_t_y[A_t_plus_order], A_t_x[A_t_plus_order, :])[0])
      w = (1 - eta * lam) * w + eta / k * np.dot(A_t_y[A_t_plus_order], A_t_x[A_t_plus_order, :])[0] # convert to 1-d row vector
      w = min(1, 1 / math.sqrt(lam) / sum(np.square(w))) * w
    return w
    #return np.zeros((X.shape[1])) # correct dimension
