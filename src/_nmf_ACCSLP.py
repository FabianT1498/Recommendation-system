""" Non-negative matrix factorization applied to recommendation system using ACCSLP model.
"""
# Author: Fabian Trillo
#         
#   References:
#        
#   [1] Tang, M., & Wang, W. (2023). Cold-start link prediction integrating community information via multi-nonnegative
#    matrix factorization. Chaos, Solitons & Fractals, 162. https://www.sciencedirect.com/science/article/pii/S0960077922006312


import time
import warnings
from math import sqrt

import numpy as np
import scipy.sparse as sp

from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._param_validation import (
    StrOptions
)
from sklearn.utils.validation import check_non_negative

from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils import check_array, check_random_state

import sys

np.set_printoptions(threshold=sys.maxsize)

EPSILON = np.finfo(np.float32).eps

def norm(x):
    """Dot product-based Euclidean norm implementation.

    See: http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix.
    Y : array-like
        Second matrix.
    """
    return np.dot(X.ravel(), Y.ravel())

def _beta_divergence(S, X, Z, U, H, W, V, alpha, beta, loss_function="kullback-leibler"):
    """Compute the beta-divergence of X and dot(W, H).

    Parameters
    ----------
    S : float or array-like of shape (n_samples, n_features)

    X : float or array-like of shape (n_samples, n_features)
    
    Z : float or array-like of shape (n_samples, n_features)

    W : float or array-like of shape (n_samples, n_components)

    H : float or array-like of shape (n_components, n_features)

    W : float or array-like of shape (n_samples, n_components)

    V : float or array-like of shape (n_components, n_features)

    loss_function : 1 or 'kullback-leibler'
      
    Returns
    -------
        res : float
            Beta divergence: D(S|UH) + alpha * D(X | WH) + beta (Z | UV) 
        
        ------------------------------------------------------------
        References:
        
        [1] Tang, M., & Wang, W. (2023). Cold-start link prediction integrating community information via multi-nonnegative
            matrix factorization. Chaos, Solitons & Fractals, 162. https://www.sciencedirect.com/science/article/pii/S0960077922006312

    """
    loss_function = _beta_loss_to_float(loss_function)

    # The method can be called with scalars
    if not sp.issparse(S):
        S = np.atleast_2d(S)
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    if not sp.issparse(Z):
        Z = np.atleast_2d(Z)

    U = np.atleast_2d(U)
    H = np.atleast_2d(H)
    W = np.atleast_2d(W)
    V = np.atleast_2d(V)

    if sp.issparse(S):
        # compute np.dot(W, H) only where X is nonzero
        UH_data = _special_sparse_dot(U, H, S).data
        S_data = S.data
    else:
        UH = np.dot(U, H)
        UH_data = UH.ravel()
        S_data = S.ravel()

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    if sp.issparse(Z):
        # compute np.dot(W, H) only where X is nonzero
        UV_data = _special_sparse_dot(U, V, Z).data
        Z_data = Z.data
    else:
        UV = np.dot(U, V)
        UV_data = UV.ravel()
        Z_data = Z.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = S_data > EPSILON
    S_data = S_data[indices]
    UH_data = UH_data[indices]

    # indices = X_data > EPSILON
    # X_data = X_data[indices]
    # WH_data = WH_data[indices]
    zero_indices = WH_data == 0.0
    WH_data[zero_indices] = EPSILON
  
    # indices = Z_data > EPSILON
    # UV_data = UV_data[indices]
    # Z_data = Z_data[indices]
    zero_indices = UV_data == 0.0
    UV_data[zero_indices] = EPSILON

    # ---- Computes np.sum(np.dot(U, H)) - S * np.sum((UH/sum_UH) * log(UH /(UH/sum_UH))) ----

    # fast and memory efficient computation of np.sum(np.dot(U, H))
    sum_UH = np.dot(np.sum(U, axis=0), np.sum(H, axis=1))

    res = np.dot(S_data, np.log(UH_data))

    res_S = sum_UH - res

    # ---- Computes np.sum(np.dot(W, H)) - X * np.sum((WH/sum_WH) * log(WH /(WH/sum_WH))) ----

    sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
 
    res = np.dot(X_data, np.log(WH_data))

    res_X = alpha * (sum_WH - res)

    # ---- Computes np.sum(np.dot(U, V)) - Z * np.sum((UV/sum_UV) * log(UV /(UV/sum_UV))) ----

    sum_UV = np.dot(np.sum(U, axis=0), np.sum(V, axis=1))
    
    res = np.dot(Z_data, np.log(UV_data))

    res_Z = beta * (sum_UV - res)

    res = res_S + res_X + res_Z

    return res

def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)

def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float."""
    beta_loss_map = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    if isinstance(beta_loss, str):
        beta_loss = beta_loss_map[beta_loss]
    return beta_loss

def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None, matrix_target="Matrix S"):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    print("Initializing ", matrix_target)

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:

            if x_p_nrm != 0:
                u = x_p / x_p_nrm
            else:
                u = np.zeros_like(x_p)

            if y_p_nrm != 0:    
                v = y_p / y_p_nrm
            else:
                v = np.zeros_like(y_p)
                
            sigma = m_p
        else:
            if x_n_nrm != 0:
                u = x_n / x_n_nrm
            else:
                u = np.zeros_like(x_n)

            if y_n_nrm != 0:
                v = y_n / y_n_nrm
            else:
                v = np.zeros_like(y_n)
                
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H

def _multiplicative_update_u(
    S,
    Z,
    U,
    H,
    V,
    beta,
):
    """Update U in Multiplicative Update NMF."""
 
    # Numerator

    # if S is sparse, compute WH only where S is non zero
    UH_safe_S = _special_sparse_dot(U, H, S)
    if sp.issparse(S):
        UH_safe_S_data = UH_safe_S.data
        S_data = S.data
    else:
        UH_safe_S_data = UH_safe_S
        S_data = S
        
    # if Z is sparse, compute UV only where Z is non zero
    UV_safe_Z = _special_sparse_dot(U, V, Z)
    if sp.issparse(Z):
        UV_safe_Z_data = UV_safe_Z.data
        Z_data = Z.data
    else:
        UV_safe_Z_data = UV_safe_Z
        Z_data = Z
        
     # to avoid division by zero
    UH_safe_S_data[UH_safe_S_data < EPSILON] = EPSILON
    UV_safe_Z_data[UV_safe_Z_data < EPSILON] = EPSILON

    np.divide(S_data, UH_safe_S_data, out=UH_safe_S_data)
    np.divide(Z_data, UV_safe_Z_data, out=UV_safe_Z_data)
   
    # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
    numerator = safe_sparse_dot(UH_safe_S, H.T)
    numerator += (beta * safe_sparse_dot(UV_safe_Z, V.T))

    # Denominator
    H_sum = np.sum(H, axis=1)  # shape(n_components, )
    V_sum = np.sum(V, axis=1)  # shape(n_components, )
    denominator = H_sum[np.newaxis, :]
    denominator += (beta * V_sum[np.newaxis, :])

    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_U = numerator

    U *= delta_U

    return U

def _multiplicative_update_h(
    S,
    X,
    U,
    W,
    H,
    alpha,
):
    """Update H in Multiplicative Update NMF."""
 
    # Numerator

    # if S is sparse, compute WH only where S is non zero
    UH_safe_S = _special_sparse_dot(U, H, S)
    if sp.issparse(S):
        UH_safe_S_data = UH_safe_S.data
        S_data = S.data
    else:
        UH_safe_S_data = UH_safe_S
        S_data = S

    # if X is sparse, compute WH only where X is non zero
    WH_safe_X = _special_sparse_dot(W, H, X)
    if sp.issparse(X):
        WH_safe_X_data = WH_safe_X.data
        X_data = X.data
    else:
        WH_safe_X_data = WH_safe_X
        X_data = X
       
    # to avoid division by zero
    UH_safe_S_data[UH_safe_S_data < EPSILON] = EPSILON
    WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

    np.divide(S_data, UH_safe_S_data, out=UH_safe_S_data)
    np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
   
    # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
    numerator = safe_sparse_dot(U.T, UH_safe_S)
    numerator += (alpha * safe_sparse_dot(W.T, WH_safe_X))

    # Denominator
    U_sum = np.sum(U, axis=0)  # shape(n_components, )
    W_sum = np.sum(W, axis=0)  # shape(n_components, )
    denominator = U_sum[:, np.newaxis]
    denominator += (alpha * W_sum[:, np.newaxis])

    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_H = numerator

    H *= delta_H

    return H

def _multiplicative_update_w(
    X,
    W,
    H,
):
    """Update W in Multiplicative Update NMF."""
 
    # Numerator

    # if X is sparse, compute WH only where X is non zero
    WH_safe_X = _special_sparse_dot(W, H, X)
    if sp.issparse(X):
        WH_safe_X_data = WH_safe_X.data
        X_data = X.data
    else:
        WH_safe_X_data = WH_safe_X
        X_data = X
    
    # to avoid taking a negative power of zero
    WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON

    np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
   
    # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
    numerator = safe_sparse_dot(WH_safe_X, H.T)

    # Denominator
    H_sum = np.sum(H, axis=1)  # shape(n_components, )
    denominator = H_sum[np.newaxis, :]
       
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    W *= delta_W


    return W

def _multiplicative_update_v(
    Z,
    U,
    V,
):
    """Update V in Multiplicative Update NMF."""
 
    # Numerator

    # if X is sparse, compute WH only where X is non zero
    UV_safe_Z = _special_sparse_dot(U, V, Z)
    if sp.issparse(Z):
        UV_safe_Z_data = UV_safe_Z.data
        Z_data = Z.data
    else:
        UV_safe_Z_data = UV_safe_Z
        Z_data = Z
       
    # to avoid taking a negative power of zero
    UV_safe_Z_data[UV_safe_Z_data < EPSILON] = EPSILON

    np.divide(Z_data, UV_safe_Z_data, out=UV_safe_Z_data)
   
    # here numerator = dot(Z * (dot(W, H) ** (beta_loss - 2)), H.T)
    numerator = safe_sparse_dot(U.T, UV_safe_Z)

    # Denominator
    U_sum = np.sum(U, axis=0)  # shape(n_components, )

    denominator = U_sum[:, np.newaxis]
       
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_V = numerator

    V *= delta_V

  
    return V

def _fit_multiplicative_update(
    S,
    X,
    Z,
    U,
    H,
    W,
    V,
    alpha,
    beta,
    beta_loss="kullback-leibler",
    max_iter=200,
    tol=1e-4,
    verbose=0,
):
    """Compute Non-negative Matrix Factorization with Multiplicative Update.

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of U, H, W and V. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : ''float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='kullback-leibler'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between S
        and the dot product UH.

    max_iter : int, default=200
        Number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # used for the convergence criterion
    error_at_init = _beta_divergence(S=S, X=X, Z=Z, U=U, H=H, W=W, V=V, alpha=alpha, beta=beta, loss_function=beta_loss)
    previous_error = error_at_init

    for n_iter in range(1, max_iter + 1):
        # update U
        U = _multiplicative_update_u(
            S=S,
            Z=Z,
            U=U,
            H=H,
            V=V,
            beta=beta,
        )

        H = _multiplicative_update_h(
            S=S,
            X=X,
            U=U,
            W=W,
            H=H,
            alpha=alpha,
        )

        W = _multiplicative_update_w(
            X=X,
            W=W,
            H=H,
        )

        V = _multiplicative_update_v(
            Z=Z,
            U=U,
            V=V,
        )

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(S=S, X=X, Z=Z, U=U, H=H, W=W, V=V, alpha=alpha, beta=beta, loss_function=beta_loss)

            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            print("(previous_error - error) / error_at_init):", (previous_error - error) / error_at_init)
            
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )

    return U, H, n_iter

class NMF_ACCSLP():
    """Non-Negative Matrix Factorization (NMF) adapted to prediction model ACCSLP.

    Parameters
    ----------
    n_components : int or {'auto'} or None, default=None
        Number of components, if n_components is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from W or H shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'nndsvd', 'nndsvda'}, default=nndsvd
        
        Method used to initialize the procedure.
        Valid options:

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)

    solver : {'mu'}, default='mu'
        Numerical solver to use:
        - 'mu' is a Multiplicative Update solver.

        Note: The multiplicative update ('mu') solver cannot update 
        zeros present in the initialization, and so leads to
        poorer results when used jointly with init='nndsvd'.
      
    beta_loss : float or { 'kullback-leibler' }, default='kullback-leibler'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    alpha : float, default=20.0
        Balance parameter used to constraint the role of the extracted
        information during training in the following loss function equation.

        D(S|UH) + alpha * D(X | WH) + beta (Z | UV) 

    beta : float or "same", default="27.0"
        Balance parameter used to constraint the role of the extracted
        information during training in the following loss function equation.

        D(S|UH) + alpha * D(X | WH) + beta (Z | UV) 
        
    verbose : int, default=0
        Whether to be verbose.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0
    """

    _parameter_constraints: dict = {
        "init": [StrOptions({"nndsvd", "nndsvda"})],
        "solver": [StrOptions({"mu"})],
    }

    def __init__(
        self,
        n_components="auto",
        *,
        init="nndsvd",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=200,
        tol=1e-4,
        alpha=20.0,
        beta=27.0,
        verbose=0,
    ):
       
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.solver = solver
        self.alpha = alpha
        self.beta = beta

    def _check_params(self, X):

        self._n_components = self.n_components
        
        # initialize method
        if self.init not in ("nndsvda", "nndsvd"):
            # This class only uses 'nndsvd' and 'nndsvda' for matrix initialization
            raise ValueError(
                f"This class only uses 'nndsvd' or 'nndsvda' for matrix initializacion")

        # solver
        if self.solver != "mu" or self.beta_loss not in (1, "kullback-leibler"):
            # This class only uses 'mu' and beta loss function 'kullback-leibler to solve matrix optimization problem
            raise ValueError(
                f"This class only uses 'mu' and beta loss function 'kullback-leibler' to solve matrix optimization problem"
            )

        # beta_loss
        self._beta_loss = _beta_loss_to_float(self.beta_loss)
  
        if self.solver == "mu" and self.init == "nndsvd":
            warnings.warn(
                (
                    "The multiplicative update ('mu') solver cannot update "
                    "zeros present in the initialization, and so leads to "
                    "poorer results when used jointly with init='nndsvd'. "
                    "You may try init='nndsvda' or init='nndsvdar' instead."
                ),
                UserWarning,
            )
        
        return self

    def fit_transform(self, S, Z, X, y=None, U=None, H=None):
        """Learn a NMF model ACCSLP for the data S, Z, X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        S : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            adjancency matrix

        Z : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            attribute similarity matrix

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            community membership information matrix

        y : Ignored
            Not used, present for API consistency by convention.

        U : array-like of shape (n_samples, n_components), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`. Ignored

        H : array-like of shape (n_components, n_features), default=None
            If `init='custom'`, it is used as initial guess for the solution.
            If `None`, uses the initialisation method specified in `init`. Ignored

        Returns
        -------
        U : array-like of shape (n_samples, n_components), default=None
            Transformed data.
        """
        S = check_array(
            S, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        Z = check_array(
            Z, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        X = check_array(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

      
        U, H, n_iter = self._fit_transform(S, Z, X)

        # self.reconstruction_err_ = _beta_divergence(
        #     X, W, H, self._beta_loss, square_root=True
        # )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        return U

    def _fit_transform(self, S, Z, X):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        S : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            adjancency matrix

        Z : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            attribute similarity matrix

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. This is the
            community membership information matrix

        Returns
        -------
        U : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """
        check_non_negative(S, "NMF (input S)")
        check_non_negative(Z, "NMF (input Z)")
        check_non_negative(X, "NMF (input X)")

        # check parameters
        self._check_params(S)

        if (S.min() == 0 or Z.min() == 0 or X.min() == 0) and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        # initialize or check W and H
        U, H = self._initialize_w_h(S, matrix_target="Matrix S")
        W, R = self._initialize_w_h(X, matrix_target="Matrix X")
        L, V = self._initialize_w_h(Z, matrix_target="Matrix Z")

        n_iter = 0

        if self.solver == "mu":
            U, H, n_iter, *_ = _fit_multiplicative_update(
                S=S,
                X=X,
                Z=Z,
                U=U,
                H=H,
                W=W,
                V=V,
                alpha=self.alpha,
                beta=self.beta,
                beta_loss=self._beta_loss,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence."
                % self.max_iter,
                ConvergenceWarning,
            )

        return U, H, n_iter

    def _initialize_w_h(self, X, matrix_target="Matrix S"):
        """initialize W and H."""

        if self._n_components == "auto":
            self._n_components = X.shape[1]

        W, H = _initialize_nmf(
            X, self._n_components, init=self.init, matrix_target=matrix_target
        )

        return W, H