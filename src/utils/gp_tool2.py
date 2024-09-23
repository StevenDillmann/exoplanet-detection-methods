import george
from george import kernels, GP
import numpy as np
import emcee
from scipy.stats import uniform, lognorm, norm
from dynesty import NestedSampler
import scipy.linalg as spl


def multi_stellar_gp(x, y, yerr, params_shared_bounds, params_individual_bounds, kepler_params_bounds_1, kepler_params_bounds_2, jitters, model):
    """
    Fit a multi-dimensional quasi-periodic Gaussian Process to the data and include simplified Keplerian models.

    Parameters
    ----------
    x : array
        The independent variable of the data. 
    y : array
        The dependent variable of the data.
    yerr : array
        The uncertainty in the dependent variable.
    params_shared_bounds : list
        The bounds for shared hyperparameters.
    params_individual_bounds : list
        The bounds for individual hyperparameters.
    kepler_params_bounds_1 : list
        The bounds for the Keplerian parameters for model 1.
    kepler_params_bounds_2 : list
        The bounds for the Keplerian parameters for model 2.
    jitters : array
        The jitter terms for each output.
    model : str
        The number of planets in the system.

    Returns
    -------
    sampler : dynesty.NestedSampler
        The nested sampler object.
    """
    # Define the log-likelihood function for GP fitting
    def log_likelihood_GP(K, y_res):
        factor, flag = spl.cho_factor(K) # Cholesky factorisation
        logdet = 2 * np.sum(np.log(np.diag(factor))) # Log determinant
        gof = np.dot(y_res, spl.cho_solve((factor, flag), y_res)) # Goodness of fit
        return -0.5 * (gof + logdet + len(y_res) * np.log(2 * np.pi)) # Log-likelihood

    # Define the log-likelihood function
    def ln_likelihood(params_combined, t, y, yerr, model):
        P, lambda_p, lambda_e = params_combined[:3]
        params_individual = np.array(params_combined[3:6]).reshape(3, 1)
        K = K_qp_multi(t, [P, lambda_p, lambda_e], params_individual, jitters)
        y_flatten = y.flatten()
        yerr_flatten = yerr.flatten()
        K += np.diag(yerr_flatten ** 2) 

        # Subtract Keplerian model from the mean function
        if model == '0':
            y_model = keplerian_model_0(t, params_combined[6:])
        elif model == '1':
            y_model = keplerian_model_1(t, params_combined[6:])
        elif model == '2':
            y_model = keplerian_model_2(t, params_combined[6:])

        y_model_combined = np.zeros_like(y_flatten)
        y_model_combined[:len(t)] = y_model
        
        y_res = y_flatten - y_model_combined

        return log_likelihood_GP(K, y_res)
    
    # Define the prior transform function
    def prior_transform(uparams, params_shared_bounds, params_individual_bounds, kepler_params_bounds_1, kepler_params_bounds_2, model):
        u_P, u_lambda_p, u_lambda_e = uparams[:3]
        P = u_P * (params_shared_bounds[0][1] - params_shared_bounds[0][0]) + params_shared_bounds[0][0]
        lambda_p = u_lambda_p * (params_shared_bounds[1][1] - params_shared_bounds[1][0]) + params_shared_bounds[1][0]
        lambda_e = u_lambda_e * (params_shared_bounds[2][1] - params_shared_bounds[2][0]) + params_shared_bounds[2][0]

        params_individual_transformed = []
        for i in range(3):
            u_h = uparams[3 + i]
            h = u_h * (params_individual_bounds[i][0][1] - params_individual_bounds[i][0][0]) + params_individual_bounds[i][0][0]
            params_individual_transformed.append(h)
    
        if model == '0':
            return [P, lambda_p, lambda_e] + params_individual_transformed
        elif model == '1':
            u_P_kep, u_K, u_t_c = uparams[6:9]
            P_kep = u_P_kep * (kepler_params_bounds_1[0][1] - kepler_params_bounds_1[0][0]) + kepler_params_bounds_1[0][0]
            K = u_K * (kepler_params_bounds_1[1][1] - kepler_params_bounds_1[1][0]) + kepler_params_bounds_1[1][0]
            t_c = u_t_c * (kepler_params_bounds_1[2][1] - kepler_params_bounds_1[2][0]) + kepler_params_bounds_1[2][0]
            return [P, lambda_p, lambda_e] + params_individual_transformed + [P_kep, K, t_c]
        elif model == '2':
            u_P1, u_K1, u_t_c1, u_P2, u_K2, u_t_c2 = uparams[6:12]
            P1 = u_P1 * (kepler_params_bounds_2[0][1] - kepler_params_bounds_2[0][0]) + kepler_params_bounds_2[0][0]
            K1 = u_K1 * (kepler_params_bounds_2[1][1] - kepler_params_bounds_2[1][0]) + kepler_params_bounds_2[1][0]
            t_c1 = u_t_c1 * (kepler_params_bounds_2[2][1] - kepler_params_bounds_2[2][0]) + kepler_params_bounds_2[2][0]
            P2 = u_P2 * (kepler_params_bounds_2[3][1] - kepler_params_bounds_2[3][0]) + kepler_params_bounds_2[3][0]
            K2 = u_K2 * (kepler_params_bounds_2[4][1] - kepler_params_bounds_2[4][0]) + kepler_params_bounds_2[4][0]
            t_c2 = u_t_c2 * (kepler_params_bounds_2[5][1] - kepler_params_bounds_2[5][0]) + kepler_params_bounds_2[5][0]
            return [P, lambda_p, lambda_e] + params_individual_transformed + [P1, K1, t_c1, P2, K2, t_c2]
    
    # Initialise nested sampler
    ndim = 3 + 3 + (0 if model == '0' else (3 if model == '1' else 6))
    sampler = NestedSampler(
        lambda theta: ln_likelihood(theta, x, y, yerr, model), 
        lambda utheta: prior_transform(utheta, params_shared_bounds, params_individual_bounds, kepler_params_bounds_1, kepler_params_bounds_2, model),
        ndim,
        bound='multi',
        sample='rwalk')
    
    # Run the nested sampling
    print(f'Running Nested Sampling for {model} planets...')
    sampler.run_nested(dlogz=0.01, print_progress=True)

    return sampler


# Define quasi-periodic kernel
def K_qp(tau, h, P, lambda_p, lambda_e, jitter):
    """
    Compute the quasi-periodic kernel matrix.

    Parameters
    ----------
    tau : array
        The time differences between data points.
    h : float
        The amplitude of the quasi-periodic kernel.
    P : float
        The period of the quasi-periodic kernel.
    lambda_p : float
        The decay length of the quasi-periodic kernel.
    lambda_e : float
        The length scale of the quasi-periodic kernel.

    Returns
    -------
    K : array
        The kernel matrix.

    """
    K = h**2 * np.exp(-((np.sin(np.pi * tau / P)**2)/(lambda_p**2) + (tau/lambda_e)**2)/2) # Quasi-periodic kernel
    np.fill_diagonal(K, np.diag(K) + jitter) # Add jitter term fo
    return K

# Define multi-dimensional quasi-periodic kernel
def K_qp_multi(x, params_shared, params_individual, jitters):
    """
    Compute the multi-dimensional quasi-periodic kernel matrix.

    Parameters
    ----------
    x : array
        The independent variable of the data.
    params_shared : list
        The shared hyperparameters.
    params_individual : list
        The individual hyperparameters.
    jitters : array
        The jitter terms for each output.

    Returns
    -------
    K_multi : array
        The kernel matrix.
    """
    tau = np.subtract.outer(x, x)
    P, lambda_p, lambda_e = params_shared
    K_blocks = [] # List to store the kernel matrices for each output
    for h, jitter in zip(params_individual, jitters): # Loop over each output
        K = K_qp(tau, h, P, lambda_p, lambda_e, jitter) # Compute the kernel matrix
        K_blocks.append(K) # Append the kernel matrix to the list
    K_multi = spl.block_diag(*K_blocks) # Combine the kernel matrices into a block diagonal matrix
    return K_multi

def gp_predict(X_train, Y_train, X_test, params_shared, params_individual, jitters, model, keplerian_params):
    """
    Predict new data points using the Gaussian Process model with simplified Keplerian models.
    
    Parameters
    ----------
    X_train : array 
        The independent variable of the training data.
    Y_train : array
        The dependent variable of the training data.
    X_test : array
        The independent variable of the test data.
    params_shared : list
        The shared hyperparameters.
    params_individual : list
        The individual hyperparameters.
    jitters : array
        The jitter terms for each output.
    model : str
        The number of planets in the system.
    keplerian_params : list
        The parameters of the Keplerian model.

    Returns
    -------
    means : array
        The predicted means.
    variances : array
        The predicted variances.    
    """
    # Get the number of training and test data points
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Compute the kernel matrix for the training data
    K_train = K_qp_multi(X_train, params_shared, params_individual, jitters)

    # Compute the inverse of the training kernel matrix
    K_train_inv = np.linalg.inv(K_train)

    # Predict mean and variance for each output
    means = []
    variances = []
    for output_idx in range(Y_train.shape[0]):
        Y_train_output = Y_train[output_idx]

        # Compute the kernel matrix between training and test data
        tau_train_test = np.subtract.outer(X_train, X_test)
        K_train_test = K_qp(tau_train_test, params_individual[output_idx], params_shared[0], params_shared[1], params_shared[2], jitters[output_idx])

        # Expand K_train_test to match the block structure
        K_train_test_full = np.zeros((n_train * Y_train.shape[0], n_test))
        K_train_test_full[output_idx * n_train:(output_idx + 1) * n_train, :] = K_train_test
        
        # Compute the kernel matrix for the test data
        tau_test = np.subtract.outer(X_test, X_test)
        K_test = K_qp(tau_test, params_individual[output_idx], params_shared[0], params_shared[1], params_shared[2], jitters[output_idx])
        
        # Predict mean
        mean = K_train_test_full.T @ K_train_inv @ Y_train.flatten()

        # Add Keplerian model to the mean prediction for the first dataset
        if output_idx == 0:
            if model == '0':
                y_model = 0
            elif model == '1':
                y_model = keplerian_model_1(X_test, keplerian_params)
            elif model == '2':
                y_model = keplerian_model_2(X_test, keplerian_params)
            mean += y_model
        
        # Predict variance
        variance = K_test - K_train_test_full.T @ K_train_inv @ K_train_test_full
        
        means.append(mean)
        variances.append(np.diag(variance))

    return np.array(means), np.array(variances)


def keplerian_model_0(t, params):
    return np.zeros_like(t)

def keplerian_model_1(t, params):
    """
    Simple Keplerian model with one planet
    """
    P, K, t_c = params
    return K * np.sin(2 * np.pi * (t - t_c) / P)

def keplerian_model_2(t, params):
    """
    Simple Keplerian model with two planets
    """
    P1, K1, t_c1, P2, K2, t_c2 = params
    return (K1 * np.sin(2 * np.pi * (t - t_c1) / P1) +
            K2 * np.sin(2 * np.pi * (t - t_c2) / P2))
