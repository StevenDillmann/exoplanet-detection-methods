import george
from george import kernels, GP
import numpy as np
import emcee
from scipy.stats import uniform, lognorm, norm, beta
from dynesty import NestedSampler
import scipy.linalg as spl


def multi_stellar_gp(x, y, yerr, params_shared_bounds, params_individual_bounds, kepler_params_bounds_1, kepler_params_bounds_2, jitters, model):
    """
    Fit a multi-dimensional quasi-periodic Gaussian Process to the data and include a Keplerian model.

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
        The bounds for Keplerian parameters for model 1.
    kepler_params_bounds_2 : list
        The bounds for Keplerian parameters for model 2.
    jitters : array
        The jitter terms for each output.
    model : str
        The number of planets in the model.

    Returns
    -------
    sampler : dynesty.NestedSampler
        The nested sampler object.
    """

    # Define the log-likelihood function for GP fitting
    def log_likelihood_GP(K, y_res):
        factor, flag = spl.cho_factor(K)
        logdet = 2 * np.sum(np.log(np.diag(factor)))
        gof = np.dot(y_res, spl.cho_solve((factor, flag), y_res))
        return -0.5 * (gof + logdet + len(y_res) * np.log(2 * np.pi))

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
            u_P_kep, u_K, u_t_c, u_e, u_omega  = uparams[6:11]
            P_kep = u_P_kep * (kepler_params_bounds_1[0][1] - kepler_params_bounds_1[0][0]) + kepler_params_bounds_1[0][0]
            K = u_K * (kepler_params_bounds_1[1][1] - kepler_params_bounds_1[1][0]) + kepler_params_bounds_1[1][0]
            t_c = u_t_c * (kepler_params_bounds_1[2][1] - kepler_params_bounds_1[2][0]) + kepler_params_bounds_1[2][0]
            e = beta.ppf(u_e, kepler_params_bounds_1[3][0], kepler_params_bounds_1[3][1])
            omega = u_omega * (kepler_params_bounds_1[4][1] - kepler_params_bounds_1[4][0]) + kepler_params_bounds_1[4][0]
            return [P, lambda_p, lambda_e] + params_individual_transformed + [P_kep, K, t_c, e, omega]
        elif model == '2':
            u_P1, u_K1, u_t_c1, u_e1, u_omega1, u_P2, u_K2, u_t_c2, u_e2, u_omega2 = uparams[6:16]
            P1 = u_P1 * (kepler_params_bounds_2[0][1] - kepler_params_bounds_2[0][0]) + kepler_params_bounds_2[0][0]
            K1 = u_K1 * (kepler_params_bounds_2[1][1] - kepler_params_bounds_2[1][0]) + kepler_params_bounds_2[1][0]
            t_c1 = u_t_c1 * (kepler_params_bounds_2[2][1] - kepler_params_bounds_2[2][0]) + kepler_params_bounds_2[2][0]
            e1 = beta.ppf(u_e1, kepler_params_bounds_1[3][0], kepler_params_bounds_1[3][1])
            omega1 = u_omega1 * (kepler_params_bounds_2[4][1] - kepler_params_bounds_2[4][0]) + kepler_params_bounds_2[4][0]
            P2 = u_P2 * (kepler_params_bounds_2[5][1] - kepler_params_bounds_2[5][0]) + kepler_params_bounds_2[5][0]
            K2 = u_K2 * (kepler_params_bounds_2[6][1] - kepler_params_bounds_2[6][0]) + kepler_params_bounds_2[6][0]
            t_c2 = u_t_c2 * (kepler_params_bounds_2[7][1] - kepler_params_bounds_2[7][0]) + kepler_params_bounds_2[7][0]
            e2 = beta.ppf(u_e2, kepler_params_bounds_2[8][0], kepler_params_bounds_2[8][1])
            omega2 = u_omega2 * (kepler_params_bounds_2[9][1] - kepler_params_bounds_2[9][0]) + kepler_params_bounds_2[9][0]
            return [P, lambda_p, lambda_e] + params_individual_transformed + [P1, K1, t_c1, e1, omega1, P2, K2, t_c2, e2, omega2]

    # Initialise nested sampler
    ndim = 3 + 3 + (0 if model == '0' else (5 if model == '1' else 10))
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
    K = h**2 * np.exp(-((np.sin(np.pi * tau / P)**2)/(lambda_p**2) + (tau/lambda_e)**2)/2)
    np.fill_diagonal(K, np.diag(K) + jitter)
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
    K_blocks = []
    for h, jitter in zip(params_individual, jitters):
        K = K_qp(tau, h, P, lambda_p, lambda_e, jitter)
        K_blocks.append(K)
    K_multi = spl.block_diag(*K_blocks)
    return K_multi

def gp_predict(X_train, Y_train, X_test, params_shared, params_individual, jitters, model, keplerian_params):
    """
    Predict new data points using a Gaussian Process model with a Keplerian model.

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
        The number of planets in the model.
    keplerian_params : list
        The Keplerian parameters.

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
                y_model = keplerian_model_0(X_test, keplerian_params)
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
    Compute the Keplerian model for a single planet.
    """
    P, K, t_c, e, omega  = params
    M = 2 * np.pi * (t - t_c) / P  # Mean anomaly
    E = solve_kepler(M, e)  # Eccentric anomaly
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))  # True anomaly
    v = K * (np.cos(theta + omega) + e * np.cos(omega))
    return v

def keplerian_model_2(t, params):
    """
    Compute the Keplerian model for two planets.
    """
    P1, K1, t_c1, e1, omega1, P2, K2, t_c2, e2, omega2 = params
    M1 = 2 * np.pi * (t - t_c1) / P1  # Mean anomaly for planet 1
    M2 = 2 * np.pi * (t - t_c2) / P2  # Mean anomaly for planet 2
    E1 = solve_kepler(M1, e1)  # Eccentric anomaly for planet 1
    E2 = solve_kepler(M2, e2)  # Eccentric anomaly for planet 2
    theta1 = 2 * np.arctan2(np.sqrt(1 + e1) * np.sin(E1 / 2), np.sqrt(1 - e1) * np.cos(E1 / 2))  # True anomaly for planet 1
    theta2 = 2 * np.arctan2(np.sqrt(1 + e2) * np.sin(E2 / 2), np.sqrt(1 - e2) * np.cos(E2 / 2))  # True anomaly for planet 2
    v1 = K1 * (np.cos(theta1 + omega1) + e1 * np.cos(omega1))
    v2 = K2 * (np.cos(theta2 + omega2) + e2 * np.cos(omega2))
    return v1 + v2  # Sum of two Keplerian signals

def solve_kepler(M, e, tol=1e-10):
    """Solve Kepler's equation M = E - e*sin(E) for E using Newton's method."""
    E = M
    for _ in range(100):
        E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        if np.all(np.abs(E - E_new) < tol):
            return E_new
        E = E_new
    return E