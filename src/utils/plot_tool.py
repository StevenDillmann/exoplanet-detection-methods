import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import matplotlib.gridspec as gridspec
from numpy.polynomial.polynomial import Polynomial
import seaborn as sns
import corner
import dynesty.plotting as dyplot
from dynesty.utils import resample_equal

def plot_time_series_all(df, cols = ['red', 'blue', 'green']):

    print("\nTime series\n----------------------")
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)
                           
    ax[0].errorbar(df['time'], df['rv'], yerr=df['rv_err'], fmt='.', color=cols[0], label='RV Data')
    ax[0].set_ylabel('RV [km/s]')
    ax[0].legend(loc = 'upper center')

    ax[1].errorbar(df['time'], df['fwhm'], yerr=df['fwhm_err'], fmt='.', color=cols[1], label='FWHM Data')
    ax[1].set_ylabel('FWHM [km/s]')
    ax[1].legend(loc = 'upper center')

    ax[2].errorbar(df['time'], df['bispan'], yerr=df['bispan_err'], fmt='.', color=cols[2], label='BS Data')
    ax[2].set_xlabel('Time [BJD]')
    ax[2].set_ylabel('BS [km/s]')
    ax[2].legend(loc = 'upper center')

    ax[0].tick_params(which='both', bottom = False)
    ax[1].tick_params(which='both', top = False, bottom = False)
    ax[2].tick_params(which='both', top = False)

    plt.tight_layout()
    plt.show()


def plot_periodograms_all(df, min_p=0.1, max_p=10000, n_p=1000000, cols = ['red', 'blue', 'green'], fap = None, plot_max = False, max_cols = ['red', 'blue', 'green']):

    # Define the frequency grid for the periodograms
    min_f = 1 / max_p
    max_f = 1 / min_p
    freqs = np.linspace(min_f, max_f, n_p)
    periods = 1 / freqs

    # Compute the periodogram for the RV data
    ls = LombScargle(df['time'], df['rv'], df['rv_err'])
    power = ls.power(frequency=freqs)

    # Compute the periodogram for the FWHM data
    ls_fwhm = LombScargle(df['time'], df['fwhm'], df['fwhm_err'])
    power_fwhm = ls_fwhm.power(frequency=freqs)

    # Compute the periodogram for the bs data
    ls_bs = LombScargle(df['time'], df['bispan'], df['bispan_err'])
    power_bs = ls_bs.power(frequency=freqs)

    # Plot the periodograms
    print("\nPeriodograms\n----------------------")
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)
    ax[0].plot(periods, power, color=cols[0], label='RV', alpha=0.5)
    ax[1].plot(periods, power_fwhm, color=cols[1], label='FWHM', alpha=0.5)
    ax[2].plot(periods, power_bs, color=cols[2], label='BS', alpha=0.5)

    # Get the powers for the FAP specified
    if fap is not None:
        # Get the power for the FAP specified
        power_fap = ls.false_alarm_level(fap)
        power_fap_fwhm = ls_fwhm.false_alarm_level(fap)
        power_fap_bs = ls_bs.false_alarm_level(fap)

        # Plot the FAP levels
        ax[0].axhline(power_fap, color=max_cols[0], linestyle='--', label='FAP = ' + str(fap*100) + '%')
        ax[1].axhline(power_fap_fwhm, color=max_cols[1], linestyle='--', label='FAP = ' + str(fap*100) + '%')
        ax[2].axhline(power_fap_bs, color=max_cols[2], linestyle='--', label='FAP = ' + str(fap*100) + '%')

        # Print all the periods above the FAP level, but dont print multiple similar periods
        print("Periods above FAP = ", fap)
        print("RV: ", periods[power > power_fap])
        print("FWHM: ", periods[power_fwhm > power_fap_fwhm])
        print("BS: ", periods[power_bs > power_fap_bs])

    # Plot and print maximum power
    if plot_max:
        max_power_period = periods[np.argmax(power)]
        max_power_period_fwhm = periods[np.argmax(power_fwhm)]
        max_power_period_bs = periods[np.argmax(power_bs)]
        max_power = np.max(power)
        max_power_fwhm = np.max(power_fwhm)
        max_power_bs = np.max(power_bs)

        ax[0].scatter(max_power_period, max_power, color=max_cols[0])
        ax[1].scatter(max_power_period_fwhm, max_power_fwhm, color=max_cols[1])
        ax[2].scatter(max_power_period_bs, max_power_bs, color=max_cols[2])

        # Print the maximum power period
        print("Maximum power period for RV: ", max_power_period)
        print("Maximum power period for FWHM: ", max_power_period_fwhm)
        print("Maximum power period for BS: ", max_power_period_bs)

        # Print the false alarm probability for the maximum power period
        print("FAP for maximum power period for RV: ", ls.false_alarm_probability(max_power, method='naive'))
        print("FAP for maximum power period for FWHM: ", ls_fwhm.false_alarm_probability(max_power_fwhm, method='naive'))
        print("FAP for maximum power period for BS: ", ls_bs.false_alarm_probability(max_power_bs, method='naive'))

    # Set plot properties
    ax[0].set_xscale('log')
    ax[0].set_ylabel('Normalised Power')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('Normalised Power')
    ax[2].set_xscale('log')
    ax[2].set_ylabel('Normalised Power')
    ax[0].tick_params(which='both', bottom = False)
    ax[1].tick_params(which='both', top = False, bottom = False)
    ax[2].tick_params(which='both', top = False)
    ax[0].legend(loc = 'upper right')
    ax[1].legend(loc = 'upper right')
    ax[2].legend(loc = 'upper right')

    plt.xlabel('Period [days]')
    plt.tight_layout()
    plt.show()

def plot_correlations(df, data_cols = ['blue', 'green'], model_cols = ['blue', 'green']):

    # Calculate the Pearson correlation coefficient and the p-value
    corr_fwhm, p_value_fwhm = pearsonr(df['fwhm'], df['rv'])
    corr_bispan, p_value_bispan = pearsonr(df['bispan'], df['rv'])
    print("\nCorrelation matrix\n----------------------")
    print(f"FWHM Pearson correlation coefficient: {corr_fwhm:.4f}, p-value: {p_value_fwhm:.4e}")
    print(f"BIS Pearson correlation coefficient: {corr_bispan:.4f}, p-value: {p_value_bispan:.4e}")

    # Correlation matrix
    corrs = df[['rv', 'fwhm', 'bispan']].corr()
    plt.figure(figsize=(5, 4))
    heatmap = sns.heatmap(corrs, annot=True, fmt='.2f', cmap='pink_r', cbar=True, vmin=0.5, vmax=1)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=13)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=13)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation Coefficient', fontsize=12)

    plt.show()

    # Fit a linear model to the data
    X_fwhm = np.array(df['fwhm'])
    X_bispan = np.array(df['bispan'])
    X_fwhm = sm.add_constant(X_fwhm)
    X_bispan = sm.add_constant(X_bispan)
    model_fwhm = sm.OLS(df['rv'], X_fwhm).fit()
    model_bispan = sm.OLS(df['rv'], X_bispan).fit()
    predicted_rv_fwhm = model_fwhm.predict(X_fwhm)
    predicted_rv_bispan = model_bispan.predict(X_bispan)
    residuals_fwhm = df['rv'] - predicted_rv_fwhm
    residuals_bispan = df['rv'] - predicted_rv_bispan

    print("FWHM model summary\n----------------------")
    print(model_fwhm.summary())
    print("\nBIS model summary\n----------------------")
    print(model_bispan.summary())

    # Get R^2 values
    r2_fwhm = model_fwhm.rsquared
    r2_bispan = model_bispan.rsquared

    # Create the figure for FWHM
    print("\nLinear model\n----------------------")
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Scatter plot and linear model for FWHM
    ax1.errorbar(df['fwhm'], df['rv'], yerr=df['rv_err'], xerr=df['fwhm_err'], fmt='.', color=data_cols[0], label='Data', capsize=0, alpha=0.75)
    ax1.plot(df['fwhm'], predicted_rv_fwhm, color=model_cols[0], label=r'Linear Model (R$^2$ = {:.2f})'.format(r2_fwhm))
    ax1.set_ylabel('RV [km/s]')
    ax1.legend()

    # Residuals plot for FWHM
    ax2.errorbar(df['fwhm'], residuals_fwhm, yerr=df['rv_err'], xerr=df['fwhm_err'], fmt='.', color=model_cols[0], capsize=0)
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_xlabel('FWHM [km/s]')
    ax2.set_ylabel('Residuals')
    plt.show()

    # Create the figure for BIS
    fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Scatter plot and linear model for BIS
    ax3.errorbar(df['bispan'], df['rv'], yerr=df['rv_err'], xerr=df['bispan_err'], fmt='.', color=data_cols[1], label='Data', capsize=0, alpha=0.75)
    ax3.plot(df['bispan'], predicted_rv_bispan, color=model_cols[1], label=r'Linear Model (R$^2$ = {:.2f})'.format(r2_bispan))
    ax3.set_ylabel('RV [km/s]')
    ax3.legend()

    # Residuals plot for BIS
    ax4.errorbar(df['bispan'], residuals_bispan, yerr=df['rv_err'], xerr=df['bispan_err'], fmt='.', color=model_cols[1], capsize=0)
    ax4.axhline(y=0, color='k', linestyle='--')
    ax4.set_xlabel('BS [km/s]')
    ax4.set_ylabel('Residuals')
    plt.show()

    return residuals_fwhm, residuals_bispan


def polynomial_trend(time, data, degree=1):
    pol = Polynomial.fit(time, data, degree)
    t_values = np.linspace(time.min(), time.max(), 1000)
    trend = pol(t_values)
    return trend, t_values, pol

def plot_long_term_trend(df, degrees = [1, 3, 5], cols = ['red', 'blue', 'green'], fit_cols = ['orange', 'purple', 'pink', 'turquoise'], ls = ['-', '--', '-.', ':']):

    print("\nLong-term trends\n----------------------")
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)

    ax[0].errorbar(df['time'], df['rv'], yerr=df['rv_err'], fmt='.', color=cols[0], label='RV Data')
    ax[1].errorbar(df['time'], df['fwhm'], yerr=df['fwhm_err'], fmt='.', color=cols[1], label='FWHM Data')
    ax[2].errorbar(df['time'], df['bispan'], yerr=df['bispan_err'], fmt='.', color=cols[2], label='BS Data')

    for i, degree in enumerate(degrees):
        trend, times, pol = polynomial_trend(df['time'], df['rv'], degree=degree)
        trend_fwhm, times, pol_fwhm = polynomial_trend(df['time'], df['fwhm'], degree=degree)
        trend_bispan, times, pol_bispan = polynomial_trend(df['time'], df['bispan'], degree=degree)
        r2 = 1 - np.sum((df['rv'] - pol(df['time']))**2) / np.sum((df['rv'] - np.mean(df['rv']))**2)
        r2_fwhm = 1 - np.sum((df['fwhm'] - pol_fwhm(df['time']))**2) / np.sum((df['fwhm'] - np.mean(df['fwhm']))**2)
        r2_bispan = 1 - np.sum((df['bispan'] - pol_bispan(df['time']))**2) / np.sum((df['bispan'] - np.mean(df['bispan']))**2)
        n = len(df['rv'])
        p = degree + 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        adj_r2_fwhm = 1 - (1 - r2_fwhm) * (n - 1) / (n - p - 1)
        adj_r2_bispan = 1 - (1 - r2_bispan) * (n - 1) / (n - p - 1)
        ax[0].plot(times, trend, color=fit_cols[i], label=f'n = {degree} ' + r'(Adj. R$^2$ = ' + f'{adj_r2:.2f})', linestyle = ls[i])
        ax[1].plot(times, trend_fwhm, color=fit_cols[i], label=f'n = {degree} ' + r'(Adj. R$^2$ = ' + f'{adj_r2_fwhm:.2f})', linestyle = ls[i])
        ax[2].plot(times, trend_bispan, color=fit_cols[i], label=f'n = {degree} ' + r'(Adj. R$^2$ = ' + f'{adj_r2_bispan:.2f})', linestyle = ls[i])
    
    ax[0].set_ylabel('RV [km/s]')
    ax[0].legend(loc = 'upper left', ncol=2)
    ax[1].set_ylabel('FWHM [km/s]')
    ax[1].legend(loc = 'upper left', ncol=2)
    ax[2].set_xlabel('Time [BJD]')
    ax[2].set_ylabel('BS [km/s]')
    ax[2].legend(loc = 'upper left', ncol=2)
    ax[0].tick_params(which='both', bottom = False)
    ax[1].tick_params(which='both', top = False, bottom = False)
    ax[2].tick_params(which='both', top = False)

    plt.tight_layout()
    plt.show()

    return None

def plot_stellar_gp_results(sampler, get_value = 'median', truth_color = 'k', corner_color = 'k', run_color = 'k', cmap = 'rainbow', post_color = 'k', alpha = 0.5, model = '0', fontsize = 15):

    # Extract the results
    results = sampler.results
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    normalized_weights = weights / np.sum(weights)
    log_likelihoods = results.logl
    posterior_samples = resample_equal(samples, normalized_weights)

    # Find the Mean and Standard Deviation
    weighted_mean = np.average(samples, axis=0, weights=normalized_weights)
    weighted_std = np.sqrt(np.average((samples - weighted_mean)**2, axis=0, weights=normalized_weights))
    print("Weighted Mean:", weighted_mean)
    print("Weighted Standard Deviation:", weighted_std)

    # Find the MAP estimate
    log_posteriors = log_likelihoods + results.logwt - results.logz[-1]
    map_index = np.argmax(log_posteriors)
    map_estimate = samples[map_index]
    print("MAP Estimate:", map_estimate)

    # Parameter labels
    labels = [r'$P_{rot}$', r'$\lambda_p$', r'$\lambda_e$', r'$h_{RV}$', r'$h_{FWHM}$', r'$h_{BS}$', r'$P_{1}$', r'$K_{1}$', r'$T_{c1}$',  r'$P_{2}$', r'$K_{2}$', r'$T_{c2}$']
    if model == '0':
        labels = labels[:6]
    elif model == '1':
        labels = labels[:9]
    elif model == '2':
        labels = labels

    # Find the best fit parameters
    if get_value == 'mean':
        best_fit_parameters = weighted_mean
        best_fit_errors = weighted_std
        print("Best Fit Parameters:", best_fit_parameters)
        print("Error Estimates:", best_fit_errors)
    elif get_value == 'map':
        best_fit_parameters = map_estimate
        best_fit_errors = np.std(posterior_samples, axis=0)
        print("Best Fit Parameters:", best_fit_parameters)
        print("Error Estimates:", best_fit_errors)
    elif get_value == 'median':
        percentiles = [16, 50, 84]
        percentiles2 = [2.5, 50, 97.5]
        best_fit_parameters = np.median(posterior_samples, axis=0)
        lower = np.percentile(posterior_samples, percentiles[0], axis=0)
        upper = np.percentile(posterior_samples, percentiles[2], axis=0)
        lower2 = np.percentile(posterior_samples, percentiles2[0], axis=0)
        upper2 = np.percentile(posterior_samples, percentiles2[2], axis=0)
        # Print the best fit parameters with their 68% credible intervals
        for i, param in enumerate(labels):
            lower68 = best_fit_parameters[i] - lower[i]
            upper68 = upper[i] - best_fit_parameters[i]
            lower95 = best_fit_parameters[i] - lower2[i]
            upper95 = upper2[i] - best_fit_parameters[i]
            print(f"{param} = {best_fit_parameters[i]:.5f} with 68% credible interval [-{lower68:.5f}, +{upper68:.5f}] and 95% credible interval [-{lower95:.5f}, +{upper95:.5f}]")
 
    # Plot the results
    dyplot.runplot(results, color = run_color)
    fig = corner.corner(posterior_samples, labels=labels, truths=best_fit_parameters, truth_color=truth_color, color = corner_color, show_titles=True,  title_kwargs={"fontsize": fontsize}, label_kwargs={"fontsize": fontsize})
    fig, axes = dyplot.traceplot(results, labels=labels, truths=best_fit_parameters, truth_color=truth_color, show_titles=True, trace_cmap=cmap, post_color = post_color, post_kwargs={"alpha": alpha}, title_kwargs={"fontsize": fontsize}, label_kwargs={"fontsize": fontsize})
    plt.show()

    return best_fit_parameters


def plot_gp_predict(x, y, yerr, x_test, mean, variance, colors_data = ['red', 'blue', 'green'], colors = ['red', 'blue', 'green'], alpha = 0.5):

    # Predictions
    labels = ['RV', 'FWHM', 'BS']
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)
    for i in range(y.shape[0]):
        ax[i].errorbar(x, y[i], yerr[i], fmt='.', label=f'{labels[i]} Data', color = colors_data[i])
        ax[i].plot(x_test, mean[i], 'b', lw=2, label='GP Mean', color = colors[i])
        ax[i].fill_between(x_test, 
                        mean[i] - 1.96 * np.sqrt(variance[i]), 
                        mean[i] + 1.96 * np.sqrt(variance[i]), 
                        color=colors[i], alpha=alpha, label='95% CI')
        ax[i].set_ylabel(f'{labels[i]} [km/s]')
        ax[i].legend(ncol = 3, loc = 'upper right')
    ax[2].set_xlabel('Time [days]')
    plt.tight_layout()
    plt.show()

    return


def plot_gp_predict(x, y, yerr, x_test, mean, variance, colors_data = ['red', 'blue', 'green'], colors = ['red', 'blue', 'green'], alpha = 0.5):

    # Predictions
    labels = ['RV', 'FWHM', 'BS']
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7), sharex=True)
    for i in range(y.shape[0]):
        ax[i].errorbar(x, y[i], yerr[i], fmt='.', label=f'{labels[i]} Data', color = colors_data[i])
        ax[i].plot(x_test, mean[i], 'b', lw=2, label='GP Mean', color = colors[i])
        ax[i].fill_between(x_test, 
                        mean[i] - 1.96 * np.sqrt(variance[i]), 
                        mean[i] + 1.96 * np.sqrt(variance[i]), 
                        color=colors[i], alpha=alpha, label='95% CI')
        ax[i].set_ylabel(f'{labels[i]} [km/s]')
        ax[i].legend(ncol = 3, loc = 'upper right')
    ax[2].set_xlabel('Time [days]')
    plt.tight_layout()
    plt.show()

    return

def plot_stellar_gp_results2(sampler, get_value = 'median', truth_color = 'k', corner_color = 'k', run_color = 'k', cmap = 'rainbow', post_color = 'k', alpha = 0.5, model = '1', fontsize = 15):

    # Extract the results
    results = sampler.results
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    normalized_weights = weights / np.sum(weights)
    log_likelihoods = results.logl
    posterior_samples = resample_equal(samples, normalized_weights)

    # Find the Mean and Standard Deviation
    weighted_mean = np.average(samples, axis=0, weights=normalized_weights)
    weighted_std = np.sqrt(np.average((samples - weighted_mean)**2, axis=0, weights=normalized_weights))
    print("Weighted Mean:", weighted_mean)
    print("Weighted Standard Deviation:", weighted_std)

    # Find the MAP estimate
    log_posteriors = log_likelihoods + results.logwt - results.logz[-1]
    map_index = np.argmax(log_posteriors)
    map_estimate = samples[map_index]
    print("MAP Estimate:", map_estimate)

    # Parameter labels
    labels = [r'$P_{rot}$', r'$\lambda_p$', r'$\lambda_e$', r'$h_{RV}$', r'$h_{FWHM}$', r'$h_{BS}$', r'$P_{1}$', r'$K_{1}$', r'$T_{c1}$',  r'$e_{1}$', r'$\omega_{1}$', r'$P_{2}$', r'$K_{2}$', r'$T_{c2}$',  r'$e_{2}$', r'$\omega_{2}$']
    if model == '0':
        labels = labels[:6]
    elif model == '1':
        labels = labels[:11]
    elif model == '2':
        labels = labels

    # Find the best fit parameters
    if get_value == 'mean':
        best_fit_parameters = weighted_mean
        best_fit_errors = weighted_std
        print("Best Fit Parameters:", best_fit_parameters)
        print("Error Estimates:", best_fit_errors)
    elif get_value == 'map':
        best_fit_parameters = map_estimate
        best_fit_errors = np.std(posterior_samples, axis=0)
        print("Best Fit Parameters:", best_fit_parameters)
        print("Error Estimates:", best_fit_errors)
    elif get_value == 'median':
        percentiles = [16, 50, 84]
        percentiles2 = [2.5, 50, 97.5]
        best_fit_parameters = np.median(posterior_samples, axis=0)
        lower = np.percentile(posterior_samples, percentiles[0], axis=0)
        upper = np.percentile(posterior_samples, percentiles[2], axis=0)
        lower2 = np.percentile(posterior_samples, percentiles2[0], axis=0)
        upper2 = np.percentile(posterior_samples, percentiles2[2], axis=0)
        # Print the best fit parameters with their 68% credible intervals
        for i, param in enumerate(labels):
            lower68 = best_fit_parameters[i] - lower[i]
            upper68 = upper[i] - best_fit_parameters[i]
            lower95 = best_fit_parameters[i] - lower2[i]
            upper95 = upper2[i] - best_fit_parameters[i]
            print(f"{param} = {best_fit_parameters[i]:.5f} with 68% credible interval [-{lower68:.5f}, +{upper68:.5f}] and 95% credible interval [-{lower95:.5f}, +{upper95:.5f}]")
           
    # Plot the results
    dyplot.runplot(results, color = run_color)
    fig = corner.corner(posterior_samples, labels=labels, truths=best_fit_parameters, truth_color=truth_color, color = corner_color, show_titles=True,  title_kwargs={"fontsize": fontsize}, label_kwargs={"fontsize": fontsize})
    fig, axes = dyplot.traceplot(results, labels=labels, truths=best_fit_parameters, truth_color=truth_color, show_titles=True, trace_cmap=cmap, post_color = post_color, post_kwargs={"alpha": alpha}, title_kwargs={"fontsize": fontsize}, label_kwargs={"fontsize": fontsize})
    plt.show()

    return best_fit_parameters