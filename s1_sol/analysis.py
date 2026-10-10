'''Estimate detector calibration parameters and bootstrap their uncertainties.'''

from __future__ import annotations

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares, UnbinnedNLL
from scipy.stats import norm

PARAMETERS = ('lamb', 'delt', 'a', 'b', 'c')
METHODS = ('sample_ests', 'individual_fits', 'simultaneous_fit')


def model_mean(E0: np.ndarray, lamb: float, delt: float) -> np.ndarray:
    '''Return the mean reconstructed energy in GeV.'''
    return lamb * E0 + delt


def model_std(E0: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    '''Return the energy resolution in GeV for positive true energies.'''
    return np.sqrt(a**2 * E0 + b**2 + c**2 * E0**2)


def _minimise(fit: Minuit) -> Minuit:
    '''Minimise the cost and calculate HESSE uncertainties.'''
    fit.migrad().hesse()
    if not fit.valid:
        fit.strategy = 2
        fit.simplex().migrad().hesse()
    if not fit.valid:
        fit.migrad().hesse()
    if not fit.valid:
        raise RuntimeError(f'Fit did not converge: {fit.fmin}')
    return fit


def _resolution_limits(fit: Minuit) -> None:
    '''Select positive amplitudes for the three resolution terms.'''
    fit.limits['a'] = (0.01, None)
    fit.limits['b'] = (0.01, None)
    fit.limits['c'] = (0.001, None)


def compute_sample_estimates(
    E_arr: np.ndarray, E0_arr: np.ndarray, E0_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Calculate sample means, widths and approximate Gaussian standard errors.
    Args
        E_arr (ndarray): Reconstructed energies in GeV, shape (n_events,).
        E0_arr (ndarray): Corresponding true energies in GeV.
        E0_vals (ndarray): Energy groups to estimate, shape (n_groups,).
    Returns
        tuple: Arrays of means, mean errors, widths and width errors.
    '''
    estimates = []
    for E0 in E0_vals:
        sample = E_arr[E0_arr == E0]
        n = sample.size
        if n < 2:
            raise ValueError('Each energy group needs at least two measurements.')
        width = np.std(sample, ddof=1)
        estimates.append((np.mean(sample), width / np.sqrt(n),
                          width, width / np.sqrt(2 * (n - 1))))
    return tuple(np.asarray(estimates).T)


def compute_mle_estimates(
    E_arr: np.ndarray, E0_arr: np.ndarray, E0_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Minuit]]:
    '''Fit an unbinned Gaussian likelihood separately at each true energy.
    Args
        E_arr (ndarray): Reconstructed energies in GeV, shape (n_events,).
        E0_arr (ndarray): Corresponding true energies in GeV.
        E0_vals (ndarray): Energy groups to fit, shape (n_groups,).
    Returns
        tuple: Mean, mean-error, width and width-error arrays, and fitted objects.
    '''
    estimates, fits = [], []
    for E0 in E0_vals:
        residual = E_arr[E0_arr == E0] - E0

        def logpdf(x, mu, sigma):
            return norm.logpdf(x, loc=mu, scale=sigma)

        fit = Minuit(UnbinnedNLL(residual, logpdf, log=True),
                     mu=np.mean(residual), sigma=np.std(residual))
        fit.limits['sigma'] = (1e-9, None)
        _minimise(fit)
        estimates.append((fit.values['mu'] + E0, fit.errors['mu'],
                          fit.values['sigma'], fit.errors['sigma']))
        fits.append(fit)
    return (*tuple(np.asarray(estimates).T), fits)


def fit_trends(
    E0_vals: np.ndarray, means: np.ndarray, means_err: np.ndarray,
    stds: np.ndarray, stds_err: np.ndarray,
) -> tuple[Minuit, Minuit]:
    '''Fit the mean and resolution models by weighted least squares.
    Args
        E0_vals (ndarray): True energies in GeV, shape (n_groups,).
        means (ndarray): Estimated reconstructed energies in GeV.
        means_err (ndarray): Standard errors on the means in GeV.
        stds (ndarray): Estimated resolution widths in GeV.
        stds_err (ndarray): Standard errors on the widths in GeV.
    Returns
        tuple: Mean-model and resolution-model fits with HESSE errors.
    '''
    mean_fit = Minuit(LeastSquares(E0_vals, means, means_err, model_mean),
                      lamb=1.0, delt=0.0)
    width_fit = Minuit(LeastSquares(E0_vals, stds, stds_err, model_std),
                       a=1.0, b=1.0, c=0.01)
    _resolution_limits(width_fit)
    return _minimise(mean_fit), _minimise(width_fit)


def compute_simultaneous_fit(E_arr: np.ndarray, E0_arr: np.ndarray) -> Minuit:
    '''Fit all five calibration parameters in one unbinned likelihood.
    Args
        E_arr (ndarray): Reconstructed energies in GeV, shape (n_events,).
        E0_arr (ndarray): Corresponding true energies in GeV.
    Returns
        Minuit: Joint parameter estimates, covariance and HESSE errors.
    '''
    def logpdf(E, lamb, delt, a, b, c):
        return norm.logpdf(E, loc=model_mean(E0_arr, lamb, delt),
                           scale=model_std(E0_arr, a, b, c))

    fit = Minuit(UnbinnedNLL(E_arr, logpdf, log=True),
                 lamb=1.0, delt=2.0, a=0.4, b=1.2, c=0.04)
    _resolution_limits(fit)
    return _minimise(fit)


def summarise_fits(*fits: Minuit) -> dict:
    '''Collect fitted values and standard errors in parameter order.'''
    return {
        attribute: {name: float(getattr(fit, attribute)[name])
                    for fit in fits for name in fit.parameters}
        for attribute in ('values', 'errors')
    }


def bootstrap_analysis(
    E_arr: np.ndarray, E0_arr: np.ndarray, E0_vals: np.ndarray,
    n_bootstrap: int = 2500, seed: int = 42,
) -> dict[str, np.ndarray]:
    '''Resample paired events and refit all three estimators.
    Args
        E_arr (ndarray): Reconstructed energies in GeV, shape (n_events,).
        E0_arr (ndarray): Corresponding true energies in GeV.
        E0_vals (ndarray): Energy groups used by the two-stage estimators.
        n_bootstrap (int): Number of whole-sample bootstrap replicates.
        seed (int): Seed for sampling event indices with replacement.
    Returns
        dict: One array per method, shape (n_bootstrap, 5), ordered as PARAMETERS.
    '''
    rng = np.random.default_rng(seed)
    samples = {method: np.empty((n_bootstrap, 5)) for method in METHODS}
    for i in range(n_bootstrap):
        indices = rng.integers(0, E_arr.size, size=E_arr.size)
        energy, truth = E_arr[indices], E0_arr[indices]
        sample = compute_sample_estimates(energy, truth, E0_vals)
        individual = compute_mle_estimates(energy, truth, E0_vals)
        for method, estimates in zip(METHODS[:2], (sample, individual[:4])):
            mean_fit, width_fit = fit_trends(E0_vals, *estimates)
            samples[method][i] = (*mean_fit.values, *width_fit.values)
        samples['simultaneous_fit'][i] = compute_simultaneous_fit(energy, truth).values
    return samples
