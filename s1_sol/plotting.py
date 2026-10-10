'''Plot calibration fits and bootstrap parameter distributions.'''

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from scipy.stats import norm

from .analysis import METHODS, PARAMETERS, model_mean, model_std

LABELS = ('Sample statistics', 'Individual MLE', 'Simultaneous MLE')
PARAMETER_LABELS = (r'$\lambda$', r'$\Delta$ [GeV]', r'$a$ [$\sqrt{\mathrm{GeV}}$]',
                    r'$b$ [GeV]', r'$c$')


def save_figure(fig: plt.Figure, name: str) -> None:
    '''Save a figure as a PNG in the local figs directory.'''
    Path('figs').mkdir(exist_ok=True)
    fig.savefig(Path('figs') / f'{name}.png', dpi=160, bbox_inches='tight')


def plot_distributions(
    energy: np.ndarray, truth: np.ndarray, levels: np.ndarray,
    means: np.ndarray, widths: np.ndarray,
) -> plt.Figure:
    '''Compare energy residuals with fitted Gaussian densities.
    Args
        energy (ndarray): Measured energies in GeV, shape (n_events,).
        truth (ndarray): Corresponding true energies in GeV.
        levels (ndarray): Distinct true energies in GeV.
        means (ndarray): Fitted means in GeV, one per energy group.
        widths (ndarray): Fitted widths in GeV, one per energy group.
    Returns
        Figure: Grouped and pooled residual distributions.
    '''
    residual = energy - truth
    grid = np.linspace(residual.min() - 1, residual.max() + 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mixture = np.zeros_like(grid)
    for i, level in enumerate(levels):
        mask = truth == level
        density = norm.pdf(grid, means[i] - level, widths[i])
        axes[0].hist(residual[mask], bins=30, density=True, histtype='step',
                     color=f'C{i}', alpha=0.5)
        axes[0].plot(grid, density, color=f'C{i}', label=f'{level:g} GeV', lw=1)
        mixture += np.mean(mask) * density
    axes[1].hist(residual, bins=50, density=True, color='0.75', label='Data')
    axes[1].plot(grid, mixture, color='C0', label='Gaussian mixture')
    for ax in axes:
        ax.set(xlabel=r'$E-E_0$ [GeV]', ylabel='Density [GeV$^{-1}$]')
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_calibration(
    levels: np.ndarray, sample: tuple, fit: Minuit, seed: int = 41,
) -> plt.Figure:
    '''Show the joint calibration model with covariance-based uncertainty bands.
    Args
        levels (ndarray): True energy values in GeV.
        sample (tuple): Sample means, mean errors, widths and width errors.
        fit (Minuit): Simultaneous fit with covariance.
        seed (int): Seed for 1000 multivariate-normal parameter draws.
    Returns
        Figure: Energy bias and relative resolution with one-sigma bands.
    '''
    means, mean_err, widths, width_err = sample
    grid = np.linspace(levels.min(), levels.max(), 200)
    values = np.asarray(fit.values)
    draws = np.random.default_rng(seed).multivariate_normal(values, fit.covariance, 1000)
    mean_curves = np.asarray([model_mean(grid, *p[:2]) for p in draws])
    width_curves = np.asarray([model_std(grid, *p[2:]) for p in draws])
    mean = model_mean(grid, *values[:2]) - grid
    relative_width = model_std(grid, *values[2:]) / grid
    mean_error = mean_curves.std(axis=0, ddof=1)
    width_error = width_curves.std(axis=0, ddof=1) / grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(levels, means - levels, yerr=mean_err, fmt='o', capsize=3,
                     label='Sample statistics')
    axes[1].errorbar(levels, widths / levels, yerr=width_err / levels,
                     fmt='o', capsize=3, label='Sample statistics')
    for ax, centre, error in zip(axes, (mean, relative_width), (mean_error, width_error)):
        ax.plot(grid, centre, color='C1', label='Simultaneous MLE')
        ax.fill_between(grid, centre - error, centre + error, color='C1', alpha=0.2,
                        label='Covariance propagation (1 SD)')
        ax.set_xlabel(r'$E_0$ [GeV]')
        ax.legend(fontsize=9)
    axes[0].set_ylabel(r'$\mu(E_0)-E_0$ [GeV]')
    axes[1].set_ylabel(r'$\sigma(E_0)/E_0$')
    fig.tight_layout()
    return fig


def plot_estimates(results: dict, bootstrap: dict | None = None) -> plt.Figure:
    '''Compare parameter estimates and errors relative to the joint-fit value.
    Args
        results (dict): Values and HESSE errors for each method.
        bootstrap (dict or None): Bootstrap values and standard deviations.
    Returns
        Figure: Five parameter comparisons on a common relative scale.
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    reference = np.array([results['simultaneous_fit']['values'][p] for p in PARAMETERS])
    for i, method in enumerate(METHODS):
        y = np.arange(5) + (i - 1) * 0.23
        value = np.array([results[method]['values'][p] for p in PARAMETERS])
        error = np.array([results[method]['errors'][p] for p in PARAMETERS])
        ax.errorbar(value / reference, y, xerr=error / np.abs(reference),
                     fmt='o', capsize=3, color=f'C{i}', label=LABELS[i])
        if bootstrap is not None:
            value = np.array([bootstrap[method]['values'][p] for p in PARAMETERS])
            error = np.array([bootstrap[method]['errors'][p] for p in PARAMETERS])
            ax.errorbar(value / reference, y + 0.07, xerr=error / np.abs(reference),
                         fmt='s', mfc='none', capsize=3, color=f'C{i}',
                         label=f'{LABELS[i]}: bootstrap')
    ax.axvline(1, color='0.5', linestyle='--', lw=1)
    ax.set(yticks=np.arange(5), yticklabels=PARAMETER_LABELS,
           xlabel='Estimate / simultaneous MLE estimate')
    ax.legend(fontsize=9, ncol=2 if bootstrap is not None else 1)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    return fig


def plot_bootstrap(samples: dict, results: dict) -> plt.Figure:
    '''Show bootstrap distributions and point estimates for all parameters.
    Args
        samples (dict): Arrays with shape (n_bootstrap, 5), one per method.
        results (dict): Fitted point estimates for each method.
    Returns
        Figure: Five parameter distributions with point estimates as dashed lines.
    '''
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, (parameter, label) in enumerate(zip(PARAMETERS, PARAMETER_LABELS)):
        ax = axes.flat[j]
        for i, method in enumerate(METHODS):
            ax.hist(samples[method][:, j], bins=40, density=True, histtype='step',
                     color=f'C{i}', label=LABELS[i])
            ax.axvline(results[method]['values'][parameter], color=f'C{i}',
                        linestyle='--', alpha=0.6, lw=1)
        ax.set(xlabel=label, ylabel='Density')
    axes.flat[0].legend(fontsize=9)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    return fig
