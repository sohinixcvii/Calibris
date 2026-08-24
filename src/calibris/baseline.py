import numpy as np


def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    """Squared exponential kernel — encodes 'nearby x values should have
    similar signal values', which is what makes the GP produce a smooth
    curve rather than fitting every noisy point exactly."""
    sqdist = (x1[:, None] - x2[None, :]) ** 2
    return variance * np.exp(-0.5 * sqdist / lengthscale**2)


def naive_gp_baseline(x, observed, sigma, lengthscale=1.0, variance=1.0):
    """
    GP regression assuming i.i.d. (white) noise. This is the 'naive'
    baseline: it ignores any correlation structure in the noise, even
    though the true noise here is AR(1)-correlated.
    """
    n = len(x)
    K = rbf_kernel(x, x, lengthscale, variance)
    noise_cov = sigma**2 * np.eye(n)          # <- the naive assumption
    K_plus_noise = K + noise_cov

    # np.linalg.solve instead of explicitly inverting — standard practice,
    # more numerically stable than computing K_plus_noise^{-1} directly
    alpha = np.linalg.solve(K_plus_noise, observed)
    posterior_mean = K @ alpha

    K_inv_K = np.linalg.solve(K_plus_noise, K)
    posterior_cov = K - K @ K_inv_K
    posterior_std = np.sqrt(np.diag(posterior_cov))

    return posterior_mean, posterior_std, posterior_cov
