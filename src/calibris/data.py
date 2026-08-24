import numpy as np

def make_latent_signal(n_points=200, x_min=0.0, x_max=10.0):
    x = np.linspace(x_min, x_max, n_points)
    signal = np.sin(x) + 0.3 * np.sin(3 * x)
    return x, signal

def make_ar1_noise(n_points, phi=0.8, sigma=0.3, rng=None):
    rng = np.random.default_rng(rng)
    noise = np.zeros(n_points)
    innovation_std = sigma * np.sqrt(1 - phi**2)
    noise[0] = rng.normal(0, sigma)
    for t in range(1, n_points):
        noise[t] = phi * noise[t - 1] + rng.normal(0, innovation_std)
    return noise

def generate_dataset(n_points=200, phi=0.8, sigma=0.3, seed=None):
    x, signal = make_latent_signal(n_points)
    noise = make_ar1_noise(n_points, phi=phi, sigma=sigma, rng=seed)
    observed = signal + noise
    return {
        "x": x,
        "true_signal": signal,
        "observed": observed,
        "phi": phi,
        "sigma": sigma,
    }