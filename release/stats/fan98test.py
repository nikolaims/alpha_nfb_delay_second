import numpy as np
import os

def fan98test(data_points1, data_points2, plot_steps=False, transform='legendre'):
    # mean, std and number of samples
    x1_mean = np.mean(data_points1, 0)
    x1_std = np.std(data_points1, 0)
    x1_n = data_points1.shape[0]
    x2_mean = np.mean(data_points2, 0)
    x2_std = np.std(data_points2, 0)
    x2_n = data_points2.shape[0]

    # degree of freedom(see page 1011, left top paragraph)
    d = x1_n + x2_n - 2

    # z - score over time(see formula 13)
    z_score = (x1_mean - x2_mean) / (x1_std ** 2 / x1_n + x2_std ** 2 / x2_n) ** 0.5

    stat = eval_fan98_stat(z_score, d, transform)
    # eval statistic for each number of first blocks (see fomula 6)

    # chi2 statistic benchmark(see formula 2)
    # chi2stat = np.sum(z_score**2)

    return stat


def eval_fan98_stat(z_star, d, transform):
    if transform == 'legendre':
        z_star = legendre_transform(z_star)
    elif transform == 'fft':
        z_star = fourier_transform(z_star)
    elif transform is None or transform=='identity':
        pass
    else:
        raise TypeError('Unknown method type')

    T_an = np.zeros(len(z_star))
    d_factor = 1 if d is None else ((d - 2) ** 2 * (d - 4) / (d ** 2 * (d - 1))) ** 0.5
    for m in range(len(z_star)):
        T_an[m] = np.sum(z_star[:m + 1] ** 2 - 1) * d_factor / (2 * (m + 1)) ** 0.5

    # find maximum T (see formula 6)
    stat = np.max(T_an)

    # compute final stat(see fomula 4)
    loglogn = np.log(np.log(len(z_star)))

    stat = (2 * loglogn) ** 0.5 * stat - (2 * loglogn + 0.5 * np.log(loglogn) - 0.5 * np.log(4 * np.pi))
    return stat

def fourier_transform(x):
    # fft (see formula between 17 and 18)
    z_fft = np.fft.fft(x) / len(x) ** 0.5
    # colect real and imag coeffs(see example 1, page 1013, 2nd paragraph)
    z_star = np.zeros(len(z_fft) * 2 - 1)
    z_star[0::2] = np.real(z_fft)
    z_star[1::2] = np.imag(z_fft[1:])
    return z_star[:len(z_fft)]


def legendre_transform(x):
    n = len(x)
    a = np.arange(n)
    basis = np.zeros((n, n))
    for k in a:
        basis[:, k] = (a - np.mean(a))**k
    q, _ = np.linalg.qr(basis)
    return x.dot(q)


def simulate_h0_distribution(n, d=None, n_iter=100000, transform='legendre'):
    cash_dir = '_fan98_temp'
    cash_file = os.path.join(cash_dir, 'h0_n{}_d{}_n_iter{}_{}.npy'.format(n, d, n_iter, transform))
    if os.path.exists(cash_file):
        print('Load from {}'.format(cash_file))
        stats_h0 = np.load(cash_file)
    else:
        print('Simulate and save to {}'.format(cash_file))
        stats_h0 = np.zeros(n_iter)
        for k in range(n_iter):
            if d is None:
                z_star = np.random.normal(size=n)
            else:
                z_star = np.random.standard_t(d, size=n)
            stats_h0[k] = eval_fan98_stat(z_star, d, transform)
        if not os.path.exists(cash_dir):
            os.makedirs(cash_dir)
        np.save(cash_file, stats_h0)
    return stats_h0


def get_p_val(val, h0_distribution):
    p_val = np.sum(val < h0_distribution)/h0_distribution.shape[0]
    return p_val

if __name__ == '__main__':
    n = 20

    import pandas as pd
    paper_table = pd.read_csv('release/stats/upperquartile.csv', sep=';').values
    p_vals = paper_table[0, 1:]
    stats = paper_table[paper_table[:, 0] == n, 1:][0]


    d = None
    stats_h0 = simulate_h0_distribution(n, d, transform='legendre')


    levels = np.zeros_like(p_vals)
    for k, p_val in enumerate(p_vals):
        levels[k] = np.quantile(stats_h0, 1 - p_val)

    print(' '.join(['{:.2f}'.format(p*100) for p in p_vals]))
    print(' '.join(['{:.2f}'.format(level) for level in levels]))
    print(' '.join(['{:.2f}'.format(level) for level in stats]))
    print(' '.join(['{:.2f}'.format(level) for level in (stats-levels)/stats*100]))

