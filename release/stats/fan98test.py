import numpy as np


def fan98test(data_points1, data_points2, plot_steps=False, method='legendre'):
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

    if method == 'legendre':
        z_star = legendre_transform(z_score)
    elif method == 'fft':
        z_star = fourier_transform(z_score)
    else:
        raise TypeError('Unknown method type')


    # eval statistic for each number of first blocks (see fomula 6)
    T_an = np.zeros(len(z_star))
    for m in range(len(z_star)):
        T_an[m] = np.sum(z_star[:m+1]**2 - 1)*((d - 2)**2 * (d - 4) / (2 * (m+1) * d**2 * (d - 1)))**0.5


    # find maximum T (see formula 6)
    stat = np.max(T_an)



    # compute final stat(see fomula 4)
    loglogn = np.log(np.log(len(z_score)))


    stat = (2 * loglogn)**0.5 * stat - (2 * loglogn + 0.5 * np.log(loglogn) - 0.5 * np.log(4 *np.pi))

    # chi2 statistic benchmark(see formula 2)
    chi2stat = np.sum(z_score**2)

    return stat, chi2stat


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