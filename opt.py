from scipy.optimize import minimize, LinearConstraint
from scipy.stats import linregress
import numpy as np
import pylab as plt


k = np.arange(15)
snr = 0.5
beta = 1.
eta = beta/snr
a = -1/16
noise = np.random.normal(size=len(k))*0.5


def env(k, eta, beta, a):
    y = eta + beta*(a*k+1)
    return y

y = env(k, eta, beta, a) + noise
plt.plot(y, label='raw')
plt.plot(y-noise, label='a = {:.3f}'.format(a))


def fun(x):
    eta, beta, a = x
    return np.sum((eta + beta*(a*k+1) - y)**2)

def jac(x):
    eta, beta, a = x
    d_eta = np.sum(2*(eta + beta*(a*k+1) - y))
    d_beta = np.sum(2 * (eta + beta * (a * k + 1) - y) * (a*k+1))
    d_a = np.sum(2 * (eta + beta * (a * k + 1) - y) * beta * k)
    return np.array([d_eta, d_beta, d_a])

def hess(x):
    eta, beta, a = x
    d_eta2 = 2*(max(k)+1)
    d_beta2 = np.sum(2*(a*k+1)**2)
    d_a2 = np.sum(2 * (beta * k) ** 2)
    d_eta_beta = np.sum(2*(a*k+1))
    d_eta_a = np.sum(2 * beta * k)
    d_beta_a = np.sum(2*k*(eta+2*beta*(a*k+1) - y))
    return np.array([[d_eta2, d_eta_beta, d_eta_a], [d_eta_beta, d_beta2, d_beta_a], [d_eta_a, d_beta_a, d_a2]])

bounds = LinearConstraint(np.eye(3), [0,0,-1/len(k)], np.inf)
res = minimize(fun, np.array([1,1,0]), jac=jac, hess=hess, method='trust-constr', constraints=bounds)

lin_reg = linregress(k, y)
plt.plot(lin_reg.intercept + lin_reg.slope * k, label='a-lr = {:.3f}'.format(lin_reg.slope))

plt.plot(env(k, *res.x), label='a-c = {:.3f}'.format(res.x[2]))
plt.legend()
a, res.x[2], lin_reg.slope



