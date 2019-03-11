from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def kronecker_delta(x_1, x_2):
    """
    Parameters
    ----------
        x_1, x_2: floats (scalars)
    
    Returns
    ----------
        delta: 0 or 1
            .. math:: \\delta(x_1, x_2)
    """
    if x_1 == x_2:
        delta = 1
    else:
        delta = 0
    return delta

def matern_52(x, hyperparams):
    """
    Parameters
    ----------
        x: float (vector)
            Vector of points

        hyperparams: float (tuple)
            Hyperparameters for a Gaussian process

    Returns
    ----------
        C: float (matrix)
            Returns a Matern (5,2) square covariance matrix of size(x)
            .. math:: C_{5,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{5} d / b + (5/3) (d/b)^2 ] e^{-\\sqrt{5} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
    """

    # Unpack hypereparameters
    b, tau_1_squared, tau_2_squared = hyperparams

    # Initialize covariance matrix
    C = np.zeros((x.shape[0], x.shape[0]))

    # Evaluate (i,j) components of covariance matrix
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            d = np.abs(x[i] - x[j])
            C[i][j] = tau_1_squared*(1 + np.sqrt(5)*(d/b) + (5/3)*(d/b)**2)*np.exp(-np.sqrt(5)*(d/b)) + tau_2_squared*kronecker_delta(x[i], x[j])

    return C

# Initialize random seed
np.random.seed(3)

# Randomly draw values from a uniform distribution and sort from lowest to highest
x = np.sort(np.random.uniform(size=100))

# Create hyperparameters
b = np.logspace(-1, 1, num=4)
tau_1_squared = np.logspace(-1, 0, num=4)
tau_2_squared = np.logspace(-6, 0, num=4)

#### Plot varied b
plt.figure()
plt.title('$\\tau_1^2$={:.2f}'.format(tau_1_squared[0])+'; $\\tau_2^2$={:.6f}'.format(tau_2_squared[0]))
# Iterates over vector of b values and plots them on the same plot
for i in range(len(b)):
	hyperparams = b[i], tau_1_squared[0], tau_2_squared[0]
    # Calculates covariance given x and current hyperparameters
	cov = matern_52(x, hyperparams)
    # Generates random sample from a multivariate normal
	fx = multivariate_normal.rvs(mean=np.zeros(x.shape[0]), cov=cov)
	plt.plot(x, fx, label='b={:.2f}'.format(b[i]))
plt.legend(loc=0)
plt.ylim([-3, 3])
# plt.show()
plt.savefig('figures/matern_52_varied_b.pdf')
plt.close()


#### Plot varied tau_1^2
plt.figure()
plt.title('b={:.2f}'.format(b[i])+'; $\\tau_2^2$={:.6f}'.format(tau_2_squared[0]))
# Iterates over vector of tau_1^2 values and plots them on the same plot
for i in range(len(tau_1_squared)):
    hyperparams = b[0], tau_1_squared[i], tau_2_squared[0]
    # Calculates covariance given x and current hyperparameters
    cov = matern_52(x, hyperparams)
    # Generates random sample from a multivariate normal
    fx = multivariate_normal.rvs(mean=np.zeros(x.shape[0]), cov=cov)
    plt.plot(x, fx, label='$\\tau_1^2$={:.2f}'.format(tau_1_squared[i]))
plt.legend(loc=0)
plt.ylim([-3, 3])
# plt.show()
plt.savefig('figures/matern_52_varied_tau_1.pdf')
plt.close()

#### Plot varied tau_2^2
plt.figure()
plt.title('b={:.2f}'.format(b[i])+'; $\\tau_1^2$={:.2f}'.format(tau_1_squared[0]))
# Iterates over vector of tau_2^2 values and plots them on the same plot
for i in range(len(tau_1_squared)):
    hyperparams = b[0], tau_1_squared[0], tau_2_squared[i]
    # Calculates covariance given x and current hyperparameters
    cov = matern_52(x, hyperparams)
    # Generates random sample from a multivariate normal
    fx = multivariate_normal.rvs(mean=np.zeros(x.shape[0]), cov=cov)
    plt.plot(x, fx, label='$\\tau_2^2$={:.6f}'.format(tau_2_squared[i]))
plt.legend(loc=0)
plt.ylim([-3, 3])
# plt.show()
plt.savefig('figures/matern_52_varied_tau_2.pdf')
plt.close()

##### Plot b vs tau_1^2
fig, ax = plt.subplots(b.shape[0], tau_1_squared.shape[0], sharex='col', sharey='row')
fig.subplots_adjust(hspace=0.3, wspace=0.2)
fig.suptitle('$\\tau_2^2$={:.6f}'.format(tau_2_squared[0]))
fig.set_size_inches(8,4.5)
for i in range(b.shape[0]):
    for j in range(tau_1_squared.shape[0]):
        ax[i, j].set_title('b={:.2f}'.format(b[i]) + '; $\\tau_1^2$={:.2f}'.format(tau_1_squared[j]), fontsize=8)
        hyperparams = b[i], tau_1_squared[j], tau_2_squared[0]
        # Calculates covariance given x and current hyperparameters
        cov = matern_52(x, hyperparams)
        # Generates random sample from a multivariate normal
        fx = multivariate_normal.rvs(mean=np.zeros(x.shape[0]), cov=cov)
        ax[i, j].plot(x, fx, '-k')
# plt.show()
plt.savefig('figures/matern_52_b_vs_tau_1.pdf')