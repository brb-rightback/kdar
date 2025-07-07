import numpy as np
import math
from scipy.linalg import cholesky
from numpy.linalg import inv

def get_bin_centers(x):
    centers = []
    for i in range(len(x)-1): centers.append( x[i] + (x[i+1]-x[i])/2 )
    return centers
    
def get_angle(momentum_0,momentum_1,momentum_2,to_numi=False):
    momentum_0_new = momentum_0
    momentum_1_new = momentum_1
    momentum_2_new = momentum_2    
    if to_numi:
        R = np.array([[0.9210,0.0227,0.3889], 
        [0.00005,0.9983,-0.0584],
        [-0.3895,0.05383,0.9195]])
        R = inv(R)
        momentum_0_new = momentum_0*R[0][0] + momentum_1*R[0][1] + momentum_2*R[0][2] 
        momentum_1_new = momentum_0*R[1][0] + momentum_1*R[1][1] + momentum_2*R[1][2]
        momentum_2_new = momentum_0*R[2][0] + momentum_1*R[2][1] + momentum_2*R[2][2]
    
    momentum_perp = np.sqrt(momentum_0_new * momentum_0_new + momentum_1_new * momentum_1_new)
    # defined in https://root.cern/doc/master/TVector3_8cxx_source.html
    theta = np.arctan2(momentum_perp, momentum_2_new)
    phi = np.arctan2(momentum_1_new, momentum_0_new)
    return theta, phi

def get_x_err_Enu(x, x_bins):
    x_err = np.zeros_like([x,x])
    for i in range(len(x)):
            x_err[0][i] = x[i] - x_bins[i]
            x_err[1][i] = x_bins[i+1] - x[i]
    return x_err

def wilson(n_pass, n, z = 1.96):
    if n==0: return 0,0    
    p = n_pass/n
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    if lower_bound<0: lower_bound=0
    if upper_bound<0: upper_bound=0
    return lower_bound, upper_bound

def wilson_err(n_pass, n):
    y_err = [[],[]]
    for i in range(len(n_pass)):
        if n[i] == 0:
            y_err[0].append(0)
            y_err[1].append(0)
            continue
        y_low,y_high = wilson(n_pass[i],n[i])
        y = n_pass[i]/n[i]
        low_err = y-y_low
        high_err = y_high-y
        if low_err<0: low_err=0
        if high_err<0: high_err=0
        y_err[0].append(low_err)
        y_err[1].append(high_err)
    return y_err

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def calc_chi2(pred, data, cov, pre_inverse=False):
    if not is_invertible(cov): return 0
    M = np.matrix([w-x for w,x in zip(pred, data)])
    Mt = M.transpose()
    cov_inv = cov
    if pre_inverse==False: cov_inv = np.linalg.inv(np.matrix(cov))
    Mret = np.matmul(M, np.matmul(cov_inv, Mt))
    return Mret[0,0]

def gaus(x,a,x0,sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def gaus_noC(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def gaus_sum(x,a_1,x0_1,sigma_1,a_2,sigma_2):
    return a_1*np.exp(-(x-x0_1)**2/(2*sigma_1**2))+a_2*np.exp(-(x-x0_1)**2/(2*sigma_2**2))