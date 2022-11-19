import numpy as np
from scipy.ndimage import convolve1d
  
def moving_average_half_gaussian(a, sigma=25, axis=-1, truncate=3.0):
    fsize = int(truncate * np.ceil(sigma))
    weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
    throw = fsize//2 + 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    ret = convolve1d(a, weights, axis=axis, origin=1 )
    return ret[:,throw:-throw]
 
def read_out_smooth(rtraj, stride=1,  le_dt=0.002,  smooth= 1, throw=0):
    rdata = (rtraj[None,throw:,0]+1)*20 
    if smooth > 1:
        rdata = moving_average_half_gaussian(rdata,smooth)
    else:
        rdata = rdata
    rdata = rdata[:,::stride]
    r = rdata[:,1:-1]
    v  =  (rdata[:,1:-1] - rdata[:,:-2]) / (le_dt)  #  retarded velocity
    a =  (rdata[:,2:] + rdata[:,:-2] - 2* rdata[:,1:-1]) / (le_dt )**2
    return  r,v,a
 