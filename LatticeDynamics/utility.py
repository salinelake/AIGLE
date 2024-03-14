import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy as np
import torch as th
from scipy.ndimage import gaussian_filter1d, convolve1d

mpl.rcParams['axes.linewidth'] = 2.0 #set the value globally
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['lines.linestyle'] = 'dashed'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.marker'] = 'o'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.frameon'] = False
font = {'family': 'serif',
'color':  'k',
'weight': 'normal',
'size': 15,
        }
inset_font = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 10,
        }

kb = 8.617333262e-2 # meV/K
kjmol2mev=10.364
meV2KJpmol = 1/kjmol2mev
amu=0.1035 ## atomic mass unit in meV / (A/ps)^2
e2uC = 1.60217663e-13

# def correlation(x,y):
#     corr = ((x*y).mean() - x.mean()*y.mean()) / ((x*x).mean()*(y*y).mean())**0.5
#     return corr
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
def th2np(x):
    return x.detach().cpu().numpy()

def NormCov(x,y, axis=-1):
    corr = ((x*y).mean(axis) - x.mean(axis)*y.mean(axis)) / ((x*x).mean(axis)*(y*y).mean(axis))**0.5
    return corr

def NormCov_xyz(x,y, z, axis=-1):
    corr = ((x*y).mean(axis) - x.mean(axis)*y.mean(axis)) / ((x*x).mean(axis)*(z*z).mean(axis))**0.5
    return corr

def Cov(x,y, axis=-1):
    corr = ((x*y).mean(axis) - x.mean(axis)*y.mean(axis)) 
    return corr

def Corr_t(x,y, l ):
    """"x(0)y(t)"""
    corr = [  (x*y).mean( )] + [ (x[...,:-iT] * y[...,iT:]).mean( )  for iT in range(1, l)]
    if type(x) == np.ndarray:
        return np.array(corr)
    else:
        corr = th.stack(corr)
        return th2np(corr)

def MultiCorr_t(x,y, l, axis=[-1] ):
    """"x(0)y(t)"""
    corr = [  (x*y).mean(axis)] + [ (x[...,:-iT] * y[...,iT:]).mean(axis)  for iT in range(1, l)]
    if type(x) == np.ndarray:
        return np.array(corr)
    else:
        corr = th.stack(corr)
        return th2np(corr)
        
def AutoCorr_t(x, l ):
    '''
    Args:
        x: array of shape (*, nframes)
        l: maximal lag time
        # axis: axes to be averaged over
    Returns:
        corr: numpy array of (l, **)
    '''
    """"x(0)x(t)"""
    corr = [  (x*x).mean( )] + [ (x[...,:-iT] * x[...,iT:]).mean( )  for iT in range(1, l)]
    if type(x) == np.ndarray:
        return np.array(corr)
    else:
        corr = th.stack(corr)
        return th2np(corr)

def nabla(x):
    nablax = np.roll(x,1,axis=0) + np.roll(x,-1,axis=0) + np.roll(x,1,axis=1) + np.roll(x,-1,axis=1) - 4 * x 
    return nablax/4


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2
    
# def moving_average_block(a, n=3, axis=0):
#     if axis != 0:
#         raise NotImplementedError
#     ret = np.cumsum(a, axis=0)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

# def moving_average_gaussian(a, sigma=25, axis=0, truncate=3.0):
#     ret = gaussian_filter1d(a, sigma, axis=axis, mode='reflect',  truncate=truncate)
#     throw = int(np.ceil(sigma * truncate))
#     return ret[throw:-throw]


# def moving_average_triangle(a, span=25, axis=-1):
#     fsize = span
#     weights = [ 1 - x/fsize for x in range(fsize) ]
#     throw = fsize//2 + 1
#     weights = np.array(weights)
#     weights = weights / weights.sum()
#     ret = convolve1d(a, weights, axis=axis, origin=0 )
#     return ret[:,throw:-throw]

# def moving_average_uniform(a, span=25, axis=-1):
#     throw = span//2 + 1
#     weights = np.ones(span)/span
#     ret = convolve1d(a, weights, axis=axis, origin=0 )
#     return ret[:,throw:-throw]
 
def moving_average_half_gaussian(a, sigma=25, axis=-1, truncate=3.0):
    fsize = int(truncate * np.ceil(sigma))
    weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
    throw = fsize//2 + 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    ret = convolve1d(a, weights, axis=axis, origin=1 )
    return ret[..., throw:-throw]

def moving_average_half_gaussian_torch(a, sigma=25, axis=-1, truncate=3.0):
    fsize = int(truncate * np.ceil(sigma))
    weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
    throw = fsize//2 + 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    dev = 'cuda' if th.cuda.is_available() else 'cpu'
    torch_a = th.tensor(a, dtype=th.float32, device=dev)
    torch_w = th.tensor(weights, dtype=th.float32, device=dev)[None,None,:]
    ret = th.nn.functional.conv1d(torch_a, torch_w,  padding='valid')
    return th2np(ret)

def read_out_smooth(rtraj, stride=1,  le_dt=0.002,  smooth= 1, throw=0, block=1):
    '''
    Args:
        rtraj: numpy.array = (#frames, #nx, #ny, #nz, 3)
        stride: downsampling stride
        le_dt: timestep of coarse-grained trajectory, equals MD timestep X stride
        smooth: window size of the Half Gaussian filter
    Returns:
        r: numpt.array = (#nx, #ny, #nz, 3, #cg_frames)
    '''
    rdata = rtraj[throw:]     ## (   (#frames, #nx, #ny, #nz, 3)
 
    rdata =  rdata.transpose(1,2,3, 4, 0)  ##   (#nx, #ny, #nz, 3, #frames)
    nx,ny,nz,ndim,nframe = rdata.shape
    # if smooth > 1:
    #     # print('smooth window:', smooth)
    #     rdata = moving_average_half_gaussian(rdata, smooth)
    if smooth > 1:
        # print('smooth window:', smooth)
        rdata = rdata.reshape(-1,1,nframe)
        rdata = moving_average_half_gaussian_torch(rdata, smooth)
        rdata = rdata.reshape(nx,ny,nz,ndim,-1)
    rdata = rdata[..., ::stride]
    r =  rdata[..., 1:-1]
    v =  (rdata[..., 1:-1] - rdata[..., :-2]) / (le_dt)  #  retarded velocity
    a =  (rdata[...,   2:] + rdata[..., :-2] - 2* rdata[..., 1:-1]) / (le_dt )**2
    return  r,v,a

def fwhm(x, y):
    max_value = np.max(y)
    max_position = x[np.argmax(y)]
    half_max = max_value / 2
    left_idx = np.argmax(y >= half_max)
    right_idx = np.argmax(y >= half_max) + np.argmax(y[np.argmax(y >= half_max):] < half_max)
    fwhm = x[right_idx] - x[left_idx]
    return fwhm
if __name__ == "__main__":
    # processing
    # elist = np.array([0.0, 1.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, -1.0, -2.0, -2.2, -2.4, -2.6, -2.8, -3.0])
    # e_save = []
    # nseeds = 4
    # cv = []
    # with open('./data/cv_final.npy', 'wb') as f:
    #     for idx, efield in enumerate(elist):
    #         for ns in range(nseeds): 
    #             folder = './md_data/40X20X20/data/E{:.2f}_new/00{:d}'.format(efield, ns)
    #             colvar_dir = os.path.join(folder, 'COLVAR')
    #             colvar = np.loadtxt( colvar_dir )
    #             np.save(f, colvar[:,[7]])
    #             e_save.append(efield)
    # np.save('data/e_final.npy',e_save)
    #### load and plot
    dataset = []
    dt = 0.002
    # elist = np.load('data/e_100ps_all.npy')
    elist = np.load('data/e_final.npy')
    # with open('./data/cv_100ps_all.npy', 'rb') as f:
    with open('./data/cv_final.npy', 'rb') as f:
        for idx, efield in enumerate(elist):
            dataset.append(np.load(f,allow_pickle=True))
    fig, ax = plt.subplots(figsize=(12,6))
    for idx, efield in enumerate(elist):
        traj = (dataset[idx]-dataset[idx][0]+1)*20
        ax.plot(np.arange(dataset[idx].shape[0])*dt, traj, label='E={:.2f}'.format(efield),markersize=0)
        print('E={}, mean={}, std={}'.format(efield, traj[-1]- traj[0], traj.std()))
    ax.legend()
    # plt.savefig('cv100ps_all.png')
    plt.savefig('cv_final.png')

