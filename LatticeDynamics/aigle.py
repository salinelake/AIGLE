
import os
from  time import time
import logging 
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
import torch as th
from torch.optim.lr_scheduler import StepLR

from model import *
from utility import *
 
class aigle:
    def __init__(self, temp, dt=0.01, len_ag=50, lmem=100, nskip=1, model_ext=None, model_noise=None):
        ### system parameters
        self.dev = 'cuda' if th.cuda.is_available() else 'cpu'
        kb = 8.617333262e-2 # meV/K
        self.temp = temp
        self.beta = 1 / kb / temp
        self.dt = dt ## the time step of the data
        self.len_ag = len_ag
        self.lmem = lmem
        self.nskip = nskip
        # self.amax = 0
        if model_ext is None:
            self.model_ext = poly_3Dsystem(lmem=lmem).to(self.dev)
        else:
            self.model_ext = model_ext.to(self.dev)
        if model_noise is None:
            self.model_noise = FNN_noise( len_ag ).to(self.dev)
        else:
            self.model_noise = model_noise.to(self.dev)
        ### trajectory of simulation
        self.r_list = None
        self.v_list = None
        self.n_list = None
        self.n_buffer = None
        ##  tensorboard
        self.writer = None
        ## buffer
        self.kosl_fdt = None
        self.knnl_fdt = None
        self.kost_fdt = None
        self.knnt_fdt = None

        self.kosl_infer = None
        self.knnl_infer = None
        self.kost_infer = None
        self.knnt_infer = None

        self.corr_dict_x = None
        self.corr_dict_y = None
        self.corr_dict_z = None
        self.v2_avg = None  ## with respect to component of v ->  k_b T/m,

    def save_model(self, model_folder, label='0'):
        th.save(self.model_ext.state_dict(), os.path.join(model_folder,'model_ext.{}.ckpt'.format(label)))
        th.save(self.model_noise.state_dict(), os.path.join(model_folder,'model_noise.{}.ckpt'.format(label)))
        logging.info('Force model saved to {}/model_ext.{}.ckpt'.format(model_folder, label))
        logging.info('Noise model saved to {}/model_noise.{}.ckpt'.format(model_folder, label))
        return 

    def load_model(self, model_folder, label='0', load_noise=True):
        self.model_ext.load_state_dict(th.load(
            os.path.join(model_folder, 'model_ext.{}.ckpt'.format(label)),
            map_location=self.dev))
        if load_noise:
            self.model_noise.load_state_dict(th.load(
                os.path.join(model_folder, 'model_noise.{}.ckpt'.format(label)),
                map_location=self.dev))
        return
    ##########################    Training    ##########################
    def get_noise( self, r, noise ):
        '''
        Transform a continuous trajectory to the input. Then get predicted noise 
        Args:
            r: Tensor of shape (nx, ny, nz, 3, nbatch) 
            noise: Tensor of shape (nx, ny, nz, 3, nbatch) 
        '''
        lag = self.len_ag
        noise_ref = noise[..., lag:] ## (*,nframes-lag )
        noise_traj = []
        for i in range(1,lag+1):
            noise_traj.append(noise[...,lag-i:-i,None])
        noise_traj = th.cat(noise_traj,-1) ##   (*,nframes-lag, lag)
        noise_pred, noise_sigma = self.model_noise( r[..., lag:], noise_traj )  ##  (*,nframes-lag)
        assert noise_pred.shape == noise_ref.shape
        return noise_pred, noise_ref, noise_sigma
    
    def get_qx_target(self, _x, _v, _m,  kos, knn, e1=2, axis=2 ):
        lmem = self.lmem
        # kos =  self.model_ext.kos 
        # knn =  self.model_ext.knn 

        _xp = th.roll(_x, 1, axis)
        corr_vx  = Corr_t(  _x[_m][...,:-1], _v[_m][...,1:], lmem)    ## Corr[ x(0), v(t+0.5)]   t=0..lmem
        corr_vxp = Corr_t( _xp[_m][...,:-1], _v[_m][...,1:], lmem)    ## Corr[ x'(0), v(t+0.5)]   t=0..lmem

        C = np.zeros((lmem, lmem))   ## C[ii,jj]=Corr[v(ii+0.5-jj),v(0)]
        Cp = np.zeros((lmem, lmem))
        for ii in range(lmem):
            for jj in range(ii+1):
                C[ii,jj] = corr_vx[ ii-jj ]
                Cp[ii,jj] = corr_vxp[ ii-jj ]
        qx_target_os = C@kos + e1 * Cp @ knn  # target of  (<q(1),x(0)>,...,<q(lmem),x(0)>)
        qx_target_os = np.roll(qx_target_os, 1, 0)
        qx_target_os[0] *= 0   # target of  (<q(0),x(0)>,...,<q(lmem-1),x(0)>)
        qx_target_nn = Cp @ kos + C @ knn
        qx_target_nn = np.roll(qx_target_nn, 1, 0)
        qx_target_nn[0] *= 0   # target of  (<q(0),x'(0)>,...,<q(lmem-1),x'(0)>)

        return qx_target_os, qx_target_nn

    def get_all_correlation(self, _n, _w, _vr, _vi, _q=None, _m1=None, _m2=None, roll_axis=-1):
        lmem =self.lmem
        _npl = th.roll( _n, 1, roll_axis)                                     ##  n'(t):    (nx, ny, nz, nframes)
        _npt = th.roll( _n, 1, roll_axis-1)                                     ##  n'(t):    (nx, ny, nz, nframes)
        _npll = th.roll( _n, 2, roll_axis)                                    ##  n"(t):    (nx, ny, nz, nframes)
        _nplt  = th.roll( th.roll(_n,  1, roll_axis-1), 1, roll_axis)                            ##  n'''(t):  (nx, ny, nz, nframes)
        _vipl = th.roll(_vi, 1, roll_axis)                                    ##  v'(t):    (nx, ny, nz, nframes)
        _vipt = th.roll(_vi, 1, roll_axis-1)                                    ##  v'(t):    (nx, ny, nz, nframes)
        _vipll = th.roll(_vi, 2, roll_axis)                                   ##  v"(t):    (nx, ny, nz, nframes)

        if _q is None:
            _q = th.zeros_like(_vi).to( device=self.dev)
        if _m1 is None:
            _m1 = th.ones_like(_w).to(dtype=th.bool, device=self.dev)
        if _m2 is None:
            _m2 = th.ones_like(_w).to(dtype=th.bool, device=self.dev)
        ## compute correlation; shape=(lmem,3)
        corr_ww = AutoCorr_t( _w[_m2], lmem)     ## Corr[  w(0), w(t)]  t=0..lmem-1 
        corr_nn = AutoCorr_t( _n[_m1], lmem+1)   ## Corr[ n(0), n(t)]
        corr_vv = AutoCorr_t(_vr[_m1], lmem)     ## Corr[ v(0), v(t)]
        corr_wv =     Corr_t(_vi[_m2][..., -_w.shape[-1]:], _w[_m2], lmem)     ## Corr[ v(0), w(t)]

        corr_nnpl  = Corr_t( _n[_m2], _npl[_m2], lmem+1)        ## Corr[ n(r,0), n(r+z,t)]
        corr_nnpt  = Corr_t( _n[_m2], _npt[_m2], lmem+1)       ## Corr[ n(r,0), n(r+x,t)]
        corr_nnpll = Corr_t( _n[_m2], _npll[_m2], lmem+1)     ## Corr[ n(r,0), n(r+z+z,t)]
        corr_nnplt = Corr_t( _n[_m2], _nplt[_m2], lmem+1)     ## Corr[ n(r,0), n(r+x+z,t)]

        corr_vvpl = Corr_t(_vipl[_m1], _vi[_m1], lmem)        ## Corr[ v(r+z,0), v(t)]
        
        _viplt = th.roll( th.roll(_vi, 1, 0), 1, 2)                          ##  v'(t):    (nx, ny, nz, nframes)
        corr_vvplt = Corr_t(_viplt[_m1], _vi[_m1], lmem)        ## Corr[ v(r+x+z,0), v(t)]
        _viplt = th.roll( th.roll(_vi, 1, 1), 1, 2)                          ##  v'(t):    (nx, ny, nz, nframes)
        corr_vvplt += Corr_t(_viplt[_m1], _vi[_m1], lmem)        ## Corr[ v(r+x+z,0), v(t)]
        _viplt = th.roll( th.roll(_vi, -1, 0), 1, 2)                          ##  v'(t):    (nx, ny, nz, nframes)
        corr_vvplt += Corr_t(_viplt[_m1], _vi[_m1], lmem)        ## Corr[ v(r+x+z,0), v(t)]
        _viplt = th.roll( th.roll(_vi, -1, 1), 1, 2)                          ##  v'(t):    (nx, ny, nz, nframes)
        corr_vvplt += Corr_t(_viplt[_m1], _vi[_m1], lmem)        ## Corr[ v(r+x+z,0), v(t)]
        corr_vvplt /= 4.0

        corr_vvpll = Corr_t(_vipll[_m1], _vi[_m1], lmem)

        corr_vrvi = Corr_t(_vi[_m1][...,:-1], _vr[_m1][...,1:], lmem)           ## Corr[ v(r,0), v(r,t+0.5)]
        corr_vrvpl = Corr_t(_vipl[_m1][...,:-1], _vr[_m1][...,1:], lmem)          ## Corr[ v(r+z,0), v(r,t+0.5)] between neastest neighbor
        corr_vrvpt = Corr_t(_vipt[_m1][...,:-1], _vr[_m1][...,1:], lmem)          ## Corr[ v(r+z,0), v(r,t+0.5)] between neastest neighbor
        corr_vrvplt = Corr_t(_viplt[_m1][...,:-1], _vr[_m1][...,1:], lmem)          ## Corr[ v(r+x+z,0), v(r,t+0.5)] between neastest neighbor
        
        corr_qv = Corr_t(_vi[_m1], _q[_m1], lmem+1)                    ## Corr[ v(r,0), q(r,t)]   t=0..lmem
        corr_qvpl = Corr_t(_vipl[_m1], _q[_m1], lmem+1)                  ## Corr[ v(r+z,0), q(r,t)]  t=0..lmem
        corr_qvpt = Corr_t(_vipt[_m1], _q[_m1], lmem+1)                  ## Corr[ v(r+x,0), q(r,t)]  t=0..lmem
        corr_qvplt = Corr_t(_viplt[_m2], _q[_m2], lmem+1)                ## Corr[ v(r+y+z,0),  q(r,t)]  t=0..lmem
        corr_qvpll = Corr_t(_vipll[_m2], _q[_m2], lmem+1)                ## Corr[ v(r+z+z,0),  q(r,t)]  t=0..lmem
         
        return {
            'corr_ww': corr_ww,
            'corr_nn': corr_nn,
            'corr_vv': corr_vv,
            'corr_nnpl': corr_nnpl,
            'corr_nnpt': corr_nnpt,
            'corr_nnpll': corr_nnpll,
            'corr_nnplt': corr_nnplt,
            'corr_vvpl': corr_vvpl,
            'corr_vvplt': corr_vvplt,
            'corr_vvpll': corr_vvpll,
            'corr_wv': corr_wv,
            'corr_vrvi': corr_vrvi,
            'corr_vrvpl': corr_vrvpl,
            'corr_vrvpt': corr_vrvpt,
            'corr_vrvplt': corr_vrvplt,

            'corr_qv': corr_qv,
            'corr_qvpl': corr_qvpl,
            'corr_qvpt': corr_qvpt,
            'corr_qvplt': corr_qvplt,
        }
     
    def get_simple_kernal(self, G,Gp,rho, rhop, e1):
        return  (- (G*rho + e1*Gp*rhop) / (G**2+e1*Gp**2) ) 

    def get_kernal_fdt(self, corr_dict,  e1=2 ):
        G = corr_dict['corr_vv'][0]
        Gp = corr_dict['corr_vvplt'][0]
        rho = corr_dict['corr_nn']
        rhop =  corr_dict['corr_nnplt']
        kos_fdt = self.get_simple_kernal( G,Gp,rho, rhop, e1)
        kos_fdt *= self.dt
        knn_fdt = np.zeros_like(kos_fdt)
        kos_fdt = (kos_fdt[:-1] + kos_fdt[1:]) / 2
        knn_fdt = (knn_fdt[:-1] + knn_fdt[1:]) / 2
        return kos_fdt, knn_fdt
    
    def get_kernal_av_2rd(self, corr_dict, alpha=0, e1=4, e2=4,  onsite_only=True, debug=False):
        '''
        memory acts on second nearest neighbor
        '''
        lmem = self.lmem
        C = np.zeros((lmem, lmem))   ## C[ii,jj]=Corr[v(ii+0.5-jj),v(0)]
        Cp = np.zeros((lmem, lmem))
        for ii in range(lmem):
            for jj in range(ii+1):
                C[ii,jj] = corr_dict['corr_vrvi'][ ii-jj ]
                Cp[ii,jj] = corr_dict['corr_vrvplt'][ ii-jj ]

        if onsite_only:
            B11 = C.T@C + e1*Cp.T@Cp
            Q = corr_dict['corr_qv'][1:]
            Qp = corr_dict['corr_qvplt'][1:]
            R1 = C.T@Q + e1*Cp.T@Qp
            lstsq_left = B11 + alpha * np.identity(lmem)
            lstsq_right = R1
            kernel = np.linalg.pinv(lstsq_left,  rcond=1e-5, hermitian=True) @ lstsq_right
            kos_numpy = kernel.flatten()
            knn_numpy = np.zeros_like(kos_numpy)
        else:
            raise NotImplementedError
 
        scf_error = ((lstsq_right-lstsq_left@kernel)**2).sum() / (lstsq_right**2).sum()
        scf_error = scf_error**0.5
        ### get <R(t),v(0)> for all important terms 
        corr_nv = corr_dict['corr_qv'].copy()
        corr_nv[1:] -= C @ kos_numpy # self.model_ext.kos
        corr_nv_lt = corr_dict['corr_qvplt'].copy()
        corr_nv_lt[1:] -= Cp @ kos_numpy # self.model_ext.knn
        corr_dict_nv={
            'corr_nv' : corr_nv,
            'corr_nv_lt' : corr_nv_lt,
        }
        return kos_numpy, knn_numpy, scf_error, corr_dict_nv

    def get_YW(self, corr_dict, alpha=0, onsite_only=True):
        corr_nn = corr_dict['corr_nn']
        ########### Get matrix for computing Yule-Walker ######
        ag_size = self.len_ag
        mat_crr = np.zeros((ag_size, ag_size))
        for ii in range(ag_size):
            for jj in range(ag_size):
                mat_crr[ii,jj] = corr_nn[np.abs(jj-ii)]       
        ########### Update GAR linear coefficient  ###########
        ag_coef, res, rank, singular =  np.linalg.lstsq( mat_crr, corr_nn[1:ag_size+1], rcond=1e-5 )
        # ag_coef_torch = th.tensor(ag_coef, dtype = th.float32, device = self.dev)
        ag_error = ((corr_nn[1:ag_size+1] - mat_crr @ ag_coef)**2).sum() / (corr_nn[1:ag_size+1]**2).sum()
        ag_error = ag_error**0.5
        ag_poly = poly([1]+ag_coef.tolist()) 
        agos_numpy = ag_coef
        # agos_numpy = np.zeros_like(ag_coef)
        agnn_numpy = np.zeros_like(ag_coef)
        if onsite_only is False:
            raise NotImplementedError
        return agos_numpy, agnn_numpy, ag_error, ag_poly

    def force_match( self, trainset, params, noneq=False ):
        model_folder = params['model_folder']
        model_ext = self.model_ext
        model_noise = self.model_noise
        dev = self.dev
        lmem = self.lmem
        optimizer = th.optim.Adam( list(model_ext.parameters()) , lr = params['lr'] )
        scheduler = StepLR(optimizer, step_size=params['lr_window'], gamma = params['lr_decay'])
        for i in range(params['epoch']):
            if len(trainset) == 1:
                setidx = 0
            else:
                # setidx = np.random.randint(len(trainset))
                setidx = np.random.choice(np.arange(1,len(trainset)))
            #################### LOADING ####################
            nx, ny, nz, _ , nframes =  trainset[setidx]['r'].shape
            length =  np.minimum(params['max_frames'], nframes)
            beg = 0
            end = beg + length
            bs = [ params['window_x'],  params['window_y'],params['window_z'] ] # blocksize
            xbeg = np.random.randint(nx - bs[0]+1)
            ybeg = np.random.randint(ny - bs[1]+1)
            zbeg = np.random.randint(nz - bs[2]+1)
            dataset={ #(#nx, #ny, #nz, 3, #frames)
                'r':   th.tensor(trainset[setidx]['r'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :,beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v':   th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'a':   th.tensor(trainset[setidx]['a'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'e':   th.tensor(trainset[setidx]['e'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v_more': th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+1:end
                    ], dtype=th.float32, device=dev),
            }
            ########################### FORCE ###########################
            potential_force, damp_force  = model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
            _q = dataset['a'] - potential_force      ##  a(t)-F(t):  (nx, ny, nz, 3, nframes)
            ########################### filter for excluding margin sites ###########################
            _m = th.zeros_like(dataset['r'][:,:,:,0,0]).to(dev)
            _m[self.nskip:-self.nskip, self.nskip:-self.nskip, self.nskip:-self.nskip] += 1    ### Hide the boundary
            _m = _m.to(dtype=bool)
            ########################### separate valley and transition state ###########################
            if noneq:
                rfilter = th.abs(dataset['r'][_m]) < 2
                loss = (_q[_m][rfilter]**2).mean() + (_q[_m][~rfilter]**2).mean()
            else:
                loss = (_q[_m]**2).mean()
            ########################### optimize ###########################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ########################### Output ###########################
            if i % params['save_freq'] == 0:
                self.save_model(model_folder, label=str(i))
                logging.info('Epoch={}, E1={}, al1={}, al2={},axz_zz={}'.format(i, th2np(model_ext.E1),th2np(model_ext.al1), th2np(model_ext.al2), th2np(model_ext.axz_zz)))
                # if noneq and i > 1:
                #     for setidx in range(len(trainset)):
                #         self.validate(trainset, params,  label=str(i+setidx), model_folder=model_folder, setidx=setidx)

    def train_kernel( self, trainset, params ):
        model_ext = self.model_ext
        model_noise = self.model_noise
        model_folder = params['model_folder']
        bs = [ params['window_x'],  params['window_y'], params['window_z'] ]   # blocksize, x,y,z #,time-axis
        dev = self.dev
        dt = self.dt
        lmem = self.lmem
        nset = len(trainset)
        assert (nset == 1)
        profile, debug = params['profile'],  params['debug']
        kosl = []
        kost = []
        kosl_fdt = []
        kost_fdt = []
        yw_osl = []
        yw_ost = []
        for i in range(params['kernel_epoch']):
            setidx = 0
            #################### LOADING ####################
            nx, ny, nz, _ , nframes =  trainset[setidx]['r'].shape
            length =  np.minimum(params['max_frames'], nframes-10)
            beg = np.random.randint(nframes - length)
            end = beg + length
            xbeg = np.random.randint(nx - bs[0])
            ybeg = np.random.randint(ny - bs[1])
            zbeg = np.random.randint(nz - bs[2])
            dataset={ #(#nx, #ny, #nz, 3, #frames)
                'r':   th.tensor(trainset[setidx]['r'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v':   th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'a':   th.tensor(trainset[setidx]['a'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'e':   th.tensor(trainset[setidx]['e'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v_more': th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+1:end
                    ], dtype=th.float32, device=dev),
            }
            #################### FORCE ######################
            potential_force, damp_force  = model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
            a_pred =  potential_force + damp_force
            a_ref = dataset['a']
            assert a_pred.shape == a_ref.shape
            noise = a_ref - a_pred  ##  (nx, ny, nz, 3, nframes)
            ## 
            _x = dataset['r'] - dataset['r'].mean()
            _vr =  dataset['v']                                 ##  v(t-0.5), (nx, ny,nz, 3, nframes)
            _vi = dataset['v'] + 0.5*dataset['a']* dt           ##  v(t), (nx, ny, nz, 3, nframes)
            _n = noise.detach()                                 ##  n(t): (nx, ny, nz, 3, nframes)
            _q = a_ref - potential_force.detach()               ##  a(t)-F(t):  (nx, ny, nz, 3, nframes)
            _w = _n * 0                                         ## placeholder for white noise
            ## filtering out the boundary for ensemble average
            _ns = self.nskip 
            _m = th.zeros_like(dataset['r'][:,:,:,0,0]).to(dev)
            _m[_ns:-_ns, _ns:-_ns, _ns:-_ns] += 1    ### Hide the boundary
            _m = _m.to(dtype=bool)
            ## filtering out the boundary for ensemble average
            _ns = self.nskip + 1 
            _m2 = th.zeros_like(dataset['r'][:,:,:,0,0]).to(dev)
            _m2[_ns:-_ns, _ns:-_ns, _ns:-_ns] += 1    ### Hide the boundary
            _m2 = _m2.to(dtype=bool)
            ## get all correlation function
            corr_dict_x = self.get_all_correlation( _n[...,0,:], _w[...,0,:], _vr[...,0,:], _vi[...,0,:], _q[...,0,:], _m, _m2,
                roll_axis=0)
            corr_dict_y = self.get_all_correlation( _n[...,1,:], _w[...,1,:], _vr[...,1,:], _vi[...,1,:], _q[...,1,:], _m, _m2,
                roll_axis=1)                
            corr_dict_z = self.get_all_correlation( _n[...,2,:], _w[...,2,:], _vr[...,2,:], _vi[...,2,:], _q[...,2,:], _m, _m2,
                roll_axis=2)
            self.corr_dict_x = corr_dict_x
            self.corr_dict_y = corr_dict_y
            self.corr_dict_z = corr_dict_z
            ############################## Compute memory kernal ##############################
            _kosl_fdt, _knnl_fdt = self.get_kernal_fdt( corr_dict_z,  e1=8 )  
            _kost_fdt, _knnt_fdt = self.get_kernal_fdt( corr_dict_x,  e1=8 )  
            _kosl, _knnl, scf_errorl, corrl_dict_nv = self.get_kernal_av_2rd( 
                corr_dict_z, alpha=0, e1=8, onsite_only=model_ext.onsite_only, debug=debug)
            _kost, _knnt, scf_errort, corrt_dict_nv = self.get_kernal_av_2rd(
                corr_dict_x, alpha=0,  e1=8, onsite_only=model_ext.onsite_only, debug=debug)
            kosl.append(_kosl)
            kost.append(_kost)
            kosl_fdt.append(_kosl_fdt)
            kost_fdt.append(_kost_fdt)
            ############################## Update memory kernel ##############################
            model_ext.kosl = sum(kosl) / len(kosl)
            model_ext.kost = sum(kost) / len(kost)
            self.kosl_fdt = sum(kosl_fdt) / len(kosl_fdt)
            self.kost_fdt = sum(kost_fdt) / len(kost_fdt)
            cut_region = int(lmem /  10)
            model_ext.kosl -= model_ext.kosl[-cut_region:].mean()
            model_ext.kost -= model_ext.kost[-cut_region:].mean()
            ############################## Compute & update Yule-Walker kernal ##############################
            if i > 1:
                _yw_osl, _yw_nnl, agl_error, agl_poly = self.get_YW( corr_dict_z, alpha=0, onsite_only=True)   
                _yw_ost, _yw_nnt, agt_error, agt_poly = self.get_YW( corr_dict_x, alpha=0, onsite_only=True)   
                yw_osl.append(_yw_osl)
                yw_ost.append(_yw_ost)
                model_noise.yw_osl =  sum(yw_osl) / len(yw_osl) 
                model_noise.yw_ost =  sum(yw_ost) / len(yw_ost)  
        ############################## Save Model ##############################
        self.save_model(model_folder, label='0')
        ############################## logging ##############################
        logging.info('Total iterations={}'.format(i))
        logging.info('scf inversion error: {}'.format( scf_errorl ))
        logging.info('color noise mean:{}, std:{}'.format(th2np(_n.mean()), (th2np(_n)**2).mean()**0.5  ) )
        logging.info('total force 2-norm:{}'.format(  (th2np(a_ref)**2).mean()**0.5 ))
        logging.info('damp&noise 2-norm: {}'.format(  (th2np(_q)**2).mean()**0.5    ))
        logging.info('potforce 2-norm: {}'.format(  (th2np(potential_force)**2).mean()**0.5  ))
        logging.info('damping 2-norm: {}'.format( (th2np(damp_force)**2).mean()**0.5   ))
        logging.info('K_osl(t): {}'.format(model_ext.kosl))
        logging.info('K_ost(t): {}'.format(model_ext.kost))
        return

    def train_GAR( self, trainset, params ):
        model_ext = self.model_ext
        model_noise = self.model_noise
        model_folder = params['model_folder']
        bs = [ params['window_x'],  params['window_y'], params['window_z'] ]   # blocksize, x,y,z #,time-axis
        dt = self.dt
        dev = self.dev
        lmem = self.lmem
        nset = len(trainset)
        assert (nset == 1)
        optimizer = th.optim.Adam( list(model_noise.parameters()) , lr = params['lr_GAR'] )
        scheduler = StepLR(optimizer, step_size=params['lr_window'], gamma=params['lr_decay'])
        for i in range(params['epoch']):
            setidx = 0
            #################### LOADING ####################
            nx, ny, nz, _ , nframes =  trainset[setidx]['r'].shape
            length =  np.minimum(params['max_frames'], nframes-10)
            beg = np.random.randint(nframes - length)
            end = beg + length
            xbeg = np.random.randint(nx - bs[0])
            ybeg = np.random.randint(ny - bs[1])
            zbeg = np.random.randint(nz - bs[2])
            dataset={ #(#nx, #ny, #nz, 3, #frames)
                'r':   th.tensor(trainset[setidx]['r'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v':   th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'a':   th.tensor(trainset[setidx]['a'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'e':   th.tensor(trainset[setidx]['e'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+lmem:end
                    ], dtype=th.float32, device=dev),
                'v_more': th.tensor(trainset[setidx]['v'][
                    xbeg:xbeg + bs[0], ybeg:ybeg + bs[1], zbeg:zbeg + bs[2], :, beg+1:end
                    ], dtype=th.float32, device=dev),
            }
            #################### FORCE ######################
            potential_force, damp_force  = model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
            a_pred =  potential_force + damp_force
            a_ref = dataset['a']
            assert a_pred.shape == a_ref.shape
            #################### NOISE ######################
            noise = a_ref - a_pred  ##  (nx, ny, nz, 3, nframes)
            noise_pred, noise_ref, noise_sigma = self.get_noise( dataset['r'], noise.detach() )  ##  (nx, ny, nz, 3, nframes-lmem)
            white_noise = noise_ref - noise_pred
            #################### LOSS ######################
            ## filtering out the boundary for ensemble average
            _ns = self.nskip 
            _m = th.zeros_like(dataset['r'][:,:,:,0,0]).to(self.dev)
            _m[_ns:-_ns, _ns:-_ns, _ns:-_ns] += 1    ### Hide the boundary
            _m = _m.to(dtype=bool)
            ## filtering out the boundary for ensemble average
            _ns = self.nskip + 1 
            _m2 = th.zeros_like(dataset['r'][:,:,:,0,0]).to(self.dev)
            _m2[_ns:-_ns, _ns:-_ns, _ns:-_ns] += 1    ### Hide the boundary
            _m2 = _m2.to(dtype=bool)
            ### loss for  noise model : Maximal likelihood
            loss_white1 =  th.log(noise_sigma**2).mean()               
            loss_white2 = ((noise_ref - noise_pred)/noise_sigma)**2
            loss_white2 =  loss_white2[_m2].mean() 
            loss = loss_white1 + loss_white2  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ###########################    Output    ###########################
            if i % params['print_freq'] == 0:
                _n = noise_pred.detach()
                _w = white_noise.detach()
                _R = noise.detach()
                ## corr (n, w)
                corr_nw  = [  (_n[_m2][...,2,:] * _w[_m2][...,2,:]).mean()] + [  (_n[_m2][...,2,:-iT] * _w[_m2][...,2,iT:]).mean() for iT in range(1,lmem)] 
                corr_ww  = [  (_w[_m2][...,2,:] * _w[_m2][...,2,:]).mean()] + [  (_w[_m2][...,2,:-iT] * _w[_m2][...,2,iT:]).mean() for iT in range(1,lmem)] 
                corr_RR  = [  (_R[_m2][...,2,:] * _R[_m2][...,2,:]).mean()] + [  (_R[_m2][...,2,:-iT] * _R[_m2][...,2,iT:]).mean() for iT in range(1,lmem)] 

                fig, ax= plt.subplots(2)
                ax[0].hist( th2np(_n[_m]).flatten(), bins=100, density=True, 
                    label='Mean={:.3f},std={:.3f}'.format(th2np(_n[_m].mean()), th2np(_n[_m].std()))
                    )
                ax[0].legend()
                ax[0].set_ylabel('p(Noise Pred)')
                ax[1].hist( th2np(_w[_m2]).flatten(), bins=100, density=True,
                    label='Mean={:.3f},std={:.3f}'.format(th2np(_w[_m2].mean()), th2np(_w[_m2].std()))
                    )
                ax[1].legend()  
                ax[1].set_ylabel('p(Noise Residue)')

                plt.tight_layout()
                plt.savefig(os.path.join(model_folder, 'NoiseDist-{}.png'.format(i) ))
                plt.close(fig)

                fig, ax= plt.subplots(2)
                ax[0].plot(np.arange(lmem)*dt, th2np(th.tensor(corr_RR))/th2np(corr_RR[0]), label='<R(r,t),R(r,0)>/R^2' )
                ax[0].plot(np.arange(lmem)*dt, th2np(th.tensor(corr_ww))/th2np(corr_ww[0]), label='<w(r,t),w(r,0)>/w^2' )
                ax[0].legend()
                ax[0].set_xlabel('t[ps]')
                ax[0].set_ylabel('NACF')                
                nw_norm = th2np(corr_RR[0]*corr_ww[0])**0.5
                ax[1].plot(np.arange(lmem)*dt, th2np(th.tensor(corr_nw))/nw_norm, label='<n_pred(0), w(t)>/R/w' )
                ax[1].set_xlabel('t[ps]')
                ax[1].set_ylabel('NCF')
                ax[1].legend()
                plt.tight_layout()
                plt.savefig(os.path.join(model_folder, 'NoiseCorr-{}.png'.format(i) ))
                plt.close(fig)
                ###################################### LOGGING ###################################### 
                logging.info("Epoch={},     loss_white={:.3f} ".format(
                                        i,      loss_white1 + loss_white2 ))
                logging.info("Color_mean={:.3f}, std= {:.3f}  , white_mean={:.3f}, std= {:.3f}, noise_sigma={}".format(
                            _n.mean(), _n.std(),  _w.mean(), _w.std(), th2np(model_noise.sigma) ))

            if i % params['save_freq'] == 0 and i>1:
                self.save_model(model_folder, label=str(i))
                # self.validate(trainset, params,   label=str(i+setidx), model_folder=model_folder, setidx=0)
 

    ##########################    Simulation    ##########################
    def sim_init(self, r_list, v_list, n_list):
        self.r_list = r_list
        self.v_list = v_list
        self.n_list = n_list
        self.n_buffer = [ n_list[-i-1].unsqueeze(-1) for i in range( self.len_ag ) ]
        self.n_buffer = th.cat(self.n_buffer,-1)  ##  (nx,ny,nz, 3, nbatch, lmem)
        
    def sim_step(self, efield, pop=True, debug=False):
        '''
        Returns:
            potential_force:
            damp_force
            E_pot: (interaction + external potential energy) per particle. numpy.array of shape (3)
            E_kin: kinetic energy per particle. numpy.array of shape (3)
        '''
        dt = self.dt
        ########
        potential_force, damp_force  = self.model_ext( self.r_list[-1], self.v_list, efield, autograd=False )
        with th.no_grad():
            noise_traj= [ self.n_list[-i].unsqueeze(-1) for i in range(1, self.len_ag+1) ]
            noise_traj = th.cat(noise_traj,-1)  ##  (nx,ny,nz, 3, nbatch, lmem)
            noise_pred, noise_sigma = self.model_noise( self.r_list[-1], noise_traj )
            noise_pred += th.randn_like(noise_pred) * noise_sigma 
            self.n_list.append(noise_pred)
        a = potential_force + damp_force + noise_pred
        self.v_list.append(self.v_list[-1] + a * dt)
        self.r_list.append(self.r_list[-1] + self.v_list[-1] * dt)
        relax_steps = max(self.lmem + 1, self.len_ag + 1)
        if pop and (len(self.r_list) >  relax_steps+1):
            # self.r_list.pop(0)
            # self.v_list.pop(0)
            # self.n_list.pop(0)
            del self.r_list[0]
            del self.v_list[0]
            del self.n_list[0]
        if debug:
            print('noise 2-norm:', (th2np(noise_pred)**2).mean()**0.5 )
            print('potforce 2-norm:', (th2np(potential_force)**2).mean()**0.5 )
            print('damping 2-norm:', (th2np(damp_force)**2).mean()**0.5 )
        ## record the energy
        with th.no_grad():
            natoms = self.r_list[-1].shape[0] * self.r_list[-1].shape[1] * self.r_list[-1].shape[2]  
            E_pot = self.model_ext.get_total_pot_components(efield, self.r_list[-1]).mean(-1)   # (3)
            E_pot = th2np( E_pot / natoms).flatten()
            v2 =  (self.v_list[-2]**2).mean(-1).sum([0,1,2]) # (3)
            E_kin = th2np( 0.5 * (self.model_ext.mass * v2)  / natoms)
        return potential_force, damp_force, E_pot, E_kin
 
    def langevin_step(self, efield, damp=0.1,  pop=False, debug=False):
        dt = self.dt
        assert damp > 0
        # noise_sigma = (damp / dt * self.v2_avg *2)**0.5
        noise_sigma = (damp / dt *2 / self.beta / th2np(self.model_ext.mass) )**0.5
        ########
        gradient_force = self.model_ext.get_potential_force(self.r_list[-1]).detach()
        with th.no_grad():
            external_force = self.model_ext.get_external_force(efield, self.r_list[-1])
            damp_force = - damp * self.v_list[-1]
            noise_force = th.randn_like(damp_force) * th.tensor(noise_sigma[None,None,None,:,None]).to(device=damp_force.device)
        a = gradient_force + external_force + damp_force + noise_force
        self.v_list.append(self.v_list[-1] + a * dt)
        self.r_list.append(self.r_list[-1] + self.v_list[-1] * dt)
        self.n_list.append(noise_force)
        if pop:
            del self.r_list[0]
            del self.v_list[0]
            del self.n_list[0]
            # self.r_list.pop(0)
            # self.v_list.pop(0)
            # self.n_list.pop(0)
        natoms = self.r_list[-1].shape[0] * self.r_list[-1].shape[1] * self.r_list[-1].shape[2]  
        E_pot = self.model_ext.get_total_pot_components(efield, self.r_list[-1]).mean(-1)   # (3)
        E_pot = th2np( E_pot / natoms).flatten()
        v2 =  (self.v_list[-2]**2).mean(-1).sum([0,1,2]) # (3)
        E_kin = th2np( 0.5 * (self.model_ext.mass * v2)  / natoms)
        if debug:
            raise NotImplementedError
        return E_pot, E_kin

    def compute_trajectory(self, trainset, params,  replicate=1, nbatch = 1,  total_time=100,  setidx = 0 ):
        dev = self.dev
        lmem = self.lmem
        dt = self.dt
        ################## LOAD DATA  ###########################
        nset =  len(trainset) 
        # setidx = np.random.randint(nset)
        nx, ny, nz, _ , nframes =  trainset[setidx]['r'].shape
        beg = lmem
        end = beg + lmem + max(self.lmem+1, self.len_ag+1) + 1
        dataset={
            'r':      th.tensor(trainset[setidx]['r'][..., beg+lmem:end], dtype=th.float32, device=dev),
            'v':      th.tensor(trainset[setidx]['v'][..., beg+lmem:end], dtype=th.float32, device=dev),
            'a':      th.tensor(trainset[setidx]['a'][..., beg+lmem:end], dtype=th.float32, device=dev),
            'e':      th.tensor(trainset[setidx]['e'][..., beg+lmem:end], dtype=th.float32, device=dev),
            'v_more': th.tensor(trainset[setidx]['v'][..., beg+1:end], dtype=th.float32, device=dev),
        }
        potential_force, damp_force  = self.model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
        a_pred =  potential_force.detach()  + damp_force.detach()
        noise = dataset['a'] - a_pred
        ######################  SIMULATE #########################
        relax_steps = max(self.lmem+1, self.len_ag+1)
        r_list = [dataset['r'][...,[i]].repeat(replicate,replicate,1,1, nbatch) for i in range(relax_steps) ]
        v_list = [dataset['v'][...,[i]].repeat(replicate,replicate,1,1, nbatch) for i in range(relax_steps) ]
        n_list = [noise[...,[i]].repeat(replicate,replicate,1,1,nbatch) for i in range(relax_steps)]
        efield = dataset['e'][...,[0]].repeat(replicate,replicate,1,1,nbatch) 
        total_steps = int(total_time / dt)
        self.sim_init(r_list, v_list,n_list)
        traj_list = []
        traj_ref_list = []
        for step in range(total_steps - relax_steps):
            if params['langevin']:
                ## Langevin dynamics
                damping = np.abs( th2np(self.model_ext.kos_long).sum())
                E_pot, E_kin = self.langevin_step(efield, damping, pop=True)
            else:
                #AIGLE dynamics
                pforce, vforce, E_pot, E_kin = self.sim_step(efield, pop=True)
            if step % 100 == 0:
                try:
                    traj_ref_list.append(trainset[setidx]['r'][...,2, beg + lmem + relax_steps+step])
                    print('reference attached: step={}'.format(step))
                except:
                    pass
                traj_list.append(th2np(self.r_list[-1][:,:,:,2,0]))
                print('step:{}'.format(step))
        return np.array(traj_list), np.array(traj_ref_list) # (nframes, nx, ny, nz)
