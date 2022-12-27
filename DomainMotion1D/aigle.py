
import os
from  time import time
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
from numpy.linalg import lstsq
from matplotlib import pyplot as plt
import torch as th
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model import *
from utility import *

 
class aigle0D:
    def __init__(self, temp, dt=0.01, len_ag=50, lmem=100,  model_ext=None, model_noise=None):
        ### system parameters
        self.dev = 'cuda' if th.cuda.is_available() else 'cpu'
        self.kb =  8.6173e-5 * 1000 # meV K-1
        self.temp = temp
        self.beta = 1 / self.kb / temp
        self.dt = dt ## the time step of the data
        self.len_ag = len_ag
        self.lmem = lmem
        # self.amax = 0
        if model_ext is None:
            self.model_ext = poly_2Dsystem(lmem=lmem).to(self.dev)
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
        ## 
        self.writer = None
    
    def save_model(self, model_folder, label='0'):
        th.save(self.model_ext.state_dict(), os.path.join(model_folder,'model_ext.{}.ckpt'.format(label)))
        th.save(self.model_noise.state_dict(), os.path.join(model_folder,'model_noise.{}.ckpt'.format(label)))
        return 

    def load_model(self, model_folder, label='0'):
        self.model_ext.load_state_dict(th.load(
            os.path.join(model_folder, 'model_ext.{}.ckpt'.format(label)),
            map_location=self.dev))
                    
        self.model_noise.load_state_dict(th.load(
            os.path.join(model_folder, 'model_noise.{}.ckpt'.format(label)),
            map_location=self.dev))
        return

    def get_noise( self, noise ):
        '''
        Transform a continuous trajectory to the input. Then get predicted noise 
        Args:
            noise: Tensor of shape ( nbatch) 
        '''
        lag = self.len_ag
        noise_ref = noise[ lag:] ## (nframes-lag )
        noise_traj = []
        for i in range(1,lag+1):
            noise_traj.append(noise[lag-i:-i,None])
        noise_traj = th.cat(noise_traj,-1) ##   ( nframes-lag, lag)
        noise_pred, noise_sigma = self.model_noise( noise_traj )  ##  (*,nframes-lag)
        assert noise_pred.shape == noise_ref.shape 
        return noise_pred, noise_ref, noise_sigma
     
    def get_all_correlation(self, _n, _w, _vr, _vi, _q ):
        lmem =self.lmem
        ## compute correlation
        corr_ww = Corr_t( _w, _w, lmem)                          ## Corr[ w(0), w(t)] 
        corr_nn = Corr_t( _n, _n, lmem+1)                        ## Corr[ n(0), n(t)]
        corr_vv = Corr_t( _vr, _vr, lmem)                        ## Corr[ v(0), v(t)]
        corr_wv = Corr_t(_vi[..., -_w.shape[-1]:], _w, lmem)     ## Corr[ v(0), w(t)]
        corr_vrvi = Corr_t(_vi[...,:-1], _vr[...,1:], lmem)      ## Corr[ v(0), v(t+0.5)]
        corr_qv = Corr_t(_vi, _q, lmem+1)                        ## Corr[ v(0), q(t)] 
        return {
            'corr_ww': corr_ww,
            'corr_nn': corr_nn,
            'corr_vv': corr_vv,
            'corr_wv': corr_wv,
            'corr_vrvi': corr_vrvi,
            'corr_qv': corr_qv,
        }
    
    def get_kernel_fdt(self, corr_dict ):
        G = corr_dict['corr_vv'][0]
        kernel_fdt = - corr_dict['corr_nn'] / G * self.dt
        kernel_fdt = (kernel_fdt[:-1] + kernel_fdt[1:]) / 2
        return kernel_fdt

    def get_kernel_av(self, corr_dict ):
        lmem = self.lmem
        C = np.zeros((lmem, lmem))   ## C[ii,jj]=Corr[v(ii+0.5-jj),v(0)]
        corr_qv = corr_dict['corr_qv'] 
        for ii in range(lmem):
            for jj in range(ii+1):
                C[ii,jj] = corr_dict['corr_vrvi'][ ii-jj ]
        kernel_retard, res, rank, singular =  lstsq( C, corr_qv[1:], rcond=0.0001 )

        scf_error = ((corr_qv[1:] - C @ kernel_retard)**2).sum() / (corr_qv[1:]**2).sum()
        scf_error = scf_error**0.5
        ### get <R(t),v(0)> for all important terms 
        corr_nv = corr_qv.copy()
        corr_nv[1:] -= C @  self.model_ext.kernel
        return kernel_retard, scf_error, corr_nv

    def get_YW(self, corr_dict ):
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
        return ag_coef,  ag_error, ag_poly
 
    def sim_init(self, r_list, v_list, n_list):
        self.r_list = r_list
        self.v_list = v_list
        self.n_list = n_list

    def sim_step(self, efield, pop=True ):
        dt = self.dt
        ########
        with th.no_grad():
            potential_force, damp_force  = self.model_ext( self.r_list[-1], self.v_list, th.zeros_like(self.r_list[-1]) + efield )
            noise_traj= [ self.n_list[-i].unsqueeze(-1) for i in range(1, self.len_ag+1) ]
            noise_traj = th.cat(noise_traj,-1)  ##  (nx,ny, nbatch, lmem)
            noise_pred, noise_sigma = self.model_noise( noise_traj )
            noise_pred += th.randn_like(noise_sigma) * noise_sigma 
            self.n_list.append(noise_pred)
        a = potential_force + damp_force + noise_pred
        # a = th.clamp(a, -self.amax, self.amax)
        self.v_list.append(self.v_list[-1] + a * dt)
        self.r_list.append(self.r_list[-1] + self.v_list[-1] * dt)
        relax_steps = max(self.lmem + 1, self.len_ag + 1)
        if pop and (len(self.r_list) >  relax_steps+1):
            self.r_list.pop(0)
            self.v_list.pop(0)
            self.n_list.pop(0)
        return 
 
    def train_init( self, trainset, batchsize=4, epoch=10010 , scf_freq = 10, print_freq=100, save_freq=1000,  
                    model_folder='model' ):
        model_ext = self.model_ext
        model_noise = self.model_noise
        dev = self.dev
        dt = self.dt
        lmem = self.lmem
        nset = len(trainset)
        optimizer  = th.optim.Adam([
                {'params': model_ext.parameters()},
                {'params': model_noise.parameters()}
            ], lr=1e-3 )
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
        writer = SummaryWriter(os.path.join('runs',model_folder))
        self.writer = writer
        for i in range(epoch):
            loss = 0
            corr_dict = { 'corr_ww': 0, 'corr_nn': 0, 'corr_vv': 0, 'corr_wv': 0, 'corr_vrvi': 0, 'corr_qv': 0, }
            setidx = np.random.choice(nset, size=batchsize, replace=False)
            for j in range(batchsize):
                #################### LOADING ####################
                dataset={
                    'r':   th.tensor(trainset[setidx[j]]['r'][lmem:], dtype=th.float32, device=dev),
                    'v':   th.tensor(trainset[setidx[j]]['v'][lmem:], dtype=th.float32, device=dev),
                    'a':   th.tensor(trainset[setidx[j]]['a'][lmem:], dtype=th.float32, device=dev),
                    'e':   th.tensor(trainset[setidx[j]]['e'][lmem:], dtype=th.float32, device=dev),
                    'v_more': th.tensor(trainset[setidx[j]]['v'][1:], dtype=th.float32, device=dev),
                }
                #################### FORCE ######################
                potential_force, damp_force  = model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
                a_pred =  potential_force + damp_force
                a_ref = dataset['a']
                assert a_pred.shape == a_ref.shape
                #################### NOISE ######################
                noise = a_ref - a_pred  ##  ( nframes)
                noise_pred, noise_ref, noise_sigma = self.get_noise( noise.detach() )  ##  (nx, ny, nframes-lmem)
                white_noise = noise_ref - noise_pred
                #################### LOSS ######################
                loss_reg = (noise**2).mean()   ## loss for potential model
                loss_white =  th.log(noise_sigma**2).mean( )    ## loss for  noise model        
                loss_white += (((noise_ref - noise_pred)/noise_sigma)**2).mean()
                loss += loss_white / batchsize  
                loss += loss_reg / batchsize 
                ############################## Get all correlation ######################
                _w = white_noise.detach()                    ##  w(t):  ( nframes-lmem)
                _n = noise.detach()                          ##  n(t):  ( nframes)
                _q = a_ref - potential_force.detach()        ##  a(t)-F(t):  (  nframes)
                _vr =  dataset['v']                          ##  v(t-0.5), (  nframes)
                _vi = dataset['v'] + 0.5*dataset['a']* dt    ##  v(t), (  nframes)
                corr_dict['corr_ww'] += Corr_t( _w, _w, lmem) / batchsize                         ## Corr[ w(0), w(t)] 
                corr_dict['corr_nn'] += Corr_t( _n, _n, lmem+1) / batchsize                       ## Corr[ n(0), n(t)]
                corr_dict['corr_vv'] += Corr_t( _vr, _vr, lmem) / batchsize                       ## Corr[ v(0), v(t)]
                corr_dict['corr_wv'] += Corr_t(_vi[..., -_w.shape[-1]:], _w, lmem) / batchsize    ## Corr[ v(0), w(t)]
                corr_dict['corr_vrvi'] += Corr_t(_vi[...,:-1], _vr[...,1:], lmem) / batchsize     ## Corr[ v(0), v(t+0.5)]
                corr_dict['corr_qv'] += Corr_t(_vi, _q, lmem+1) / batchsize                       ## Corr[ v(0), q(t)] 
             ############################## SCF steps ##############################
            if i % scf_freq == 0:
                ############################## Compute memory kernel ##############################
                kernel_fdt  = self.get_kernel_fdt( corr_dict )
                kernel_numpy, scf_error, corr_nv = self.get_kernel_av( corr_dict )
                ############################## Update memory kernel ##############################
                cut_region = int(lmem /  10)
                if i==0:
                    model_ext.kernel = kernel_numpy
                else:
                    model_ext.kernel +=  0.01 * (kernel_numpy - model_ext.kernel)
                model_ext.kernel -= model_ext.kernel[-cut_region:].mean()
                disper_os = (model_ext.kernel * corr_dict['corr_vv']).sum()
                ############################## Update GAR kernel ##############################
                ag_numpy, ag_error, ag_poly = self.get_YW( corr_dict )  
                if i==0:
                    model_noise.linear =  ag_numpy 
                else:
                    model_noise.linear +=  0.01 * (ag_numpy - model_noise.linear)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            ###########################    Output    ###########################
            if i % print_freq == 0:
                with th.no_grad():
                    writer.add_scalar('loss_reg', loss_reg, i)
                    writer.add_scalar('loss_GAR', loss_white, i)
                    writer.add_scalar('SCF_error', scf_error, i)
                    writer.add_scalars('ForceConst', {
                        'well_freq_shift': th2np(model_ext.well_freq_shift).sum(), 
                        'well_coef': th2np(model_ext.well_coef).sum(), 
                        'rescale': th2np(model_ext.rescale).sum(), 
                        'well_bias': th2np(model_ext.well_bias).sum(), 
                        'E_linear': th2np(model_ext.E_linear).sum()},  i)
                    writer.add_scalars('Noise_std', {
                        'color': _n.std(), 
                        'white': _w.std(), 
                        'GAR_white':model_noise.sigma.sum()},  i)
                    
                    ktime = (np.arange(lmem)+0.5) * dt
                    fig, ax= plt.subplots()
                    ax.plot(ktime, -model_ext.kernel, label='model, sum={:.3f}'.format(-model_ext.kernel.sum()))
                    ax.plot(ktime,       -kernel_fdt, label='FDT, sum={:.3f}'.format(-kernel_fdt.sum()))
                    ax.axhline(y=0, xmin=0, xmax=lmem*dt, markersize=0)
                    ax.legend()
                    writer.add_figure('-K(t)', fig, i, close=True )

                    fig, ax= plt.subplots()
                    ax.hist( th2np(_n).flatten(), bins=100, density=True, 
                        label='({},{})'.format(_n.mean(),_n.std()))
                    ax.legend()
                    writer.add_figure('NoiseColor Dist', fig, i, close=True )
                    
                    fig, ax= plt.subplots()
                    ax.hist( th2np(_w).flatten(), bins=100, density=True,
                        label='({},{})'.format(_w.mean(),_w.std()))
                    ax.legend()
                    writer.add_figure('NoiseWhite Dist', fig, i, close=True )
                    
                    fig, ax= plt.subplots()
                    input_r = th.linspace(19,21,101).to(dev)
                    potential = model_ext.get_potential(input_r) 
                    potential = potential - potential.min()
                    for e in [0,1,2,3]:
                        external_field = model_ext.get_external_force(input_r, e).flatten()
                        pot = potential.flatten() -  external_field * input_r 
                        pot = pot - pot[pot.shape[0]//2] + (potential.max() - potential.min())
                        ax.plot( th2np(input_r), th2np(pot.flatten()), label='E={}mV/A'.format(e) )
                    ax.legend()
                    writer.add_figure('Free energy surface', fig, i, close=True )

                    nv_norm = (corr_dict['corr_nn'][0]*corr_dict['corr_vv'][0])**0.5 
                    fig, ax= plt.subplots()
                    ax.plot(np.arange(lmem+1)*dt, corr_nv/ nv_norm, label='R(t)v(0)')
                    ax.legend()
                    writer.add_figure('NCF(R,v)', fig, i, close=True )

                    fig, ax= plt.subplots()
                    ax.plot(np.arange(lmem+1)*dt, corr_dict['corr_nn']/corr_dict['corr_nn'][0], label='n(t)n(0)')
                    ax.plot(np.arange(lmem)*dt, corr_dict['corr_ww']/corr_dict['corr_ww'][0], label='w(t)w(0)')
                    ax.legend()
                    writer.add_figure('NACF-Noise', fig, i, close=True )
                    
                    fig, ax= plt.subplots()
                    ax.plot(np.arange(lmem)*dt, corr_dict['corr_vv']/corr_dict['corr_vv'][0])
                    writer.add_figure('NACF[v(t),v(0)]', fig, i, close=True )
            if i % save_freq == 0 and i>1:
                self.save_model(model_folder, label=str(i))
                self.validate(trainset, label=str(i), model_folder=model_folder)
 
    def validate(self, trainset,  nbatch = 10, total_time=100, label='0', model_folder='model' ):
        dev = self.dev
        lmem = self.lmem
        dt = self.dt
        ################## LOAD DATA  ###########################
        nset =  len(trainset) 
        corr_ww = 0
        corr_nn = 0
        corr_vv = 0
        nM = int(20 / self.dt)
        for setidx in range(nset):
            dataset= {
                'v':   th.tensor(trainset[setidx]['v'][lmem:], dtype=th.float32, device=dev),
                'r':   th.tensor(trainset[setidx]['r'][lmem:], dtype=th.float32, device=dev),
                'a':   th.tensor(trainset[setidx]['a'][lmem:], dtype=th.float32, device=dev),
                'e':   th.tensor(trainset[setidx]['e'][lmem:], dtype=th.float32, device=dev),
                'v_more': th.tensor(trainset[setidx]['v'][1:], dtype=th.float32, device=dev),
            }
            potential_force, damp_force  = self.model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
            potential_force = potential_force.detach() 
            damp_force = damp_force.detach()
            a_pred =  potential_force + damp_force
            a_ref = dataset['a']
            noise = a_ref - a_pred
            with th.no_grad():
                noise_pred, noise_ref, noise_sigma = self.get_noise( noise.detach() )  ##  (nx, ny, nframes-lmem)
                white_noise = noise_ref - noise_pred
            ################## Reference correlation  ###########################
            corr_ww += Corr_t( white_noise, white_noise, nM) / nset           ## Corr[ w(0), w(t)] 
            corr_nn += Corr_t( noise, noise, nM) / nset                       ## Corr[ n(0), n(t)]
            corr_vv += Corr_t( dataset['v'], dataset['v'], nM) / nset     
        ######################  SIMULATE #########################
        relax_steps = max(self.lmem+1, self.len_ag+1)
        r_list = [dataset['r'][[i]].repeat(nbatch) for i in range(relax_steps) ]
        v_list = [dataset['v'][[i]].repeat(nbatch) for i in range(relax_steps) ]
        n_list = [noise[[i]].repeat(nbatch) for i in range(relax_steps)]
        efield = dataset['e'][0] * 1.0
        total_steps = int(total_time / dt)
        self.sim_init(r_list, v_list, n_list)
        for step in range(total_steps - relax_steps):
            self.sim_step(efield, pop=False)
            if step % 1000 == 0:
                print('step:{}'.format(step))
                print(r_list[-1])
        ######################  Simulated correlation #########################
        self.r_list = th.cat([ x.unsqueeze(-1) for x in self.r_list], -1)   #( nbatch, nframes)
        self.v_list = th.cat([ x.unsqueeze(-1) for x in self.v_list], -1)
        self.n_list = th.cat([ x.unsqueeze(-1) for x in self.n_list], -1)
        corr_nn_sim = Corr_t( self.n_list, self.n_list, nM)  
        corr_vv_sim = Corr_t( self.v_list, self.v_list, nM)
        ######################  Plot color/white noise correlation  ########################
        plot_auto = int(2/self.dt)
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        kernel = self.model_ext.kernel
        ax[0].axhline(y=0, color='black', markersize=0)
        ax[0].plot(np.arange( plot_auto )*dt, corr_nn[:plot_auto] / corr_nn[0], 
            label=r'$C^{\mathrm{data}}_{RR}$', markersize = 0, linewidth=2)
        ax[0].plot(np.arange( plot_auto )*dt, corr_ww[:plot_auto] / corr_ww[0], 
            label=r'$C^{\mathrm{data}}_{ww}$', markersize = 0, linewidth=2)
        ax[0].plot(np.arange( plot_auto )*dt, corr_nn_sim[:plot_auto] / corr_nn_sim[0], 
            label=r'$C^{\mathrm{sim}}_{RR}$', markersize = 3, marker='o', linestyle='dashed', linewidth=2)
        ax[0].plot(np.arange( kernel.size)*dt + 0.5*dt,  kernel / kernel[0], 
            label=r'$\tilde{K}(\tau)$', markersize = 0, linestyle='dashed', linewidth=2)
        ax[0].set_xlabel(r'$\tau [ps]$')
        ax[0].set_ylabel('NACF')
        ax[0].legend(framealpha=0, fontsize=11)
        ax[0].set_title('(a)',loc='left')
        ## inset
        ins = ax[0].inset_axes([0.2, 0.5, 0.45, 0.43] )
        ins.tick_params(axis='both', which='major', labelsize=9)
        ## inset, Fourier corr(n,n) from data
        corr_nn_fft_data =  np.fft.fft(corr_nn).real 
        nframes = corr_nn_fft_data.size
        x_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
        wavenumber = x_fourier / 6 / np.pi * 100
        ins.plot(wavenumber[wavenumber < 400], corr_nn_fft_data[wavenumber < 400], 
                markersize=0, linestyle='solid', linewidth=1.5, label='data' )
        ## inset, Fourier corr(n,n) from AIGLE
        corr_nn_fft =  np.fft.fft(corr_nn_sim).real 
        nframes = corr_nn_fft.size
        x_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
        wavenumber = x_fourier / 6 / np.pi * 100
        ins.plot(wavenumber[wavenumber < 400], corr_nn_fft[wavenumber < 400], 
                markersize=0, linestyle='solid', linewidth=1.5, label='GLE' )
        ###### plot velocity autocorrelation
        plot_auto = int(2/dt)
        ax[1].axhline(y=0, color='black', markersize=0)
        ax[1].plot(np.arange( plot_auto )*dt, corr_vv[:plot_auto] / corr_vv[0], 
            label='Data', markersize = 0, linewidth=2)
        ax[1].plot(np.arange( plot_auto )*dt, corr_vv_sim[:plot_auto] / corr_vv_sim[0], 
            label='Simulated', markersize = 0, linewidth=2)
        ax[1].set_xlabel(r'$\tau$[ps]')
        ax[1].set_ylabel('Velocity NACF')
        ax[1].set_yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])
        ax[1].legend(framealpha=0, fontsize=9,loc='lower right' )
        ax[1].set_ylim(-0.4, 1.1)
        ax[1].set_title('(b)',loc='left')
        ###### inset 
        ins = ax[1].inset_axes([0.2, 0.5, 0.45, 0.43] )
        ins.tick_params(axis='both', which='major', labelsize=9)
        ###### inset: Fourier corr(v,v) from data
        corr_vv_fft_data =  np.fft.fft(corr_vv).real 
        nframes = corr_vv_fft_data.size
        x_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
        wavenumber = x_fourier / 6 / np.pi * 100
        ins.plot(wavenumber[wavenumber < 200], corr_vv_fft_data[wavenumber < 200], 
        markersize=0, linestyle='solid', linewidth=1.5,  )
        ###### inset: Fourier corr(v,v) from AIGLE
        corr_vv_fft =  np.fft.fft(corr_vv_sim).real 
        nframes = corr_vv_fft.size
        x_fourier = np.arange(nframes) * 2 * np.pi / nframes / dt  
        wavenumber = x_fourier / 6 / np.pi * 100
        ins.plot(wavenumber[wavenumber < 200], corr_vv_fft[wavenumber < 200], 
                markersize=0, linestyle='solid', linewidth=1.5,  )
        self.writer.add_figure('Validation]', fig, label, close=True )
 
    def train_force( self, trainset, batchsize=4, epoch=10010 ,  print_freq=100,  save_freq=1000,  
                    model_folder='model' ):
        model_ext = self.model_ext
        model_noise = self.model_noise
        dev = self.dev
        dt = self.dt
        lmem = self.lmem
        nset = len(trainset)
        optimizer  = th.optim.Adam( list(model_ext.parameters()) , lr = 0.01 )
        scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
        writer = SummaryWriter(os.path.join('runs',model_folder))
        self.writer = writer
        for i in range(epoch):
            loss = 0
            setidx =  np.random.choice(nset, size=batchsize, replace=False) 
            for j in range(batchsize):
                #################### LOADING ####################
                dataset={
                    'r':   th.tensor(trainset[setidx[j]]['r'][lmem:], dtype=th.float32, device=dev),
                    'v':   th.tensor(trainset[setidx[j]]['v'][lmem:], dtype=th.float32, device=dev),
                    'a':   th.tensor(trainset[setidx[j]]['a'][lmem:], dtype=th.float32, device=dev),
                    'e':   th.tensor(trainset[setidx[j]]['e'][lmem:], dtype=th.float32, device=dev),
                    'v_more': th.tensor(trainset[setidx[j]]['v'][1:], dtype=th.float32, device=dev),
                }
                #################### FORCE ######################
                potential_force, damp_force  = model_ext( dataset['r'],  dataset['v_more'], dataset['e'] )
                a_pred =  potential_force + damp_force
                a_ref = dataset['a']
                assert a_pred.shape == a_ref.shape
                _q = a_ref - potential_force      ##  a(t)-F(t):  (nx, ny, nframes)
                #################### Loss ######################
                noise = a_ref - a_pred  ##  (nx, ny, nframes)
                with th.no_grad():
                    noise_pred, noise_ref, noise_sigma = self.get_noise( noise )  ##  (nx, ny, nframes-lmem)
                    white_noise = noise_ref - noise_pred
                ############### reweight the loss ##############
                loss += (noise**2  ).mean() / batchsize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % print_freq ==0 and i > 0:
                fig, ax= plt.subplots()
                input_r = th.linspace(19,21,101).to(dev)
                potential = model_ext.get_potential(input_r) 
                potential = potential - potential.min()
                for e in [0,1,2,3]:
                    external_field = model_ext.get_external_force(input_r, e).flatten()
                    pot = potential.flatten() -  external_field * input_r 
                    pot = pot - pot[pot.shape[0]//2] + (potential.max() - potential.min())
                    ax.plot( th2np(input_r), th2np(pot.flatten()), label='E={}mV/A'.format(e) )
                ax.legend()
                writer.add_figure('Free energy surface (noneq)', fig, i, close=True )
            if i % save_freq == 0 and i>1:
                self.save_model(model_folder, label=str(i))
                # self.validate(trainset, label=str(i), model_folder=model_folder)
