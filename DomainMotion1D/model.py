import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.functional import conv1d
import torch.nn.functional as F
from numpy import log2, pi

class force_model(nn.Module):
    def __init__(self, in_channels=1, out_channels=26, kernel_size=3,  lmem=10, bias=True):
        super(force_model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.E_linear = nn.Parameter( th.ones(1)*1 )
        self.register_buffer('well_freq', th.ones(1))
        self.well_freq_shift = nn.Parameter( th.ones(1)*1.01 )
        self.well_coef = nn.Parameter(  th.ones(1)   )
        self.rescale = nn.Parameter(  th.ones(1)   )
        self.well_bias = nn.Parameter(  th.zeros(1)-0.2 )
        self.register_buffer('_kernel', -  0*th.exp(-th.arange(out_channels)/4))
        # self.kernel = nn.Parameter(- th.exp(-th.arange(out_channels)/4))
    
    @property
    def kernel(self):
        return self._kernel.detach().cpu().numpy()

    @kernel.setter
    def kernel(self, x):
        if type(x) == np.ndarray:
            self._kernel = th.tensor(x, dtype=self._kernel.dtype, device=self._kernel.device).flatten()
        else:
            self._kernel = x.flatten() * 1 
        
    def get_freq(self ):
        return self.well_freq * self.well_freq_shift * 2 * pi  # 1D

    def get_phase(self):
        return self.get_freq() * self.well_bias  # 1D

    def get_potential(self, r):
        arg = self.get_freq() * r + self.get_phase()
        potential = self.well_coef * th.tanh(self.rescale * (1-th.cos(arg)))
        return potential

    def get_potential_force(self, r):
        arg = self.get_freq() * r + self.get_phase()
        tanhz = th.tanh(self.rescale * (1-th.cos(arg)))
        potential_force = - self.well_coef * (1- tanhz**2) * self.rescale * self.get_freq() * th.sin(arg)
        return potential_force
    
    def get_external_force(self,r, e ):
        return 0*r + self.E_linear * e

    def get_damping_force(self, v):
        '''
            v:  ( nbatch + lmem-1 )
        '''
        _v = v[None,None, :]
        _k = th.flip( self._kernel.reshape(1,1,-1), [-1])
        force = conv1d(_v,  _k).flatten()
        assert force.shape[-1] == v.shape[-1] - self.out_channels + 1
        return force
    def get_damping_force_parallel(self, v ):
        '''
        Args:
            v:  a list of Tensor of shape ( nbatch ) 
        Returns:
            force: Tensor of shape ( nbatch )
        '''
        lmem = self.out_channels
        v_history= [ v[-i].unsqueeze(-1) for i in range(1,lmem+1)]
        v_history = th.cat(v_history,-1)
        force = (v_history * self._kernel).sum(-1)
        return force
    def forward(self, r,  v, e):
        """FORWARD CALCULATION.
        Args 
            r:  (nbatch )
            v:  (nbatch + lmem -1 )
        Returns:
            potential_force:  (nbatch )
            damping_force:  (nbatch )
        """
        potential_force = self.get_potential_force(r) # (nbatch )
        potential_force += self.get_external_force(r,e)
        if type(v) is list:
            damping_force = self.get_damping_force_parallel(v)
        else:
            damping_force = self.get_damping_force(v)
        return   potential_force, damping_force
  

class GAR_model(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, bias=True):
        super(GAR_model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ch =  10
        self.l1 = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2 = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3 = nn.Linear(self.mid_ch, out_channels, bias=bias)
        self._max_os = nn.Parameter( th.ones(1) * 1 )
        
        self.register_buffer('_linear', 1/th.arange(1, self.in_channels+1))
        self.register_buffer('weight', th.exp( - th.arange(self.in_channels)/self.in_channels*5 ) )
        self.sigma = nn.Parameter(th.ones(1)*0)

    @property 
    def linear(self):
        return self._linear.detach().cpu().numpy()

    @linear.setter
    def linear(self, x):
        if type(x) == np.ndarray:
            self._linear = th.tensor(x, dtype=self._linear.dtype, device=self._linear.device).flatten()
        else:
            self._linear = x.flatten() * 1 

    def forward(self, x_in):
        '''
        Args:
            x_in: (nbatch, lmem)
        Returns:
            predict: (nbatch)
            sigma: (nbatch)
        '''
        fr = nn.ReLU()
        f = nn.Tanh()
        # x_weighted = (x_in * self.linear[None,:])
        x_linearAR = (x_in * self._linear[None,:])
        mu = f(self.l1(x_in * self.weight[None,:]))
        mu = f(self.l2(mu))
        mu = f(self.l3(mu)) * self._max_os
        predict = mu.flatten() + x_linearAR.sum(-1) 
        sigma =  th.zeros_like(predict) + th.exp(self.sigma) 
        return predict, sigma
     