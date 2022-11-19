import torch as th
import torch.nn as nn
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
        self.register_buffer('kernel', -  0*th.exp(-th.arange(out_channels)/4))
        # self.kernel = nn.Parameter(- th.exp(-th.arange(out_channels)/4))
    
    def get_freq(self ):
        return self.well_freq * self.well_freq_shift * 2 * pi  # 1D

    def get_phase(self):
        return self.get_freq()* self.well_bias  # 1D

    def get_argument(self ,r):
        return self.get_freq() * r + self.get_phase()

    def get_potential(self, r):
        arg = self.get_argument(r)
        potential = self.well_coef * th.tanh(self.rescale * (1-th.cos(arg)))
        return potential

    def get_potential_force(self, r):
        arg1 = self.get_argument(r)
        tanhz = th.tanh(self.rescale * (1-th.cos(arg1)))
        potential_force = - self.well_coef * (1- tanhz**2) * self.rescale * self.get_freq() * th.sin(arg1)
        return potential_force
    
    def get_external_force(self,r, e ):
        prefactor = self.E_linear  
        return prefactor * e

    def forward(self, r,  v, e):
        """FORWARD CALCULATION.
        Args 
            r:  (nbatch, 1)
            v:  (nbatch, nmem,  1)
        Returns: 
            elastic_force: (nbatch, 1)
        """
        potential_force = self.get_potential_force(r) # (nbatch, 1)
        potential_force += self.get_external_force(r,e)
        damp_force  =   (v * self.kernel[None,:,None]).sum(1)
        return   potential_force, damp_force
  

class GAR_model(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, bias=True):
        super(GAR_model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ch =  10
        self.l1 = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2 = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3 = nn.Linear(self.mid_ch, out_channels, bias=bias)
        
        self.epsilon = 1e-3
        self.register_buffer('linear', 1/th.arange(1, self.in_channels+1))
        self.register_buffer('weight', th.exp( - th.arange(self.in_channels)/self.in_channels*5 ) )
        # self.linear_sigma = nn.Parameter(th.ones(1))
        # self.linear = nn.Parameter( 1/th.arange(1, self.in_channels+1) )
        self.sigma = nn.Parameter(th.ones(1)*0)

    def forward(self, x_in):
        '''
        Args:
            x_in: (nbatch, lmem)
        '''
        fr = nn.ReLU()
        f = nn.Tanh()
        # x_weighted = (x_in * self.linear[None,:])
        x_linearAR = (x_in * self.linear[None,:])
        x = f(self.l1(x_in * self.weight[None,:]))
        x = f(self.l2(x))
        x = self.l3(x)
        predict = x + x_linearAR.sum(-1)[:,None]
        sigma =  th.zeros_like(x) + th.exp(self.sigma) 
        return predict, sigma
     