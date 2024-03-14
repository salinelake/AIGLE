from time import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv1d as conv1d
import numpy as np
from numpy import log2, pi
from utility import checkerboard

# from pytorch_memlab import profile

class poly_3Dsystem(nn.Module):
    def __init__(self,  lmem=10, onsite_only=False ):
        super(poly_3Dsystem, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.onsite_only = onsite_only
        self.lmem = lmem
        ## mass
        self.register_buffer('ml', th.ones(1)  )  
        self.register_buffer('mt', th.ones(1)  )  
        # self.register_buffer('E1', th.ones(1)  )  
        self.E1 = nn.Parameter( th.ones(1) * 10 )
        ## neighbor interaction J * tanh(wx)**2
        self.omega = nn.Parameter( th.ones(3) )
        ## onsite, same-axis
        self.al1 = nn.Parameter(  -th.ones(1) * 1   )   ## longitudinal onsite potential
        self.al2 = nn.Parameter(  th.ones(1) * 1   )   ## longitudinal onsite potential
        self.al3 = nn.Parameter(  th.ones(1) * 1   )   ## longitudinal onsite potential
        self.at1 = nn.Parameter(  th.ones(1) * 1   )   ## transverse onsite potential
        self.at2 = nn.Parameter(  th.ones(1) * 1   )   ## transverse onsite potential
        self.at3 = nn.Parameter(  th.ones(1) * 1   )   ## transverse onsite potential

        ## onsite, cross-axis
        self.gamma_xy = nn.Parameter(  th.ones(1) * 0.1   ) 
        self.gamma_xz = nn.Parameter(  th.ones(1) * 0.1   ) 

        ## nearest neighbor
        self.ax_xx = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.ax_xy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.ax_xz = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.ax_yy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.ax_yz = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.ax_zz = nn.Parameter(  th.ones(1) * 0  )  ## +

        self.az_xx = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.az_zz = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.az_xy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.az_xz = nn.Parameter(  th.ones(1) * 0  )  ## +

        ## 2rd nearest neighbor
        self.axz_xx = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axz_xy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axz_xz = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axz_yy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axz_yz = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axz_zz = nn.Parameter(  th.ones(1) * 0  )  ## +
        # self.register_buffer('axz_zz', th.ones(1) * -5 )

        self.axy_xx = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axy_xy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axy_xz = nn.Parameter(  th.ones(1) * 0  )  ## + 
        self.axy_zz = nn.Parameter(  th.ones(1) * 0  )  ## +
        
        ## 3rd nearest neighbor
        self.axyz_xx = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axyz_xy = nn.Parameter(  th.ones(1) * 0  )  ## +
        self.axyz_xz = nn.Parameter(  th.ones(1) * 0  )  ## + 
        self.axyz_zz = nn.Parameter(  th.ones(1) * 0  )  ## +  

        self.register_buffer('kos_long', - th.exp(-th.arange(lmem)/4) )
        self.register_buffer('kos_tran', - th.exp(-th.arange(lmem)/4) )
        self.register_buffer('knn_long', th.zeros(lmem) )
        self.register_buffer('knn_tran', th.zeros(lmem) )
    @property
    def p0(self):
        p0 = (-self.al1/self.al2/2)**0.5
        return p0.detach()
    
    @property
    def mass(self):
        return th.cat([self.mt,self.mt,self.ml])

    @property
    def onsite_2rd(self):
        return th.cat([self.at1, self.at1, self.al1])
    
    @property
    def onsite_4th(self):
        return  th.cat([self.at2, self.at2, self.al2])
        # return  th.exp(th.cat([self.at2, self.at2, self.al2]))
    
    @property
    def onsite_6th(self):
        return  th.cat([self.at3, self.at3, self.al3])**2
        # return  th.exp(th.cat([self.at2, self.at2, self.al2]))
 
    @property
    def Jx(self):
        l1 = th.cat([self.ax_xx, self.ax_xy, self.ax_xz])
        l2 = th.cat([self.ax_xy, self.ax_yy, self.ax_yz])
        # l3 = th.cat([self.ax_xz, self.ax_yz, - th.exp(self.ax_zz)])
        l3 = th.cat([self.ax_xz, self.ax_yz, self.ax_zz])
        return th.stack([l1,l2,l3])

    @property
    def Jy(self):
        l1 = th.cat([self.ax_yy, self.ax_xy, self.ax_yz])
        l2 = th.cat([self.ax_xy, self.ax_xx, self.ax_xz])
        # l3 = th.cat([self.ax_yz, self.ax_xz, - th.exp(self.ax_zz)])
        l3 = th.cat([self.ax_yz, self.ax_xz, self.ax_zz])
        return th.stack([l1,l2,l3])
 
    @property
    def Jz(self):
        l1 = th.cat([self.az_xx, self.az_xy, self.az_xz])
        l2 = th.cat([self.az_xy, self.az_xx, self.az_xz])
        l3 = th.cat([self.az_xz, self.az_xz,  self.az_zz ])
        # l3 = th.cat([self.az_xz, self.az_xz,  - th.exp(self.ax_zz) ])

        return th.stack([l1,l2,l3])
    
    @property
    def Jxz(self):
        l1 = th.cat([self.axz_xx, self.axz_xy, self.axz_xz])
        l2 = th.cat([self.axz_xy, self.axz_yy, self.axz_yz])
        l3 = th.cat([self.axz_xz, self.axz_yz, self.axz_zz])
        return th.stack([l1,l2,l3])
    
    @property
    def Jyz(self):
        l1 = th.cat([self.axz_yy, self.axz_xy, self.axz_yz])
        l2 = th.cat([self.axz_xy, self.axz_xx, self.axz_xz])
        l3 = th.cat([self.axz_yz, self.axz_xz, self.axz_zz])
        return th.stack([l1,l2,l3])

    @property
    def Jxy(self):
        l1 = th.cat([self.axy_xx, self.axy_xy, self.axy_xz])
        l2 = th.cat([self.axy_xy, self.axy_xx, self.axy_xz])
        l3 = th.cat([self.axy_xz, self.axy_xz, self.axy_zz])
        # l3 = th.cat([self.axy_xz, self.axy_xz, self.axz_zz])  ## symmetric force

        return th.stack([l1,l2,l3])

    @property
    def Jxyz(self):
        l1 = th.cat([self.axyz_xx, self.axyz_xy, self.axyz_xz])
        l2 = th.cat([self.axyz_xy, self.axyz_xx, self.axyz_xz])
        l3 = th.cat([self.axyz_xz, self.axyz_xz, self.axyz_zz])
        return th.stack([l1,l2,l3])

    @property
    def kosl(self):
        return self.kos_long.detach().cpu().numpy()

    @kosl.setter
    def kosl(self, x):
        if type(x) == np.ndarray:
            self.kos_long = th.tensor(x, dtype=self.kos_long.dtype, device=self.kos_long.device).flatten()
        else:
            self.kos_long = x.flatten() * 1 

    @property
    def kost(self):
        return self.kos_tran.detach().cpu().numpy()

    @kost.setter
    def kost(self, x):
        if type(x) == np.ndarray:
            self.kos_tran = th.tensor(x, dtype=self.kos_tran.dtype, device=self.kos_tran.device).flatten()
        else:
            self.kos_tran = x.flatten() * 1 

    @property 
    def knnl(self):
        return self.knn_long.detach().cpu().numpy()

    @knnl.setter
    def knnl(self, x):
        if type(x) == np.ndarray:
            self.knn_long = th.tensor(x, dtype=self.knn_long.dtype, device=self.knn_long.device).flatten()
        else:
            self.knn_long = x.flatten() * 1 
    
    def plot_meanfield_pot(self, r ):  ## mean field potential
        input_r = th.tensor(r, dtype=self.onsite_4th.dtype, device=self.onsite_4th.device).repeat(1,1,1,3,1)
        out = self.get_onsite_pot(input_r).flatten()[-1]
        return out 

    def get_onsite_pot(self, r, order= 4 ):
        '''
        Args:
            r: ##  (nx, ny, nz, 3, nframes)
        Returns:
            out: (3,nframes)
        '''
        if order == 2:
            raise NotImplementedError
        elif order == 4: ## the default choice
            ## same axis 
            out = self.onsite_4th[None, None, None, :, None] * r**4 +  self.onsite_2rd[None, None, None, :, None] * r**2
        elif order == 6:
            out = self.onsite_6th[None, None, None, :, None] * r**6 + self.onsite_4th[None, None, None, :, None] * r**4 +  self.onsite_2rd[None, None, None, :, None] * r**2
        else:   
            raise NotImplementedError
        return out.sum([0,1,2])
    
    def get_inter_pot_diagonal(self, r1, r2, J):
        '''
        Args: 
            r: tensor of shape (nx,ny,nz,3,nbatch)
        Returns:
            out: (3,nframes)
        '''
        pot =  ( (r1 - r2) )**2 * (J.diag()[None,None,None,:,None])

        return pot.sum([0,1,2] )
      
    def get_inter_pot_full(self, r1, r2, J):
        '''
        Args: 
            r: tensor of shape (nx,ny,nz,3,nbatch)
        '''
        pot = r1[...,:,None,:] * r2[...,None,:,:] * J[None,None,None,:,:,None]
        return pot.sum([0,1,2,3,4] )

    def get_potential_components(self,r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,3,nbatch)
        Returns:
            out: tensor of shape (3, nbatch)
        '''
        out = 0
        ## self energy
        out += self.get_onsite_pot( r )
        ## 1st nn
        # out += self.get_inter_pot_diagonal(r, th.roll(r,1,0), self.Jx)
        # out += self.get_inter_pot_diagonal(r, th.roll(r,1,1), self.Jy)
        # out += self.get_inter_pot_diagonal(r, th.roll(r,1,2), self.Jz)
        ## 2rd nn
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,1,0),1,1), self.Jxy)
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,-1,0),1,1), self.Jxy)
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,1,0),1,2), self.Jxz)
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,-1,0),1,2), self.Jxz)
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,1,1),1,2), self.Jyz)
        out += self.get_inter_pot_diagonal(r, th.roll(th.roll(r,-1,1),1,2), self.Jyz)
        # ## 3rd nn
        # out += self.get_inter_pot_diagonal(r, th.roll(th.roll(th.roll(r,1,0),1,1),1,2), self.Jxyz)
        # out += self.get_inter_pot_diagonal(r, th.roll(th.roll(th.roll(r,1,0),-1,1),1,2), self.Jxyz)
        # out += self.get_inter_pot_diagonal(r, th.roll(th.roll(th.roll(r,-1,0),1,1),1,2), self.Jxyz)
        # out += self.get_inter_pot_diagonal(r, th.roll(th.roll(th.roll(r,-1,0),-1,1),1,2), self.Jxyz)
        ## 4st nn
        # out += self.get_inter_pot_diagonal(r, th.roll(r,2,0), self.Jx)
        # out += self.get_inter_pot_diagonal(r, th.roll(r,2,1), self.Jy)
        # out += self.get_inter_pot_diagonal(r, th.roll(r,2,2), self.Jz)
        # out += self.get_block_pot(r)  #!
        
        return out

    def get_potential(self,r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,3,nbatch)
        Returns:
            out: tensor of shape (nbatch)
        '''
        return self.get_potential_components(r).sum(0)

    def get_potential_force(self, r):
        inputs = r.clone()
        inputs.requires_grad=True
        output = self.get_potential(inputs).sum()
        force = - th.autograd.grad(output, inputs, retain_graph=True, create_graph=True )[0]  ## (nx,ny,nz,3,nbatch)
        force /= self.mass[None,None,None,:,None]
        return force

    def get_external_force(self, e, r):
        '''
        Args:
            e: tensor of shape (nx,ny,nz,3,nbatch)
            r: tensor of shape (nx,ny,nz,3,nbatch)
        Returns:
            out: tensor of shape (nx,ny,nz,3,nbatch)
        '''
        assert e.shape == r.shape
        # coef = th.ones_like(r)
        # coef[:,:,:,-1] += self.E1
        return self.E1 * e 
 
    def get_external_pot_components(self, e,r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,3,nbatch)
        Returns:
            out: tensor of shape (3, nbatch)
        '''
        return - (self.get_external_force(e, r) * r * self.mass[None,None,None,:,None]).sum([0,1,2])

    def get_total_pot_components(self, e, r):
        return self.get_potential_components(r) + self.get_external_pot_components(e,r)
        
    def get_damping_force(self, v):
        '''
            v:  (nx,ny, nz, 3, nbatch + lmem-1 )
        '''
        nx, ny, nz, ndim, nframes = v.shape
        
        _kosl = th.flip( self.kos_long.reshape(1,1,-1), [-1])
        _kost = th.flip( self.kos_tran.reshape(1,1,-1), [-1])
        
        force_x = conv1d( v[...,0,:].reshape(-1,1,nframes), _kost)
        force_y = conv1d( v[...,1,:].reshape(-1,1,nframes), _kost)
        force_z = conv1d( v[...,2,:].reshape(-1,1,nframes), _kosl)
        force_x = force_x.reshape(nx, ny, nz, 1, -1)
        force_y = force_y.reshape(nx, ny, nz, 1, -1)
        force_z = force_z.reshape(nx, ny, nz, 1, -1)
        assert force_x.shape[-1] == nframes - self.lmem + 1
        if self.onsite_only is False:
            raise NotImplementedError
            # _knn = th.flip( self.kernel_nn.reshape(1,1,-1), [-1])
            # fnn   = conv1d(_v,  _knn).reshape(nx,ny,-1)
            # force += th.roll(fnn,1,0)
            # force += th.roll(fnn,-1,0)
            # force += th.roll(fnn,1,1)
            # force += th.roll(fnn,-1,1)
        return th.cat([force_x, force_y, force_z], 3)
    
    def get_damping_force_parallel_slow(self, v ):
        '''
        Args:
            v:  a list of Tensor of shape (nx, ny, nz, 3, nbatch) 
        Returns:
            force: Tensor of shape (nx, ny, nz, 3, nbatch)
        '''
        lmem = self.lmem
        v_history= [ v[-i].unsqueeze(-1) for i in range(1,lmem+1)]
        v_history = th.cat(v_history,-1) # (nx, ny, nz, 3, nbatch, lmem), reverse time axis
        force_x = (v_history[..., 0, :, :] * self.kos_tran).sum(-1) # (nx, ny, nz,  nbatch )
        force_y = (v_history[..., 1, :, :] * self.kos_tran).sum(-1)
        force_z = (v_history[..., 2, :, :] * self.kos_long).sum(-1)
        if self.onsite_only is False:
            raise NotImplementedError 
            # fnn   = (v_history * self.kernel_nn).sum(-1) 
            # force += th.roll(fnn,1,0)
            # force += th.roll(fnn,-1,0)
            # force += th.roll(fnn,1,1)
            # force += th.roll(fnn,-1,1)
        return th.cat([force_x.unsqueeze(-2), force_y.unsqueeze(-2), force_z.unsqueeze(-2)], -2)
    
    def get_damping_force_parallel(self, v):
        '''
        Args:
            v:  a list of Tensor of shape (nx, ny, nz, 3, nbatch) 
        Returns:
            force: Tensor of shape (nx, ny, nz, 3, nbatch)
        ''' 
        force = 0
        for i,kosl in enumerate(self.kos_long):
            kost = self.kos_tran[i]
            kos = th.stack([kost, kost, kosl], -1).reshape(1,1,1,3,1)
            force += kos * v[-i-1]
        if self.onsite_only is False:
            raise NotImplementedError 
        return force
    def forward(self, r, v, e, autograd=True):
        """FORWARD CALCULATION.
        Args 
            r:  (nx,ny, nz,3, nbatch )
            v:  (nx,ny, nz,3, nbatch + lmem-1 ) or a list of (nx,ny, nz, 3, nbatch ) 
            e:  (nx,ny, nz,3, nbatch )
        """
        nx,ny,nz,ndim,nbatch = r.shape
        gradient_force = self.get_potential_force(r)
        if autograd is False:
            gradient_force = gradient_force.detach()
        ##  damping force
        if type(v) is list:
            damping_force = self.get_damping_force_parallel(v)
        else:
            damping_force = self.get_damping_force(v)
        ### external force
        if autograd:
            external_force = self.get_external_force(e, r)
        else:
            with th.no_grad():
                external_force = self.get_external_force(e, r)
        return   gradient_force + external_force, damping_force
 
 
class poly_3Dsystem_cg(nn.Module):
    def __init__(self, model):
        super(poly_3Dsystem_cg, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.onsite_only = model.onsite_only
        self.lmem = model.lmem
        ## mass
        self.register_buffer('mass', model.ml  )  
        self.register_buffer('E1', model.E1.detach()  )  
        self.register_buffer('omega', model.omega[-1].detach()  )  
        ## onsite, same-axis
        self.register_buffer('al1', model.al1.detach()  )  
        self.register_buffer('al2', model.al2.detach()  )  
        # self.register_buffer('al3', model.al3.detach()  )  
        ## 2rd nearest neighbor
        self.register_buffer('axy_zz', model.axy_zz.detach()  )  
        self.register_buffer('axz_zz', model.axz_zz.detach()  )  
        ## kernel
        self.register_buffer('kos_long', model.kos_long )
        self.register_buffer('knn_long', model.knn_long )
        self._m1 = None
        self._m2 = None
        self._N1 = None
        self._N2 = None
    # @property
    # def mass(self):
    #     return self.ml
    @property
    def p0(self):
        p0 = (-self.al1/self.al2/2)**0.5
        return p0.detach()

    @property
    def kosl(self):
        return self.kos_long.detach().cpu().numpy()

    @kosl.setter
    def kosl(self, x):
        if type(x) == np.ndarray:
            self.kos_long = th.tensor(x, dtype=self.kos_long.dtype, device=self.kos_long.device).flatten()
        else:
            self.kos_long = x.flatten() * 1 

    @property 
    def knnl(self):
        return self.knn_long.detach().cpu().numpy()

    @knnl.setter
    def knnl(self, x):
        if type(x) == np.ndarray:
            self.knn_long = th.tensor(x, dtype=self.knn_long.dtype, device=self.knn_long.device).flatten()
        else:
            self.knn_long = x.flatten() * 1 
    
    def plot_meanfield_pot(self, r ):  ## mean field potential
        input_r = th.tensor(r, dtype=self.al1.dtype, device=self.al1.device).repeat(1,1,1,1)
        out = self.get_onsite_pot(input_r).flatten()[-1]
        return out 

    def get_onsite_pot(self, r, order= 4 ):
        '''
        Args:
            r: ##  (nx, ny, nz, nframes)
        Returns:
            out: (nframes)
        '''
        out = self.al2 * r**4 +  self.al1 * r**2
        return out.sum([0,1,2])
    
    def get_inter_pot(self, r1, r2, J):
        '''
        Args: 
            r: tensor of shape (nx,ny,nz,nbatch)
        Returns:
            out: (nframes)
        '''
        pot =  ( (r1 - r2)   )**2 * J
        return pot.sum([0,1,2] )
     
    def get_potential(self,r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,nbatch)
        Returns:
            out: tensor of shape (nbatch)
        '''
        out = 0
        ## onsite energy
        out += self.get_onsite_pot( r )
        ## 2rd nn
        out += self.get_inter_pot(r, th.roll(th.roll(r,1,0),1,1), self.axy_zz)
        out += self.get_inter_pot(r, th.roll(th.roll(r,-1,0),1,1), self.axy_zz)
        out += self.get_inter_pot(r, th.roll(th.roll(r,1,0),1,2), self.axz_zz)
        out += self.get_inter_pot(r, th.roll(th.roll(r,-1,0),1,2), self.axz_zz)
        out += self.get_inter_pot(r, th.roll(th.roll(r,1,1),1,2), self.axz_zz)
        out += self.get_inter_pot(r, th.roll(th.roll(r,-1,1),1,2), self.axz_zz)
        return out

    def get_potential_force(self, r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,nbatch)
        Returns:
            out: tensor of shape (nx,ny,nz,nbatch)
        '''
        inputs = r.clone()
        inputs.requires_grad=True
        output = self.get_potential(inputs).sum()
        force = - th.autograd.grad(output, inputs, retain_graph=True, create_graph=True )[0]  ## (nx,ny,nz,nbatch)
        force /= self.mass
        return force

    def get_potential_force_manual(self,r):
        force =  - 4 * self.al2 * r**3 - 2 *  self.al1 * r
        force -= (2 * self.axy_zz) * (2 * r - th.roll(th.roll(r,1,0),1,1) - th.roll(th.roll(r,-1,0),-1,1))
        force -= (2 * self.axy_zz) * (2 * r - th.roll(th.roll(r,-1,0),1,1) - th.roll(th.roll(r,1,0),-1,1))
        force -= (2 * self.axz_zz) * (2 * r - th.roll(th.roll(r,1,0),1,2) - th.roll(th.roll(r,-1,0),-1,2))
        force -= (2 * self.axz_zz) * (2 * r - th.roll(th.roll(r,-1,0),1,2) - th.roll(th.roll(r,1,0),-1,2))
        force -= (2 * self.axz_zz) * (2 * r - th.roll(th.roll(r,1,1),1,2) - th.roll(th.roll(r,-1,1),-1,2))
        force -= (2 * self.axz_zz) * (2 * r - th.roll(th.roll(r,-1,1),1,2) - th.roll(th.roll(r,1,1),-1,2))
        force /= self.mass
        return force

    def get_external_force(self, e, r):
        '''
        Args:
            e: tensor of shape (nx,ny,nz,nbatch)
            r: tensor of shape (nx,ny,nz,nbatch)
        Returns:
            out: tensor of shape (nx,ny,nz,nbatch)
        '''
        assert e.shape == r.shape
        # coef = th.ones_like(r)
        # coef[:,:,:,-1] += self.E1
        return self.E1 * e 
 
    def get_external_pot(self, e,r):
        '''
        Args:
            r: tensor of shape (nx,ny,nz,nbatch)
        Returns:
            out: tensor of shape (nbatch)
        '''
        return - (self.get_external_force(e, r) * r * self.mass).sum([0,1,2])
 
    def get_total_pot(self, e, r):
        return self.get_potential(r) + self.get_external_pot(e,r)

    def get_total_pot_components(self,e,r):
        return self.get_total_pot(e,r)

    def get_damping_force(self, v):
        '''
            Args: 
                v:  (nx,ny, nz,  nbatch + lmem-1 )
            returns:
                f: tensor of shape (nx,ny,nz,nbatch)
        '''
        nx, ny, nz,  nframes = v.shape
        _kosl = th.flip( self.kos_long.reshape(1,1,-1), [-1])
        force_z = conv1d( v.reshape(-1,1,nframes), _kosl)
        force_z = force_z.reshape(nx, ny, nz, -1)
        assert force_z.shape[-1] == nframes - self.lmem + 1
        if self.onsite_only is False:
            raise NotImplementedError
        return force_z
    
    def get_damping_force_parallel_slowest(self, v ):
        '''
        th.cat is very expensive..
        Args:
            v:  a list of Tensor of shape (nx, ny, nz, nbatch) 
        Returns:
            force: Tensor of shape (nx, ny, nz, nbatch)
        
        '''
        lmem = self.lmem
        v_history= [ v[-i].unsqueeze(-1) for i in range(1,lmem+1)]
        v_history = th.cat(v_history,-1) # (nx, ny, nz, nbatch, lmem), reverse time axis
        force_z = (v_history * self.kos_long).sum(-1)  # (nx, ny, nz,  nbatch )
        return force_z
    
    def get_damping_force_parallel(self, v ):
        '''
        Args:
            v:  a list of Tensor of shape (nx, ny, nz, nbatch) 
        Returns:
            force: Tensor of shape (nx, ny, nz, nbatch)
        '''
        force_z = 0
        for i,kos in enumerate(self.kos_long):
            force_z += kos * v[-i-1]
        return force_z
 
    # @profile
    def forward(self, r, v, e, autograd=False):
        """FORWARD CALCULATION.
        Args 
            r:  (nx,ny, nz, nbatch )
            v:  (nx,ny, nz, nbatch + lmem-1 ) or a list of (nx,ny, nz, nbatch ) 
            e:  (nx,ny, nz, nbatch )
        """
        nx,ny,nz,nbatch = r.shape

        # ## gradient force, not including external force
        # gradient_force = self.get_potential_force(r)
        # if autograd is False:
        #     gradient_force = gradient_force.detach()
        
        if autograd:
            gradient_force = self.get_potential_force_manual(r)
        else:
            with th.no_grad():
                gradient_force = self.get_potential_force_manual(r)

        ##  damping force
        if type(v) is list:
            damping_force = self.get_damping_force_parallel(v)
        else:
            damping_force = self.get_damping_force(v)

        ### external force
        if autograd:
            external_force = self.get_external_force(e, r)
        else:
            with th.no_grad():
                external_force = self.get_external_force(e, r)
        return   gradient_force + external_force, damping_force
    
    def get_sublatts_filter(self, x):
        nx, ny, nz, nbatch = x.shape  ## assert ndim=4
        _m1 = checkerboard(x.shape[:3])
        _m1 = th.tensor(_m1, dtype=x.dtype, device=x.device)
        _m1 = _m1.unsqueeze(-1)
        _m2 = 1 - _m1
        return _m1, _m2
    
    def average_over_sublatts(self, x):
        '''
        average a quantity over sublattice 1 and sublattice 2 respectively
        '''
        _m1, _m2 = self.get_sublatts_filter(x)
        x_latt1 = (x * _m1).sum([0,1,2]) / _m1.sum() 
        x_latt2 = (x * _m2).sum([0,1,2]) / _m2.sum() 
        return x_latt1, x_latt2

    # def forward_biased(self, r, v, e, bias_K, bias_x):
    #     """FORWARD CALCULATION.
    #     x = r.mean([0,1,2])
    #     biased by  potential U = 0.5 * bias_K * (x-bias_x)^2  -> F= - bias_K/m * (x-bias_x) /nx/ny/nz
    #     Args 
    #         r:  (nx,ny, nz, nbatch )
    #         v:  (nx,ny, nz, nbatch + lmem-1 ) or a list of (nx,ny, nz, nbatch ) 
    #         e:  (nx,ny, nz, nbatch )
    #     """
    #     unbiased_force, damping_force = self.forward(r,v,e, autograd=False)
        
    #     inputs = r.clone()
    #     inputs.requires_grad=True
    #     x1, x2 = self.average_over_sublatts(inputs)
    #     bias_pot = ( (x1 - bias_x)**2 + (x2 - bias_x)**2 ).sum()

    #     bias_force = th.autograd.grad(bias_pot, inputs, retain_graph=True, create_graph=True )[0]  ## (nx,ny,nz,nbatch)
    #     bias_force = bias_force.detach()
    #     bias_force *= - 0.5 * bias_K / self.mass

    #     return   unbiased_force + bias_force, damping_force
    def get_biasing_force(self, r, bias_K, bias_x):
        if self._m1 is None:
            _m1, _m2 = self.get_sublatts_filter(r)
            _m1 = _m1.squeeze(-1)
            _m2 = _m2.squeeze(-1)
            self._N1 = _m1.sum()
            self._N2 = _m2.sum()
            self._m1 = _m1.to(dtype=th.bool)
            self._m2 = _m2.to(dtype=th.bool)
        nx, ny, nz, nbatch = r.shape  ## assert ndim=4
        x_latt1 = r[self._m1].mean(0) 
        x_latt2 = r[self._m2].mean(0) 
        bias_force = th.zeros_like(r)
        bias_force[self._m1] += (x_latt1 - bias_x)[None,:] / self._N1
        bias_force[self._m2] += (x_latt2 - bias_x)[None,:] / self._N2
        bias_force *= - bias_K / self.mass
        return bias_force

    def get_gradient_force_biased(self, r, e, bias_K, bias_x):
        with th.no_grad():
            unbiased_force = self.get_potential_force_manual(r)
            external_force = self.get_external_force(e, r)
            bias_force = self.get_biasing_force(r, bias_K, bias_x)
        return   unbiased_force + bias_force + external_force

class FNN_noise(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, bias=False, onsite_only=False):
        super(FNN_noise, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.onsite_only = onsite_only
        self.mid_ch =  12
        ## transverse noise
        self.l1t = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2t = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3t = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ## longitudinal noise
        self.l1l = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2l = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3l = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ## longitudinal (z) nearest neighbor  noise
        self.l1l_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2l_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3l_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)        
        ## transverse (x/y) nearest neighbor noise
        self.l1t_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2t_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3t_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)     

        self.epsilon = 1e-3
        self.register_buffer('ag_os_long', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_os_tran', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_nn',  th.zeros(self.in_channels) )
        self.register_buffer('weight', th.exp( - th.arange(self.in_channels)/self.in_channels*5 ) )
        self.sigma_t = nn.Parameter( th.ones(1) * 4 )
        self.sigma_l = nn.Parameter( th.ones(1) * 4 )

        self._max_os = nn.Parameter( th.ones(1) * 4 )
        self._max_nn = nn.Parameter( th.ones(1) * 2 )

    @property
    def sigma(self):
        return th.exp(th.cat([self.sigma_t, self.sigma_t, self.sigma_l]))

    @property
    def max_os(self):
        return th.exp(self._max_os)
    @property
    def max_nn(self):
        return th.exp(self._max_nn)

    @property 
    def yw_osl(self):
        return self.ag_os_long.detach().cpu().numpy()

    @yw_osl.setter
    def yw_osl(self, x):
        if type(x) == np.ndarray:
            self.ag_os_long = th.tensor(x, dtype=self.ag_os_long.dtype, device=self.ag_os_long.device).flatten()
        else:
            self.ag_os_long = x.flatten() * 1 

    @property 
    def yw_ost(self):
        return self.ag_os_tran.detach().cpu().numpy()

    @yw_ost.setter
    def yw_ost(self, x):
        if type(x) == np.ndarray:
            self.ag_os_tran = th.tensor(x, dtype=self.ag_os_tran.dtype, device=self.ag_os_tran.device).flatten()
        else:
            self.ag_os_tran = x.flatten() * 1 

    @property 
    def yw_nn(self):
        return self.ag_nn.detach().cpu().numpy()

    @yw_nn.setter
    def yw_nn(self, x):
        if type(x) == np.ndarray:
            self.ag_nn = th.tensor(x, dtype=self.ag_nn.dtype, device=self.ag_nn.device).flatten()
        else:
            self.ag_nn = x.flatten() * 1 


    def get_AR_long(self, x):
        noise = (x * self.ag_os_long).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_AR_tran(self, x):
        noise = (x * self.ag_os_tran).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_neural_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t(x))
        x = f(self.l2t(x))
        x = f(self.l3t(x)) * self.max_os
        return x
    
    def get_neural_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l(x))
        x = f(self.l2l(x))
        x = f(self.l3l(x)) * self.max_os
        return x

    def get_neural_nn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l_nn(x))
        x = f(self.l2l_nn(x))
        x = f(self.l3l_nn(x)) * self.max_os
        return x

    def get_neural_nn_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t_nn(x))
        x = f(self.l2t_nn(x))
        x = f(self.l3t_nn(x)) * self.max_os
        return x
 
    def forward(self, r_in, x_in):   
        '''
        Args:
            r:       (nx, ny, nz, 3, nbatch)
            x_in:    (nx, ny, nz, 3, nbatch, lmem)
        Returns:
            predict: (nx, ny, nz, 3, nbatch )
            sigma:   (nx, ny, nz, nbatch )
        '''
        nx, ny, nz, _, nbatch, lmem = x_in.shape
        assert lmem == self.in_channels
        
        ## get onsite part
        x_os_x = (x_in[...,0,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_x = self.get_neural_tran(x_os_x).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_x += self.get_AR_tran(x_in[...,0,:,:])

        x_os_y = (x_in[...,1,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_y = self.get_neural_tran(x_os_y).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_y += self.get_AR_tran(x_in[...,1,:,:])

        x_os_z = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_z = self.get_neural_long(x_os_z).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_z += self.get_AR_long(x_in[...,2,:,:])
        
        x_nn_l = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem) 
        x_nn_l = self.get_neural_nn_long(x_nn_l).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)

        # x_nn_t = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem)
        # x_nn_t = self.get_neural_nn_tran(x_nn_t).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        
        # x_os_z += (
        #     + th.roll(x_nn_l, 1, 2) + th.roll(x_nn_l, -1, 2) 
        #     + th.roll(x_nn_t, 1, 0) + th.roll(x_nn_t, -1, 0) 
        #     + th.roll(x_nn_t, 1, 1) + th.roll(x_nn_t, -1, 1)
        #             ) / 6

        x_os_z += (th.roll(th.roll(x_nn_l, 1,0),1,2) + th.roll(th.roll(x_nn_l, 1,0),-1,2)
                + th.roll(th.roll(x_nn_l,-1,0),1,2) + th.roll(th.roll(x_nn_l,-1,0),-1,2)
                + th.roll(th.roll(x_nn_l, 1,1),1,2) + th.roll(th.roll(x_nn_l, 1,1),-1,2)
                + th.roll(th.roll(x_nn_l,-1,1),1,2) + th.roll(th.roll(x_nn_l,-1,1),-1,2))/8
 
        x_noise = th.cat([x_os_x[...,None,:], x_os_y[...,None,:], x_os_z[...,None,:]],-2) # (nx, ny, nz, 3, nbatch)
        x_sigma = self.sigma[None,None,None,:,None]
        return x_noise, x_sigma


class FNN_noise_nnn(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, bias=False, onsite_only=False):
        super(FNN_noise_nnn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.onsite_only = onsite_only
        self.mid_ch =  12
        ## transverse noise
        self.l1t = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2t = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3t = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ## longitudinal noise
        self.l1l = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2l = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3l = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ## longitudinal (z) nearest neighbor  noise
        self.l1l_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2l_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3l_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)        
        ## longitudinal (z) next nearest neighbor  noise
        self.l1ll_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2ll_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3ll_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)        
        ## transverse (x/y) nearest neighbor noise
        self.l1t_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2t_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3t_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)     

        self.epsilon = 1e-3
        self.register_buffer('ag_os_long', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_os_tran', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_nn',  th.zeros(self.in_channels) )
        self.register_buffer('weight', th.exp( - th.arange(self.in_channels)/self.in_channels*5 ) )
        self.sigma_t = nn.Parameter( th.ones(1) * 4 )
        self.sigma_l = nn.Parameter( th.ones(1) * 4 )

        self._max_os = nn.Parameter( th.ones(1) * 4 )
        self._max_nn = nn.Parameter( th.ones(1) * 2 )

    @property
    def sigma(self):
        return th.exp(th.cat([self.sigma_t, self.sigma_t, self.sigma_l]))

    @property
    def max_os(self):
        return th.exp(self._max_os)
    @property
    def max_nn(self):
        return th.exp(self._max_nn)

    @property 
    def yw_osl(self):
        return self.ag_os_long.detach().cpu().numpy()

    @yw_osl.setter
    def yw_osl(self, x):
        if type(x) == np.ndarray:
            self.ag_os_long = th.tensor(x, dtype=self.ag_os_long.dtype, device=self.ag_os_long.device).flatten()
        else:
            self.ag_os_long = x.flatten() * 1 

    @property 
    def yw_ost(self):
        return self.ag_os_tran.detach().cpu().numpy()

    @yw_ost.setter
    def yw_ost(self, x):
        if type(x) == np.ndarray:
            self.ag_os_tran = th.tensor(x, dtype=self.ag_os_tran.dtype, device=self.ag_os_tran.device).flatten()
        else:
            self.ag_os_tran = x.flatten() * 1 

    @property 
    def yw_nn(self):
        return self.ag_nn.detach().cpu().numpy()

    @yw_nn.setter
    def yw_nn(self, x):
        if type(x) == np.ndarray:
            self.ag_nn = th.tensor(x, dtype=self.ag_nn.dtype, device=self.ag_nn.device).flatten()
        else:
            self.ag_nn = x.flatten() * 1 


    def get_AR_long(self, x):
        noise = (x * self.ag_os_long).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_AR_tran(self, x):
        noise = (x * self.ag_os_tran).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_neural_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t(x))
        x = f(self.l2t(x))
        x = f(self.l3t(x)) * self.max_os
        return x
    
    def get_neural_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l(x))
        x = f(self.l2l(x))
        x = f(self.l3l(x)) * self.max_os
        return x

    def get_neural_nn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l_nn(x))
        x = f(self.l2l_nn(x))
        x = f(self.l3l_nn(x)) * self.max_os
        return x

    def get_neural_nnn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1ll_nn(x))
        x = f(self.l2ll_nn(x))
        x = f(self.l3ll_nn(x)) * self.max_nn
        return x

    def get_neural_nn_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t_nn(x))
        x = f(self.l2t_nn(x))
        x = f(self.l3t_nn(x)) * self.max_os
        return x
 
    def forward(self, r_in, x_in):   
        '''
        Args:
            r:       (nx, ny, nz, 3, nbatch)
            x_in:    (nx, ny, nz, 3, nbatch, lmem)
        Returns:
            predict: (nx, ny, nz, 3, nbatch )
            sigma:   (nx, ny, nz, nbatch )
        '''
        nx, ny, nz, _, nbatch, lmem = x_in.shape
        assert lmem == self.in_channels
        
        ## get onsite part
        x_os_x = (x_in[...,0,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_x = self.get_neural_tran(x_os_x).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_x += self.get_AR_tran(x_in[...,0,:,:])

        x_os_y = (x_in[...,1,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_y = self.get_neural_tran(x_os_y).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_y += self.get_AR_tran(x_in[...,1,:,:])

        x_os_z = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_z = self.get_neural_long(x_os_z).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_z += self.get_AR_long(x_in[...,2,:,:])
        
        x_nn_l = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem) 
        x_nn_l = self.get_neural_nn_long(x_nn_l).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)

        x_nn_ll = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem) 
        x_nn_ll = self.get_neural_nnn_long(x_nn_ll).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        # x_nn_t = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem)
        # x_nn_t = self.get_neural_nn_tran(x_nn_t).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        
        # x_os_z += (
        #     + th.roll(x_nn_l, 1, 2) + th.roll(x_nn_l, -1, 2) 
        #     + th.roll(x_nn_t, 1, 0) + th.roll(x_nn_t, -1, 0) 
        #     + th.roll(x_nn_t, 1, 1) + th.roll(x_nn_t, -1, 1)
        #             ) / 6

        x_os_z += (th.roll(th.roll(x_nn_l, 1,0),1,2) + th.roll(th.roll(x_nn_l, 1,0),-1,2)
                + th.roll(th.roll(x_nn_l,-1,0),1,2) + th.roll(th.roll(x_nn_l,-1,0),-1,2)
                + th.roll(th.roll(x_nn_l, 1,1),1,2) + th.roll(th.roll(x_nn_l, 1,1),-1,2)
                + th.roll(th.roll(x_nn_l,-1,1),1,2) + th.roll(th.roll(x_nn_l,-1,1),-1,2))/8
        ## add also nnn contribution
        x_os_z += (th.roll(x_nn_ll, 2,2) + th.roll(x_nn_ll, -2,2))/2
        ## get noise
        x_noise = th.cat([x_os_x[...,None,:], x_os_y[...,None,:], x_os_z[...,None,:]],-2) # (nx, ny, nz, 3, nbatch)
        x_sigma = self.sigma[None,None,None,:,None]
        return x_noise, x_sigma


class FNN_noise_additive(nn.Module):
    def __init__(self, in_channels=20, out_channels=1, bias=False, onsite_only=False):
        super(FNN_noise_additive, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.onsite_only = onsite_only
        self.mid_ch =  12
        ## transverse noise
        self.l1t = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2t = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3t = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ## longitudinal noise
        self.l1l = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        self.l2l = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        self.l3l = nn.Linear( self.mid_ch, out_channels, bias=bias)
        ##
        self.register_buffer('ad1',  th.zeros(1) )
        self.register_buffer('ad2',  th.zeros(1) )


        # ## longitudinal (z) nearest neighbor  noise
        # self.l1l_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        # self.l2l_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        # self.l3l_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)        
        # ## transverse (x/y) nearest neighbor noise
        # self.l1t_nn = nn.Linear( self.in_channels, self.mid_ch,  bias=bias)
        # self.l2t_nn = nn.Linear( self.mid_ch, self.mid_ch,  bias=bias)
        # self.l3t_nn = nn.Linear( self.mid_ch, out_channels, bias=bias)     

        self.epsilon = 1e-3
        self.register_buffer('ag_os_long', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_os_tran', 1/th.arange(1, self.in_channels+1) )
        self.register_buffer('ag_nn',  th.zeros(self.in_channels) )
        self.register_buffer('weight', th.exp( - th.arange(self.in_channels)/self.in_channels*5 ) )
        self.sigma_t = nn.Parameter( th.ones(1) * 4 )
        self.sigma_l = nn.Parameter( th.ones(1) * 4 )

        self._max_os = nn.Parameter( th.ones(1) * 4 )
        self._max_nn = nn.Parameter( th.ones(1) * 2 )

    @property
    def sigma(self):
        return th.exp(th.cat([self.sigma_t, self.sigma_t, self.sigma_l]))

    @property
    def max_os(self):
        return th.exp(self._max_os)
    @property
    def max_nn(self):
        return th.exp(self._max_nn)

    @property 
    def yw_osl(self):
        return self.ag_os_long.detach().cpu().numpy()

    @yw_osl.setter
    def yw_osl(self, x):
        if type(x) == np.ndarray:
            self.ag_os_long = th.tensor(x, dtype=self.ag_os_long.dtype, device=self.ag_os_long.device).flatten()
        else:
            self.ag_os_long = x.flatten() * 1 

    @property 
    def yw_ost(self):
        return self.ag_os_tran.detach().cpu().numpy()

    @yw_ost.setter
    def yw_ost(self, x):
        if type(x) == np.ndarray:
            self.ag_os_tran = th.tensor(x, dtype=self.ag_os_tran.dtype, device=self.ag_os_tran.device).flatten()
        else:
            self.ag_os_tran = x.flatten() * 1 

    @property 
    def yw_nn(self):
        return self.ag_nn.detach().cpu().numpy()

    @yw_nn.setter
    def yw_nn(self, x):
        if type(x) == np.ndarray:
            self.ag_nn = th.tensor(x, dtype=self.ag_nn.dtype, device=self.ag_nn.device).flatten()
        else:
            self.ag_nn = x.flatten() * 1 


    def get_AR_long(self, x):
        noise = (x * self.ag_os_long).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_AR_tran(self, x):
        noise = (x * self.ag_os_tran).sum(-1)
        # if self.onsite_only is False:
        #     noise_nn = (x * self.ag_nn).sum(-1)
        #     noise += th.roll(noise_nn,1,0)
        #     noise += th.roll(noise_nn,-1,0)
        #     noise += th.roll(noise_nn,1,1)
        #     noise += th.roll(noise_nn,-1,1)
        return noise
    
    def get_neural_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t(x))
        x = f(self.l2t(x))
        x = f(self.l3t(x)) * self.max_os
        return x
    
    def get_neural_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l(x))
        x = f(self.l2l(x))
        x = f(self.l3l(x)) * self.max_os
        return x

    def get_neural_nn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l_nn(x))
        x = f(self.l2l_nn(x))
        x = f(self.l3l_nn(x)) * self.max_os
        return x

    def get_neural_nn_tran(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1t_nn(x))
        x = f(self.l2t_nn(x))
        x = f(self.l3t_nn(x)) * self.max_os
        return x

    def forward_white_noise_field(self, x):
        '''
        Args:
            x: (nx, ny, nz, 3, nbatch)
        '''
        nx, ny, nz, _, nbatch = x.shape
        base = th.randn_like(x) * self.sigma[None,None,None,:,None].detach()
        _noise_z = base[:,:,:,[2],:]

        noise_z =   _noise_z + self.ad1 * (
                    + th.roll(th.roll(_noise_z, 1,0),1,2) + th.roll(th.roll(_noise_z, 1,0),-1,2)
                    + th.roll(th.roll(_noise_z,-1,0),1,2) + th.roll(th.roll(_noise_z,-1,0),-1,2)
                    + th.roll(th.roll(_noise_z, 1,1),1,2) + th.roll(th.roll(_noise_z, 1,1),-1,2)
                    + th.roll(th.roll(_noise_z,-1,1),1,2) + th.roll(th.roll(_noise_z,-1,1),-1,2)
                )  + self.ad2 * (
                    + th.roll(_noise_z, 2, 2) + th.roll(_noise_z, -2, 2)
                )
        noise = th.cat([base[:,:,:,[0],:], base[:,:,:,[1],:], noise_z],-2) # (nx, ny, nz, 3, nbatch)
        return noise


    def forward(self, r_in, x_in):   
        '''
        Args:
            r:       (nx, ny, nz, 3, nbatch)
            x_in:    (nx, ny, nz, 3, nbatch, lmem)
        Returns:
            predict: (nx, ny, nz, 3, nbatch )
            sigma:   (nx, ny, nz, nbatch )
        '''
        nx, ny, nz, _, nbatch, lmem = x_in.shape
        assert lmem == self.in_channels
        
        ## get onsite part
        x_os_x = (x_in[...,0,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_x = self.get_neural_tran(x_os_x).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_x += self.get_AR_tran(x_in[...,0,:,:])

        x_os_y = (x_in[...,1,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os_y = self.get_neural_tran(x_os_y).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os_y += self.get_AR_tran(x_in[...,1,:,:])

        r_os_z = (x_in[...,2,:,:] * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        r_os_z = self.get_neural_long(r_os_z).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        r_os_z += self.get_AR_long(x_in[...,2,:,:])
        
        x_os_z =  r_os_z + self.ad1 * (
                    + th.roll(th.roll(r_os_z, 1,0),1,2) + th.roll(th.roll(r_os_z, 1,0),-1,2)
                    + th.roll(th.roll(r_os_z,-1,0),1,2) + th.roll(th.roll(r_os_z,-1,0),-1,2)
                    + th.roll(th.roll(r_os_z, 1,1),1,2) + th.roll(th.roll(r_os_z, 1,1),-1,2)
                    + th.roll(th.roll(r_os_z,-1,1),1,2) + th.roll(th.roll(r_os_z,-1,1),-1,2)
                )  + self.ad2 * (
                    + th.roll(r_os_z, 2, 2) + th.roll(r_os_z, -2, 2)
                )
 
        x_noise = th.cat([x_os_x[...,None,:], x_os_y[...,None,:], x_os_z[...,None,:]],-2) # (nx, ny, nz, 3, nbatch)
        
        ## get the std of noise at each site
        adz = (1 + self.ad1**2 * 8 + self.ad2**2 * 2)**0.5
        x_sigma = th.tensor([0,0,1], dtype=x_noise.dtype, device=x_noise.device)
        x_sigma = x_sigma * (adz-1) + 1
        x_sigma = self.sigma * x_sigma
        x_sigma = x_sigma[None,None,None,:,None]
        return x_noise, x_sigma


class FNN_noise_cg(nn.Module):
    def __init__(self, model):
        super(FNN_noise_cg, self).__init__()
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.onsite_only = model.onsite_only
        self.mid_ch =  model.mid_ch
        ## longitudinal noise
        self.l1l = model.l1l
        self.l2l = model.l2l
        self.l3l =  model.l3l
        ## longitudinal (z) nearest neighbor  noise
        self.l1l_nn = model.l1l_nn
        self.l2l_nn = model.l2l_nn
        self.l3l_nn = model.l3l_nn       
        
        self.register_buffer('ag_os_long', model.ag_os_long.detach() )
        self.register_buffer('ag_nn',  model.ag_nn.detach() )
        self.register_buffer('weight', model.weight.detach() )
        self.register_buffer('sigma_l', model.sigma_l.detach()  )  
        self.register_buffer('_max_os', model._max_os.detach()  )  
        self.register_buffer('_max_nn', model._max_nn.detach()  )  

    @property
    def sigma(self):
        return th.exp(self.sigma_l)

    @property
    def max_os(self):
        return th.exp(self._max_os)
    @property
    def max_nn(self):
        return th.exp(self._max_nn)

    @property 
    def yw_osl(self):
        return self.ag_os_long.detach().cpu().numpy()

    @yw_osl.setter
    def yw_osl(self, x):
        if type(x) == np.ndarray:
            self.ag_os_long = th.tensor(x, dtype=self.ag_os_long.dtype, device=self.ag_os_long.device).flatten()
        else:
            self.ag_os_long = x.flatten() * 1 
 
    @property 
    def yw_nn(self):
        return self.ag_nn.detach().cpu().numpy()

    @yw_nn.setter
    def yw_nn(self, x):
        if type(x) == np.ndarray:
            self.ag_nn = th.tensor(x, dtype=self.ag_nn.dtype, device=self.ag_nn.device).flatten()
        else:
            self.ag_nn = x.flatten() * 1 


    def get_AR_long(self, x):
        noise = (x * self.ag_os_long).sum(-1)
        return noise
     
    def get_neural_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l(x))
        x = f(self.l2l(x))
        x = f(self.l3l(x)) * self.max_os
        return x

    def get_neural_nn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l_nn(x))
        x = f(self.l2l_nn(x))
        x = f(self.l3l_nn(x)) * self.max_os
        return x

    # @profile
    def forward(self, x_in):   
        '''
        Args:
            x_in:    (nx, ny, nz,  nbatch, lmem)
        Returns:
            predict: (nx, ny, nz, nbatch )
            sigma:   (nx, ny, nz, nbatch )
        '''
        nx, ny, nz, nbatch, lmem = x_in.shape
        assert lmem == self.in_channels
        ## get onsite part
        x_os = (x_in * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os = self.get_neural_long(x_os).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os += self.get_AR_long(x_in)
        
        x_nn = (x_in * self.weight).reshape(-1, lmem) 
        x_nn = self.get_neural_nn_long(x_nn).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)

        x_os += (th.roll(th.roll(x_nn, 1,0),1,2) + th.roll(th.roll(x_nn, 1,0),-1,2)
                + th.roll(th.roll(x_nn,-1,0),1,2) + th.roll(th.roll(x_nn,-1,0),-1,2)
                + th.roll(th.roll(x_nn, 1,1),1,2) + th.roll(th.roll(x_nn, 1,1),-1,2)
                + th.roll(th.roll(x_nn,-1,1),1,2) + th.roll(th.roll(x_nn,-1,1),-1,2))/8
 
        x_sigma = th.zeros_like(x_os) + self.sigma
        return x_os, x_sigma


class FNN_noise_nnn_cg(nn.Module):
    def __init__(self, model):
        super(FNN_noise_nnn_cg, self).__init__()
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels
        self.onsite_only = model.onsite_only
        self.mid_ch =  model.mid_ch
        ## longitudinal noise
        self.l1l = model.l1l
        self.l2l = model.l2l
        self.l3l =  model.l3l
        ## longitudinal (z) nearest neighbor  noise
        self.l1l_nn = model.l1l_nn
        self.l2l_nn = model.l2l_nn
        self.l3l_nn = model.l3l_nn       
        ## longitudinal (z) next nearest neighbor  noise
        self.l1ll_nn = model.l1ll_nn
        self.l2ll_nn = model.l2ll_nn
        self.l3ll_nn = model.l3ll_nn       

        self.register_buffer('ag_os_long', model.ag_os_long.detach() )
        self.register_buffer('ag_nn',  model.ag_nn.detach() )
        self.register_buffer('weight', model.weight.detach() )
        self.register_buffer('sigma_l', model.sigma_l.detach()  )  
        self.register_buffer('_max_os', model._max_os.detach()  )  
        self.register_buffer('_max_nn', model._max_nn.detach()  )  

    @property
    def sigma(self):
        return th.exp(self.sigma_l)

    @property
    def max_os(self):
        return th.exp(self._max_os)
    @property
    def max_nn(self):
        return th.exp(self._max_nn)

    @property 
    def yw_osl(self):
        return self.ag_os_long.detach().cpu().numpy()

    @yw_osl.setter
    def yw_osl(self, x):
        if type(x) == np.ndarray:
            self.ag_os_long = th.tensor(x, dtype=self.ag_os_long.dtype, device=self.ag_os_long.device).flatten()
        else:
            self.ag_os_long = x.flatten() * 1 
 
    @property 
    def yw_nn(self):
        return self.ag_nn.detach().cpu().numpy()

    @yw_nn.setter
    def yw_nn(self, x):
        if type(x) == np.ndarray:
            self.ag_nn = th.tensor(x, dtype=self.ag_nn.dtype, device=self.ag_nn.device).flatten()
        else:
            self.ag_nn = x.flatten() * 1 


    def get_AR_long(self, x):
        noise = (x * self.ag_os_long).sum(-1)
        return noise
     
    def get_neural_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l(x))
        x = f(self.l2l(x))
        x = f(self.l3l(x)) * self.max_os
        return x

    def get_neural_nn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1l_nn(x))
        x = f(self.l2l_nn(x))
        x = f(self.l3l_nn(x)) * self.max_os
        return x

    def get_neural_nnn_long(self,x ):
        '''
        Args:
            x: (nbatch, lmem)
        '''
        f  = nn.Tanh()
        x = f(self.l1ll_nn(x))
        x = f(self.l2ll_nn(x))
        x = f(self.l3ll_nn(x)) * self.max_nn
        return x
    
    # @profile
    def forward(self, x_in):   
        '''
        Args:
            x_in:    (nx, ny, nz,  nbatch, lmem)
        Returns:
            predict: (nx, ny, nz, nbatch )
            sigma:   (nx, ny, nz, nbatch )
        '''
        nx, ny, nz, nbatch, lmem = x_in.shape
        assert lmem == self.in_channels
        ## get onsite part
        x_os = (x_in * self.weight).reshape(-1, lmem)  # (nx*ny*nz*nbatch, lmem)
        x_os = self.get_neural_long(x_os).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)
        x_os += self.get_AR_long(x_in)
        
        x_nn = (x_in * self.weight).reshape(-1, lmem) 
        x_nn = self.get_neural_nn_long(x_nn).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)

        x_nn_ll = (x_in * self.weight).reshape(-1, lmem) 
        x_nn_ll = self.get_neural_nnn_long(x_nn_ll).reshape(nx,ny,nz,-1)  # (nx, ny, nz, nbatch)

        x_os += (th.roll(th.roll(x_nn, 1,0),1,2) + th.roll(th.roll(x_nn, 1,0),-1,2)
                + th.roll(th.roll(x_nn,-1,0),1,2) + th.roll(th.roll(x_nn,-1,0),-1,2)
                + th.roll(th.roll(x_nn, 1,1),1,2) + th.roll(th.roll(x_nn, 1,1),-1,2)
                + th.roll(th.roll(x_nn,-1,1),1,2) + th.roll(th.roll(x_nn,-1,1),-1,2))/8
        x_os += (th.roll(x_nn_ll, 2,2) + th.roll(x_nn_ll, -2,2))/2
 
        x_sigma = th.zeros_like(x_os) + self.sigma
        return x_os, x_sigma
