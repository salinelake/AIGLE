import os
import numpy as np
from numpy.linalg import lstsq
import torch as th
from torch.optim.lr_scheduler import StepLR
from model import *
from utility import read_out_smooth
np.set_printoptions(3, suppress=True)

def get_multiv_data( dataset, dt, step ):
    v = (dataset['r'][1:] - dataset['r'][:-1]) / dt
    input_v = [v[step:]*1]
    input_v += [v[step-i:-i]*1 for i in np.arange(1,step)]
    input_v = th.cat([v[:,None,:] for v in input_v], 1)
    return {
                'v':   input_v,
                'plain_v': dataset['v'][1+step:],
                'r':   dataset['r'][1+step:] ,
                'a':   dataset['a'][1+step:] ,
                'e':   dataset['e'][1+step:] ,
            }

def get_noise( model_noise, noise ):
    lmem = model_noise.in_channels
    noise_ref = noise[lmem:] ## (nframe, height)
    noise_traj = []
    for i in range(1,lmem+1):
        noise_traj.append(noise[lmem-i:-i,:,None])
    noise_traj = th.cat(noise_traj,-1) ##  (nframe, height, lmem)
    noise_pred, noise_sigma = model_noise(noise_traj.reshape(-1,lmem))
    noise_pred = noise_pred.reshape(noise_ref.shape)
    noise_sigma = noise_sigma.reshape(noise_ref.shape)
    return noise_pred, noise_ref, noise_sigma
 
def train_eq( model_ext, model_noise,  trainset, trainset_paras, 
                        epoch=8010 , print_freq=10, save_freq=1000,  model_folder='model'):
    dev = 'cuda' if th.cuda.is_available() else 'cpu'
    dt = trainset_paras['dt']
    batchsize = trainset_paras['batchsize']
    lmem = trainset_paras['lmem']
    optimizer  = th.optim.Adam( list(model_ext.parameters()) + list(model_noise.parameters()), lr = 0.01 )
    scheduler = StepLR(optimizer, step_size=500, gamma=0.9)
    for i in range(epoch):
        loss = 0
        corr_uv = 0
        corr_nn = 0
        corr_vvplus = 0
        setidx = np.random.choice(len(trainset), size=batchsize, replace=False)
        for j in range(batchsize):
            #################### LOADING ####################
            r = trainset[setidx[j]]['r'] 
            nframes =  r.shape[0]
            beg, end = 0, nframes
            dataset_input={
                'r':   th.tensor(r[beg:end,None], dtype=th.float32, device=dev),
                'v':   th.tensor(trainset[setidx[j]]['v'][beg:end,None], dtype=th.float32, device=dev),
                'a':   th.tensor(trainset[setidx[j]]['a'][beg:end,None], dtype=th.float32, device=dev),
                'e':   th.tensor(trainset[setidx[j]]['e'][beg:end,None], dtype=th.float32, device=dev),
            }
            dataset = get_multiv_data( dataset_input, dt, step=lmem)
            #################### FORCE ######################
            potential_force, damp_force = model_ext( dataset['r'],  dataset['v'], dataset['e'] )
            a_pred =  potential_force + damp_force
            a_ref = dataset['a']
            #################### NOISE ######################
            noise = a_ref - a_pred  ## (nframe, height)
            noise_pred, noise_ref, noise_sigma = get_noise( model_noise, noise.detach()  )
            white_noise = noise_ref - noise_pred
            #################### LOSS ######################
            loss_reg = (noise**2).mean()   ## Loss for free energy
            loss_white = th.log(noise_sigma**2).mean()  ## loss for GAR model
            loss_white += (((noise_ref - noise_pred)/noise_sigma)**2).mean()
            loss += loss_white / batchsize  
            loss += loss_reg / batchsize 
            #################### Correlation ######################
            ## <V(0.5+t),V(0)> for self consistency relation
            v_retard = dataset['plain_v'].cpu().numpy().flatten()
            v_instant = (dataset['plain_v']+0.5*dataset['a']*dt).cpu().numpy().flatten()
            _u = (dataset['a'] - potential_force.detach()).cpu().numpy().flatten()
            _n = noise.detach().cpu().numpy().flatten()
            _corr_vvplus = [  (v_instant[:-1]*v_retard[1:]).mean()] + [ (v_instant[:-1-iT]     * v_retard[1+iT:]).mean()  for iT in range(1, lmem)    ]            
            _corr_uv = [  (_u*v_instant).mean()  ]  +      [ (v_instant[:-iT]     * _u[iT:]).mean()  for iT in range(1,lmem+1)  ]
            _corr_nn = [  (_n*_n).mean()]    + [ (_n[:-iT] * _n[iT:]).mean()  for iT in range(1, lmem+1)]
            corr_vvplus += np.array(_corr_vvplus) / batchsize
            corr_uv += np.array(_corr_uv) / batchsize
            corr_nn += np.array(_corr_nn) / batchsize
 
        ###########################    Optimize   ###########################
        if i%10==0:
            ag_size = model_noise.in_channels
            mat_cvv = np.zeros((lmem, lmem))
            mat_crr = np.zeros((ag_size, ag_size))
            for ii in range(lmem):
                for jj in range(ii+1):
                    mat_cvv[ii,jj] = corr_vvplus[ ii-jj ]
            for ii in range(ag_size):
                for jj in range(ag_size):
                    mat_crr[ii,jj] = corr_nn[np.abs(jj-ii)]            
            #####
            kernel_retard, res, rank, singular =  lstsq( mat_cvv, corr_uv[1:], rcond=0.0001 )
            kernel_retard_torch = th.tensor(kernel_retard, dtype=th.float32,device=dev)
            ag_coef, res, rank, singular =  lstsq( mat_crr, corr_nn[1:ag_size+1], rcond=0.00001 )
            ag_coef_torch = th.tensor(ag_coef, dtype=th.float32,device=dev)
            if i==0:
                model_ext.kernel =  kernel_retard_torch
                model_noise.linear +=  ag_coef_torch 
            else:
                model_ext.kernel +=  0.01* (kernel_retard_torch - model_ext.kernel)
                cut_region = int(model_ext.kernel.shape[0] /  10)
                model_ext.kernel -= model_ext.kernel[-1:].mean()
                model_noise.linear +=  0.01 * (ag_coef_torch - model_noise.linear)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if i % save_freq == 0 and i>1:
            th.save(model_ext.state_dict(), os.path.join(model_folder,'model_ext.{}.ckpt'.format(i)))
            th.save(model_noise.state_dict(), os.path.join(model_folder,'model_noise.{}.ckpt'.format(i)))