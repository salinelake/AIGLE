import os
import sys 
import numpy as np
import torch as th
from model import *
from aigle import aigle0D
from utility import read_out_smooth
import logging 

################## system parameters  ##################
temp = 300
md_dt=0.002
stride = 5  # if you change this, change run.py -> noise
smooth = 40
le_dt = md_dt * stride ## the time step of the data
################## dataset parameters ##################
relax_t = 5 ## the equlibration part to be removed from data
throw = int(relax_t / md_dt)
E = 2
################## network parameters ##################
dev = 'cuda' if th.cuda.is_available() else 'cpu'
th.set_default_tensor_type(th.FloatTensor)
len_ag = 40 
lmem = 200
################## Training parameters ##################
model_folder = 'E{}g{}s{}ag{}lmem{}'.format(E, stride, smooth, len_ag, lmem )
out_folder = model_folder + '_noneq'
train_stage = 1
batchsize = 4
nepoch = 10000
##################  LOADING ##################
elist = np.load('data/e_final.npy')
dataset = []
with open('./data/cv_final.npy', 'rb') as f:
    for idx, efield in enumerate(elist):
        dataset.append(np.load(f,allow_pickle=True))
nset  = len(dataset)

##################  TRAINING ##################

if train_stage == 0:
    ## extract trainset from dataset
    trainset = []
    for efield, data in zip(elist, dataset):
        if np.abs(efield - E) < 0.01:
            r,v,a = read_out_smooth(data, stride=stride, le_dt=le_dt, smooth = smooth, throw=throw)  # (1,steps  )
            r, v, a = r[0], v[0], a[0]   # 1D vector
            trainset.append(
                {'r':r, 'v':v, 'a':a, 'e': np.zeros_like(r)+efield,}
                )
    print( ' # MD trajectories= {:d} ; # Trainset = {:d}'.format(nset, len(trainset)))
    ## train AIGLE
    model_ext = force_model( out_channels=lmem ).to(dev)
    model_noise = GAR_model( len_ag).to(dev)
    if os.path.exists(model_folder) is False:
        os.mkdir(model_folder)
    aigle_0d = aigle0D( temp, dt=le_dt, len_ag=len_ag, lmem=lmem, model_ext = model_ext, model_noise = model_noise  )
    print('Training univariant AIGLE, output folder:{}'.format(model_folder))
    aigle_0d.train_init(  trainset, batchsize=batchsize, epoch=nepoch , model_folder=model_folder )
else:
    ## extract trainset from dataset
    trainset = []
    for efield, data in zip(elist, dataset):
        if  (efield <= 2.4 and efield >=2.0):
            r,v,a = read_out_smooth(data, stride=stride, le_dt=le_dt, smooth = smooth, throw=throw)  # (1,steps  )
            r, v, a = r[0], v[0], a[0]
            trainset.append(
                {'r':r, 'v':v, 'a':a, 'e': np.zeros_like(r)+efield,}
                )
    ## train AIGLE
    model_ext = force_model( out_channels=lmem ).to(dev)
    model_noise = GAR_model( len_ag).to(dev)
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    aigle_0d = aigle0D( temp, dt=le_dt, len_ag=len_ag, lmem=lmem, model_ext = model_ext, model_noise = model_noise  )
    aigle_0d.load_model(model_folder, label='5000')
    ## fix barrier height to metadynamics result
    aigle_0d.model_ext.well_coef.data = aigle_0d.model_ext.well_coef.data*0 + 0.55
    aigle_0d.model_ext.well_coef.requires_grad = False
    aigle_0d.train_force( trainset, batchsize=batchsize,  epoch=nepoch , model_folder=out_folder )
