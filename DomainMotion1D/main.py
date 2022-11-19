import os
from utility import *
from model import *
from train import train_eq

### system parameters
temp = 300
md_dt=0.002
stride = 5  # if you change this, change run.py -> noise
smooth = 40
le_dt = md_dt * stride ## the time step of the data
relax_t = 5 ## the relaxation time from the constrained structure
throw = int(relax_t / md_dt)
len_ag = 40 
lmem = 200
batchsize = 4
E = 2
model_folder = 'E{}g{}s{}ag{}lmem{}'.format(E, stride, smooth, len_ag, lmem )

##################  LOADING ##################
elist = np.load('data/e_final.npy')
dataset = []
with open('./data/cv_final.npy', 'rb') as f:
    for idx, efield in enumerate(elist):
        dataset.append(np.load(f,allow_pickle=True))
nset  = len(dataset)
print('# Traj =', nset)
trainset = []

for efield, data in zip(elist, dataset):
    if np.abs(efield - E) < 0.01:
        r,v,a = read_out_smooth(data, stride=stride, le_dt=le_dt, smooth = smooth, throw=throw)  # (1,steps  )
        r, v, a = r[0], v[0], a[0]
        trainset.append(
            {'r':r, 'v':v, 'a':a, 'e': np.zeros_like(r)+efield,}
            )
print('# Trainset =', len(trainset))

trainset_paras = {
    'beta' : 1/(kb*temp),
    'temp': temp,
    'stride': stride,
    'smooth':smooth,
    'dt': le_dt,
    'throw': throw,
    'lmem':lmem,
    'batchsize':batchsize,
}
##################  TRAINING ##################
dev = 'cuda' if th.cuda.is_available() else 'cpu'
th.set_default_tensor_type(th.FloatTensor)
model_ext = force_model( out_channels=lmem ).to(dev)
model_noise = GAR_model( len_ag).to(dev)
if os.path.exists(model_folder) is False:
    os.mkdir(model_folder)
train_eq(  model_ext, model_noise, trainset, trainset_paras,  model_folder=model_folder)
 