import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class MolDiscriminator(nn.Module):
    def __init__(self,latent_size,num_lay=1,dropout=0.5,*args,**kwargs):
        super(MolDiscriminator,self).__init__()
        self.latent_size = latent_size

        module_list = nn.ModuleList()
        last_size = latent_size

        for n in range(num_lay):
            assert f'lay{n+1}_size' in kwargs,""
            layn_size = kwargs[f'lay{n+1}_size']
            module_list.append(nn.Linear(last_size,layn_size))
            module_list.append(nn.Dropout(dropout))
            module_list.append(nn.LeakyReLU())
            last_size = layn_size
        module_list.append(nn.Linear(last_size,1))

        self.seq = nn.Sequential(*module_list)

    def forward(self,x):
        return self.seq(x)