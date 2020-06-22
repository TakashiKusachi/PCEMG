#!/bin/bash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

class Transpose(nn.Module):
    def __init__(self,dim1,dim2):
        super(Transpose,self).__init__()
        self.dims = (dim1,dim2)
    
    def forward(self,*inputs):
        length = len(inputs)
        if length == 1:
            return inputs[0].transpose(*self.dims)
        else:
            ret = (one.transpose(*self.dims) for one in inputs)
            return ret

class GRUOutput(nn.Module):
    def __init__(self,bidirectional=False):
        super(GRUOutput,self).__init__()
        self.bidirectional = bidirectional

    def forward(self,input):
        h,_ = input
        if self.bidirectional:
            h = h[:,-1,:]+h[:,0,:]
        else:
            h = h[:,-1,:]
        return h
        

class conv_set(nn.Module):
    def __init__(self, \
            in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zero', \
            use_batchnorm=True,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, \
            ):
        super(conv_set,self).__init__()
        module_list = nn.ModuleList()

        module_list.append(
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode)
        )
        if use_batchnorm:
            module_list.append(
                nn.BatchNorm1d(num_features=out_channels,eps=eps,momentum=momentum,affine=affine,track_running_stats=track_running_stats)
            )
        module_list.append(
            nn.LeakyReLU()
        )

        self.convSequential = nn.Sequential(*module_list)
        #self.convSequential = nn.Sequential(\
        #    nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode), \
            #nn.BatchNorm1d(num_features=out_channels,eps=eps,momentum=momentum,affine=affine,track_running_stats=track_running_stats),
        #    nn.LeakyReLU()
        #    )
    
    def forward(self,input):
        return self.convSequential(input)

class Sampling(nn.Module):

    def __init__(self,input_size,output_size):
        super(Sampling,self).__init__()
        self.var = nn.Linear(input_size,output_size)
        self.mean = nn.Linear(input_size,output_size)

    def forward(self,h,sample_rate=1.0):
        batch_size = h.size(0)
        z_mean = self.mean(h)
        z_log_var = -torch.abs(self.var(h))
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon*sample_rate
        return z_vecs,kl_loss

class ms_peak_encoder_cnn(nn.Module):

    def __init__(self,max_mpz=1000,embedding_size=10,\
                 hidden_size=100,num_rnn_layers=2,bidirectional=False,\
                 output_size=56,dropout_rate=0.5,
                 varbose=True,
                 num_layer=1,*args,**kwargs):
        
        super(ms_peak_encoder_cnn, self).__init__()
        self._print = self.one_print if varbose else self.dumy
        self.embedding_size = embedding_size
        self.max_mpz = max_mpz
        self.bidirectional=bidirectional
        self.dropout_rate=dropout_rate
        
        self.embedding = nn.Embedding(max_mpz,embedding_size)
        
        last_size = self.embedding_size+1

        module_list = nn.ModuleList()

        for n_lay in range(num_layer):
            assert f'kernel{n_lay+1}_width' in kwargs,""
            assert f'conv{n_lay+1}_channel' in kwargs,""

            kerneln_width = kwargs[f'kernel{n_lay+1}_width']
            convn_channel = kwargs[f'conv{n_lay+1}_channel']
            layn_pad = int(((kerneln_width-1)/2))

            module_list.append(conv_set(last_size,convn_channel,kerneln_width,stride=1,padding=layn_pad,use_batchnorm=True))
            module_list.append(conv_set(convn_channel,convn_channel,kerneln_width,stride=1,padding=layn_pad,use_batchnorm=True))
            module_list.append(conv_set(convn_channel,convn_channel,kerneln_width,stride=1,padding=layn_pad,use_batchnorm=True))
            last_size = convn_channel
        module_list.append(Transpose(1,2))
        module_list.append(nn.Dropout(self.dropout_rate))
        module_list.append(nn.GRU(input_size=last_size, hidden_size=hidden_size, batch_first=True, num_layers=num_rnn_layers, bidirectional=bidirectional))
        module_list.append(GRUOutput(self.bidirectional))
        module_list.append(nn.Dropout(self.dropout_rate))
        self.convSequential = nn.Sequential(*module_list)

        if self.bidirectional:
            self.sampling = Sampling(hidden_size*2,output_size)
        else:
            self.sampling = Sampling(hidden_size,output_size)
        self.output = nn.Linear(output_size,output_size)
        
    def forward(self,x,y,sample=False,training=True,sample_rate=1):
        batch_size = x.size()[0]
        number_peak = x.size()[1]
        x = x.long()
        y = y.float()

        self.inp_ = self.embedding(x) # inp.size = (batch_size,number_peak,embedding_size)
        self.inp_.requires_grad_(True)
        inp = self.inp_
        
        self._print(inp.shape)
        y = y.view(batch_size,number_peak,1) # y.size = (batch_size,number_peak,1)
        self._print(y.shape)
        inp = torch.cat((inp,y),2) # inp.size = (batch_size,number_peak,embedding_size+1)
        self._print(inp.shape)
        inp = inp.transpose(1,2)
        
        h = self.convSequential(inp)
        self._print(h.shape)
        
        if sample:
            self._print = self.dumy
            return self.sampling(h,sample_rate=sample_rate)
        else:
            self._print=self.dumy
            return self.sampling(h,sample_rate=0.0)
        
    def one_print(self,*args,**kargs):
        print(args)
    def dumy(self,*args,**kargs):
        return 0
        
    def rsample(self, z_vecs, W_mean, W_var,sample_rate=1):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon*sample_rate
        return z_vecs, kl_loss
    
    def params_norm(self,only_linear=True,order=2):

        if only_linear:
            target_module = [module for key,module in self.convSequential._modules.items() if 'linear' in key]
            
            target_module.append(self.sampling)

        else:
            target_module = [self]

        s_norm = torch.tensor(0.0).cuda()
        for module in target_module:
            for param in module.parameters():
                s_norm += param.norm(order)

        #norm = torch.sum([param.norm(order) for param in target_params])
        return s_norm
    
    @staticmethod
    def config_dict(config=None):
        if config is None:
            config = {'max_mpz':1000,'embedding_size':10,'conv1_channel':64,'kernel1_width':5,'hidden_size':200,'num_rnn_layers':2,'bidirectional':False,'output_size':56,'dropout_rate':0.2}
        else:
            config ={key:conv(config[key]) for key,conv in zip(
                ['max_mpz','embedding_size','conv1_channel','kernel1_width','hidden_size','num_rnn_layers','bidirectional','output_size','dropout_rate'],
                [int,int,int,int,int,int,lambda x:x == 'True',int,float])
                
        }
        return config

    
if __name__=="__main__":
    print("What?")