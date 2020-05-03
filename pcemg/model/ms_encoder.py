#!/bin/bash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
            eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, \
            ):
        super(conv_set,self).__init__()
        self.convSequential = nn.Sequential(\
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode), \
            nn.BatchNorm1d(num_features=out_channels,eps=eps,momentum=momentum,affine=affine,track_running_stats=track_running_stats),
            nn.LeakyReLU()
            )
    
    def forward(self,input):
        return self.convSequential(input)

class ms_peak_encoder_cnn(nn.Module):

    def __init__(self,input_size,max_mpz=1000,embedding_size=10,\
                 conv1_channel=64,kernel1_width=5,\
                 conv2_channel=128,kernel2_width=5,\
                 conv_output_channel=256,conv_output_width=5,
                 hidden_size=100,num_rnn_layers=2,bidirectional=False,\
                 output_size=56,dropout_rate=0.5,
                 varbose=True):
        
        super(ms_peak_encoder_cnn, self).__init__()
        self._print = self.one_print if varbose else self.dumy
        self.embedding_size = embedding_size
        self.max_mpz = max_mpz
        self.bidirectional=bidirectional
        self.dropout_rate=dropout_rate
        
        self.embedding = nn.Embedding(max_mpz,embedding_size)
        
        lay0_pad = int(((kernel1_width-1)/2))
        lay1_pad = int(((kernel2_width-1)/2))
        lay2_pad = int(((conv_output_width-1)/2))

        self.convSequential=nn.Sequential(\
            conv_set(self.embedding_size+1, conv1_channel, kernel1_width, stride=1, padding=lay0_pad), \
            conv_set(conv1_channel,         conv1_channel, kernel1_width, stride=1, padding=lay0_pad), \
            conv_set(conv1_channel,         conv1_channel, kernel1_width, stride=1, padding=lay0_pad), \
            #nn.MaxPool1d(2),\
            #conv_set(conv1_channel,         conv2_channel, kernel2_width, stride=1, padding=lay1_pad), \
            #conv_set(conv2_channel,         conv2_channel, kernel2_width, stride=1, padding=lay1_pad), \
            #conv_set(conv2_channel,         conv2_channel, kernel2_width, stride=1, padding=lay1_pad), \
            #nn.MaxPool1d(2),\
            #conv_set(conv2_channel,         conv_output_channel, conv_output_width, stride=1, padding=lay2_pad), \
            #conv_set(conv_output_channel,   conv_output_channel, conv_output_width, stride=1, padding=lay2_pad), \
            #conv_set(conv_output_channel,   conv_output_channel, conv_output_width, stride=1, padding=lay2_pad), \
            #nn.MaxPool1d(2),\
            Transpose(1,2), \
            nn.GRU(input_size=conv1_channel, hidden_size=hidden_size, batch_first=True, num_layers=num_rnn_layers, bidirectional=bidirectional), \
            GRUOutput(self.bidirectional), \
            nn.Dropout(self.dropout_rate), \
            nn.Linear(hidden_size,hidden_size), nn.LeakyReLU(), \
            )
        
        
        #self.rnn = nn.GRU(input_size=conv_output_channel,hidden_size=hidden_size,batch_first=True,num_layers=num_rnn_layers,bidirectional=bidirectional)
        #self.hidden_linear = nn.Linear(conv_output_channel*int(input_size/2/2/2),hidden_size)
        if self.bidirectional:
            self.T_mean = nn.Linear(hidden_size*2, int(output_size/2))
            self.T_var = nn.Linear(hidden_size*2, int(output_size/2))
            self.G_mean = nn.Linear(hidden_size*2, int(output_size/2))
            self.G_var = nn.Linear(hidden_size*2, int(output_size/2))
        else:
            self.T_mean = nn.Linear(hidden_size, int(output_size/2))
            self.T_var = nn.Linear(hidden_size,int(output_size/2))
            self.G_mean = nn.Linear(hidden_size, int(output_size/2))
            self.G_var = nn.Linear(hidden_size, int(output_size/2))
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
        #inp = y*inp
        self._print(inp.shape)
        inp = inp.transpose(1,2)
        
        #h = self.convSequential(inp).transpose(1,2)
        h = self.convSequential(inp)
        #h = h.view((batch_size,-1))
        self._print(h.shape)
        
        
        #h = F.dropout(h,p=self.dropout_rate,training=training)
        #self._print(h.shape)
        #h = self.hidden_linear(h)
        #h = F.leaky_relu(h)
        #h = torch.flip(h,(1,))
        #h,_ = self.rnn(h)
        #self._print(h.shape)
        
        #if self.bidirectional:
        #    h = h[:,-1,:]+h[:,0,:]
        #else:
        #    h = h[:,-1,:]

        #self._print(h.shape)
        #h = F.dropout(h,p=self.dropout_rate,training=training)
        
        if sample:
            t_vecs,t_kl_loss = self.rsample(h,self.T_mean,self.T_var,sample_rate=sample_rate)
            g_vecs,g_kl_loss = self.rsample(h,self.G_mean,self.G_var,sample_rate=sample_rate)
            h = torch.cat((t_vecs,g_vecs),1)
            #h = F.leaky_relu(h)
            #h = self.output(h)
            kl_loss = t_kl_loss + g_kl_loss
            self._print = self.dumy
            return h,kl_loss
        else:
            t_vecs = self.T_mean(h)
            g_vecs = self.G_mean(h)
            h = torch.cat((t_vecs,g_vecs),1)
            #h = F.leaky_relu(h)
            #h = self.output(h)
            self._print=self.dumy
            return h,torch.zeros((1)).cuda(h.device)
        
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