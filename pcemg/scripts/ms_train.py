
import numpy as np
import rdkit

import sys
import tempfile
from argparse import ArgumentParser
import shutil
from pathlib import Path
import pickle
from tarfile import TarFile
import time,datetime
from configparser import ConfigParser

from pcemg.scripts.utils import is_env_notebook
if is_env_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from torch_jtnn import *
from pcemg.datautil import dataset_load
from pcemg.model.ms_encoder import ms_peak_encoder_cnn

class MS_Train():
    """
    Training of peak_encoder and JTVAE-decoder.


    """
    def __init__(self,vocab_path,dataset_path,load_model,model_config,train_vali_rate = 0.9):
        assert Path(vocab_path).is_file(),""
        assert Path(dataset_path).is_file(),""
        assert Path(load_model).is_file,""

        if not Path(model_config).exists():
            chart_config()
            raise RuntimeError("chart config ")

        self.__vocab_path = vocab_path
        self.__dataset_path = dataset_path
        self.__load_model = load_model
        self.__model_config = model_config
        self.train_vali_rate = train_vali_rate

        self.starttime = datetime.datetime.fromtimestamp(time.time())

    def __call__(self):
        tempdir = tempfile.TemporaryDirectory(prefix='ms-train')
        temp_path = Path(tempdir.name)
        print("\n")
        try:
            self.copy_require_file(tempdir)
            print(self.vocab_path)
            print(self.dataset_path)

            with open(self.vocab_path,'r') as f:
                vocab,_ = zip(*[(x.split(',')[0].strip('\n\r'),int(x.split(',')[1].strip('\n\r'))) for x in f ])

            vocab = Vocab(vocab,_)

            train_dataset,vali_dataset = dataset_load(self.dataset_path,vocab,20,self.train_vali_rate,save=(temp_path/'MS_Dataset.pkl').name)

            with open(temp_path/'vali_data.pkl','wb') as f:
                pickle.dump(vali_dataset,f)
            with open(temp_path/'train_data.pkl','wb') as f:
                pickle.dump(train_dataset,f)
                
            print("number of train dataset: ",len(train_dataset)*train_dataset.batch_size)
            print("number of validation dataset: ",len(vali_dataset)*train_dataset.batch_size)
            
            config = self.load_config()

            dec_model = JTNNVAE(vocab=vocab,**config['JTVAE']).to('cuda')
            
            enc_model = ms_peak_encoder_cnn(train_dataset.max_spectrum_size,**config['PEAK_ENCODER'],varbose=False).to('cuda')

            print(dec_model)
            print(enc_model)
            dec_model.load_state_dict(torch.load(self.load_model,map_location='cuda'))

            enc_optimizer = optim.Adam(enc_model.parameters(),lr=1e-03)
            dec_optimizer = optim.Adam(dec_model.parameters(),lr=1e-03)

            self.training(enc_model,dec_model,enc_optimizer,dec_optimizer,train_dataset,vali_dataset,temp_path,
                **config['TRAINING'])

        finally:
            str_time = self.starttime.strftime('%Y%m%d-%H%M')
            with TarFile.open('result-{}.tar.gz'.format(str_time),'w:gz') as tarfile:
                tarfile.add(tempdir.name,arcname='./{}'.format(str_time))
            tempdir.cleanup()

    def training(self, \
        enc_model,dec_model,enc_optimizer,dec_optimizer, \
        train_dataset, vali_dataset, \
        temp_path, \
        max_epoch = 300, word_rate=1,topo_rate=1,assm_rate=1,reg_rate=1, \
        fine_tunning_warmup = 100, warmup= 200, init_beta= 0, step_beta=0.002, max_beta= 1, kl_anneal_iter= 10,\
        anneal_rate=0.8, anneal_iter=1000, \
        valid_interval=200, save_interval=200,\
        ):

        log_path = temp_path/'log.csv'
        temp_path.joinpath('enc_model').mkdir()
        temp_path.joinpath('dec_model').mkdir()

        beta = init_beta
        meters = np.zeros(7)

        enc_scheduler = lr_scheduler.ExponentialLR(enc_optimizer, anneal_rate)
        dec_scheduler = lr_scheduler.ExponentialLR(dec_optimizer, anneal_rate)

        with open(log_path,'w') as f:
            f.write("epoch,iter.,kl_loss,word,topo,assm,wors_loss,topo_loss,assm_loss,vali word,vali topo,vali assm,vali_word_loss,vali_topo_loss,vali_assm_loss,l2_reg\n")

        for epoch,iteration,batch in self.get_batch(max_epoch,train_dataset,progress=True):
            x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder,x,y = batch
            
            x = x.to('cuda')
            y = y.to('cuda')

            enc_model.zero_grad()
            dec_model.zero_grad()
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            enc_model.train()
            dec_model.train()

            h,kl_loss = enc_model(x,y,training=True,sample=True)
            tree_vec = h[:,:int(h.shape[1]/2)]
            mol_vec  = h[:,int(h.shape[1]/2):]
            _, x_tree_mess = dec_model.jtnn(*x_jtenc_holder)
            word_loss, topo_loss, word_acc, topo_acc = dec_model.decoder(x_batch,tree_vec)
            assm_loss, assm_acc = dec_model.assm(x_batch, x_jtmpn_holder, mol_vec , x_tree_mess)

            l2_reg = 0
            for W in enc_model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
            for W in dec_model.parameters():
                l2_reg = l2_reg + W.norm(2)

            total_loss = word_loss*word_rate+\
                topo_loss*topo_rate+\
                assm_loss*assm_rate+\
                kl_loss*beta +\
                reg_rate*l2_reg

            total_loss.backward()
            enc_optimizer.step()
            if epoch >= fine_tunning_warmup:
                dec_optimizer.step()
                if iteration % anneal_iter == 0 :
                    dec_scheduler.step()
            
            meters = meters + np.array([kl_loss.item(),word_acc * 100, topo_acc * 100, assm_acc * 100,word_loss.item(),topo_loss.item(),assm_loss.item()])
            del x,y,h
            if iteration % valid_interval == 0:

                meters /= valid_interval
                vali_meters =self.vali_forward(enc_model,dec_model,vali_dataset)
                
                #sys.stdout.write("epoch: %04d, iteration: %08d\n" % (epoch,iteration))
                sys.stdout.write("%d epoch[%d] kl_loss %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f vali_Word: %.2f, vali_Topo: %.2f, vali_assm: %.2f, l2_reg: %.2f \n" % \
                    (epoch,iteration,meters[0], meters[1], meters[2],meters[3], vali_meters[0],vali_meters[1],vali_meters[2],l2_reg   ))             
                with open(log_path,"a") as f:
                    f.write("%d,%d," % (epoch,iteration))
                    f.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f," % (meters[0], meters[1], meters[2],meters[3],meters[4],meters[5],meters[6]))
                    f.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (vali_meters[0],vali_meters[1],vali_meters[2],vali_meters[3],vali_meters[4],vali_meters[5],l2_reg))
                    f.write("\n")
                sys.stdout.flush()
                meters *= 0
                vali_meters *= 0

            if iteration % save_interval == 0:
                torch.save(enc_model.state_dict(), temp_path/"enc_model"/"model.iter-{}".format(str(iteration)))
                torch.save(dec_model.state_dict(), temp_path/"dec_model"/"model.iter-{}".format(str(iteration)))

            if epoch % kl_anneal_iter == 0 and epoch >= warmup:
                beta = min(max_beta, beta + step_beta)

            if iteration % anneal_iter == 0 :
                enc_scheduler.step()

    def copy_require_file(self,tempdir):
        temp_path = Path(tempdir.name)
        self.vocab_path = shutil.copy(self.__vocab_path,temp_path)
        self.dataset_path = shutil.copy(self.__dataset_path,temp_path)
        self.load_model = shutil.copy(self.__load_model,temp_path)
        self.model_config = shutil.copy(self.__model_config,temp_path)

    def get_batch(self,max_epoch,dataset,progress=False):
        iteration = 0
        total_iter = len(dataset)*max_epoch

        progressbar = tqdm if progress else lambda x,*args,**kwargs:x

        with progressbar(total=total_iter) as pbar:
            for epoch in range(max_epoch):
                for batch in dataset:
                    pbar.set_description("epoch: {e}, iter: {i}".format(e=epoch,i=iteration))
                    yield epoch,iteration,batch
                    iteration += 1
                    pbar.update()

        #for epoch in progressbar(range(max_epoch),desc="epoch loop"):
        #    for batch in progressbar(dataset,desc='iteration loop',total=len(dataset),leave=False):
        #        iteration += 1
        #        yield epoch,iteration,batch
    
    def vali_forward(self,enc_model,dec_model,vali_dataset):
        vali_total = 0
        vali_meters = np.zeros(6)
        for batch in vali_dataset:
            x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder,x,y = batch
            x = x.to('cuda')
            y = y.to('cuda')
            with torch.no_grad():
                enc_model.eval()
                dec_model.eval()
                h,_ = enc_model(x,y,training=False,sample=False)
                tree_vec = h[:,:int(h.shape[1]/2)]
                mol_vec  = h[:,int(h.shape[1]/2):]
                _, x_tree_mess = dec_model.jtnn(*x_jtenc_holder)
                word_loss, topo_loss, word_acc, topo_acc = dec_model.decoder(x_batch,tree_vec)
                assm_loss, assm_acc = dec_model.assm(x_batch, x_jtmpn_holder, mol_vec , x_tree_mess)
                vali_meters = vali_meters + np.array([word_acc * 100, topo_acc * 100, assm_acc * 100,word_loss.item(),topo_loss.item(),assm_loss.item()])
                vali_total += 1    
            del x,y,h
        vali_meters /= vali_total

        return vali_meters

    def load_config(self):
        config = ConfigParser()
        config.optionxform=str
        config.read(self.model_config)
        ret = dict()
        ret['JTVAE'] = JTNNVAE.config_dict(config['JTVAE'])
        ret['PEAK_ENCODER'] = ms_peak_encoder_cnn.config_dict(config['PEAK_ENCODER'])
        ret['TRAINING'] = self.train_config(config['TRAINING'])

        return ret
    
    @staticmethod
    def train_config(config=None):
        if config is None:
            config = {'max_epoch':300, 'word_rate':1, 'topo_rate':1, 'assm_rate':1, 'reg_rate':1, \
                'fine_tunning_warmup':100, 'warmup':200, 'init_beta':0, 'step_beta':0.002, 'max_beta':1, 'kl_anneal_iter':10,\
                'anneal_rate':0.8,'anneal_iter':1000, \
                'valid_interval':200, 'save_interval':200,}
        else:
            config ={key:conv(config[key]) for key,conv in zip(
                ['max_epoch', 'word_rate', 'topo_rate', 'assm_rate', 'reg_rate', \
                'fine_tunning_warmup', 'warmup', 'init_beta', 'step_beta', 'max_beta', 'kl_anneal_iter',\
                'anneal_rate','anneal_iter', \
                'valid_interval', 'save_interval',],
                [int,float,float,float,float,int,int,float,float,float,int,float,int,int,int,])
                
        }
        return config

def chart_config():
    from configparser import ConfigParser
    config = ConfigParser()
    config.optionxform = str
    config['JTVAE'] = JTNNVAE.config_dict()
    config['PEAK_ENCODER']=ms_peak_encoder_cnn.config_dict()
    config['TRAINING'] = MS_Train.train_config()

    with open("model_config.ini",'w') as f:
        config.write(f)

def ms_train(vocab_path,dataset_path,load_model,model_config):
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    MS_Train(vocab_path,dataset_path,load_model,model_config)()

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = ArgumentParser()
    parser.add_argument("vocab_path")
    parser.add_argument("dataset_path")
    parser.add_argument('load_model')
    parser.add_argument('model_config')

    args = parser.parse_args()

    proc = MS_Train(**args.__dict__)
    proc()


if __name__=="__main__":
    main()
