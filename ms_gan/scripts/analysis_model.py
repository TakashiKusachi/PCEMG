
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from tarfile import TarFile
import time,datetime
import shutil
from configparser import ConfigParser
import pickle
from glob import glob

from ms_gan.scripts.utils import is_env_notebook
if is_env_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import re
import csv
import numpy as np

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import torch

from multiprocessing import Pool

from torch_jtnn import *
from ms_gan.datautil import dataset_load
from ms_gan.model.ms_encoder import ms_peak_encoder_cnn as ms_peak_encoder_cnn,raw_spectrum_encoder

class AnalysisModel():
    def __init__(self,trained_result_path,eval_mode):
        self.__trained_result_path = trained_result_path
        self.__eval_mode  = eval_mode
        self.starttime = datetime.datetime.fromtimestamp(time.time())
        
    def __call__(self):
        
        lg = rdkit.RDLogger.logger() 
        lg.setLevel(rdkit.RDLogger.CRITICAL)

        tempdir = None
        log = None

        print("\n")
        try:
            tempdir = tempfile.TemporaryDirectory(prefix='analysis_model-')
            temp_path = Path(tempdir.name)

            log = (temp_path/"log").open('w')

            self.copy_require_file(tempdir)
            config = self.load_config()

            with open(self.vali_file,'rb') as f:
                vali_dataset = pickle.load(f)

            with open(self.vocab_path,'r') as f:
                vocab,_ = zip(*[(x.split(',')[0].strip('\n\r'),int(x.split(',')[1].strip('\n\r'))) for x in f ])
                
            vocab = Vocab(vocab,_)

            dec_model = JTNNVAE(vocab=vocab,**config['JTVAE']).to('cuda')
            
            enc_model = ms_peak_encoder_cnn(vali_dataset.max_spectrum_size,**config['PEAK_ENCODER'],varbose=False).to('cuda')
            
            print(enc_model)
            print(dec_model)
            enc_model_path,dec_model_path = AnalysisModel.select_iter(self.model_list,self.iter_list)
            print(enc_model_path,dec_model_path)
            

            enc_model.load_state_dict(torch.load(enc_model_path,map_location='cuda'))
            dec_model.load_state_dict(torch.load(dec_model_path,map_location='cuda'))

            sample_rate_list = [
                [0.0,1],
                [1.0,5],
                [3.0,10]
            ]

            class Fetcher():
                def __init__(self,sample_rate,times,eval_mode):
                    self.vali_dataset = vali_dataset
                    self.enc_model = enc_model
                    self.dec_model = dec_model
                    self.__length = lambda: self.vali_dataset.batch_size * len(self.vali_dataset)
                    self.__eval_mode = eval_mode
                    self.sample_rate = sample_rate
                    self.times = times

                def __len__(self):
                    return self.__length()
                

                def eval_mode(self):
                    if self.__eval_mode is False:
                        return

                    self.enc_model.eval()
                    self.dec_model.eval()

                def __iter__(self):
                    vali_dataset =self.vali_dataset
                    enc_model = self.enc_model
                    dec_model = self.dec_model
                    sample_rate = self.sample_rate

                    for batch in vali_dataset:
                        x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder,x,y = batch
                        x = x.to('cuda')
                        y = y.to('cuda')

                        self.eval_mode()
                        h,_ = enc_model(x,y,training=False,sample=True,sample_rate=sample_rate)   
                        hlatent_size = int(h.shape[1]/2)  
                        tree_vec = h[:,:hlatent_size]
                        mol_vec  = h[:,hlatent_size:]
                        
                        for num in range(h.size()[0]):
                            a_tree_vec = tree_vec[num].view(1,hlatent_size)
                            a_mol_vec = mol_vec[num].view(1,hlatent_size)
                            yield a_tree_vec,a_mol_vec

            def evaluation(eval_mode):
                result = []
                true_smiles = []

                total_data = vali_dataset.batch_size * len(vali_dataset)
                total_iter = 0
                for _,times in sample_rate_list:
                    total_iter += times


                # 正解分子構造の取り出し
                for batch in vali_dataset:
                    x_batch = batch[0]
                    true_smiles.extend([[Chem.MolToSmiles(Chem.MolFromSmiles(x_data.smiles),True)] for x_data in x_batch])

                with tqdm(total=total_iter) as q:
                    # 検証のための構造予測のループ
                    for rate,times in sample_rate_list:
                        q.set_description(desc="sample rate: "+str(rate))
                        for t in range(times):
                            for num,(tree_vec,mol_vec) in tqdm(enumerate(Fetcher(rate,times,eval_mode)),leave=False, total=total_data):
                                predict_smiles = dec_model.decode(tree_vec,mol_vec,False)
                                predict_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(predict_smiles),True)
                                true_smiles[num].append(predict_smiles)
                            q.update()
                
                return true_smiles


            with torch.no_grad():

                result = evaluation(self.__eval_mode)

                print(len(result))
                print(len(result[0]))
            
            scores = AnalysisModel.calc_similarity(result)
            max_scores,hist,over_thre = self.statistics(scores)

            log_path = temp_path / "area_sample_result.csv"
            with log_path.open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['rate','number of times'])
                writer.writerows(sample_rate_list)
                writer.writerow([])
                writer.writerow(['over 0.85',str(over_thre)])
                writer.writerow([])
                writer.writerow(['true smiles',])
                writer.writerows(result)
                writer.writerow([])
                writer.writerows(scores)
                writer.writerow([])
                writer.writerow(['<'+str(p / 100) for p in range(5,105,5)])
                writer.writerow([str(one) for one in hist])
            shutil.copy(log_path,"./area_sample_result.csv")

        finally:
            str_time = self.starttime.strftime('%Y%m%d-%H%M')

            if tempdir:
                tempdir.cleanup()

            if log:
                log.close()

    def copy_require_file(self,tempdir):
        temp_path = Path(tempdir.name)
        trained_result_path = shutil.copy(self.__trained_result_path,temp_path)

        with TarFile.open(trained_result_path,"r:gz") as f:
            f.extractall(path=temp_path)

        self.model_list,self.iter_list = AnalysisModel.find_trained_model(temp_path)
        self.model_config = AnalysisModel.find_config_file(temp_path)
        self.vocab_path = AnalysisModel.find_vocab_file(temp_path)
        self.vali_file = AnalysisModel.find_validata(temp_path)

    @staticmethod
    def find_trained_model(temp_path):
        
        def extract_iter_num(mlist):
            for path in mlist:
                try:
                    yield int(path.name.split('-')[1])
                except ValueError:
                    pass
            #return int(path.name.split('-')[1])

        enc_model_list = list(temp_path.glob("./**/enc_model/model.iter-*"))
        dec_model_list = list(temp_path.glob("./**/dec_model/model.iter-*"))
        
        iter_list = sorted([n for n in extract_iter_num(enc_model_list)])

        model_list = {
            'encoder':enc_model_list,
            'decoder':dec_model_list,
        }
        return model_list,iter_list
    
    @staticmethod
    def find_config_file(temp_path):
        model_config = list(temp_path.glob('./**/model_config.ini'))[0]
        print(model_config)
        return model_config
    
    @staticmethod
    def find_vocab_file(temp_path):
        vocab = list(temp_path.glob('./**/MS_vocab.txt'))[0]
        print(vocab)
        return vocab
        
    @staticmethod
    def find_validata(temp_path):
        vali = list(temp_path.glob('./**/vali_data.pkl'))[0]
        print(vali)
        return vali


    @staticmethod
    def select_iter(model_list,iter_list):
        if len(iter_list) > 5:
            header = "please choice iterate [" + str(iter_list[0]) + " " + str(iter_list[1]) + " ... " + str(iter_list[-2]) + " " + str(iter_list[-1])+"] >> "
        else:
            header = "please choise iterate ["
            for i in iter_list:
                header += str(i) + " "
            header += "] >> "

        select_iter = input(header)
        #if not int(select_iter) in iter_list:
        #    raise IndexError()

        enc_model = None
        dec_model = None

        for model_path in model_list['encoder']:
             if "model.iter-"+select_iter in model_path.name:
                 enc_model = model_path

        for model_path in model_list['decoder']:
             if "model.iter-"+select_iter in model_path.name:
                 dec_model = model_path

        assert enc_model is not None and dec_model is not None

        return enc_model,dec_model

    @staticmethod
    def calc_similarity(result):

        class Fetcher():
            def __init__(self,result,fingerprint=AllChem.GetMACCSKeysFingerprint):
                self.result = result
                self.total_data = len(result) * (len(result[0]) - 1)
                self.fingerprint = fingerprint

            def __len__(self):
                return self.total_data

            def __iter__(self):
                for row in self.result:
                    true = Chem.MolFromSmiles(row[0])
                    true_finger = self.fingerprint(true)
                    for col in row[1:]:
                        args = {
                            'true_mol':true,
                            'true_finger':true_finger,
                            'pred_smiles':col
                        }
                        yield args

        with Pool() as p:
            scores = p.map(_proc,Fetcher(result))
        
        ret = []
        for row in range(len(result)):
            top = row*(len(result[0]) - 1)
            bottom = (row+1)*(len(result[0]) - 1)
            ret.append([one['maccs_score'] for one in scores[top:bottom]])
        
        return ret

    def statistics(self,scores):
        scores = np.asarray(scores)
        max_scores = scores.max(axis=1)
        print(max_scores)
        print(max_scores.shape)
        hist,scale = np.histogram(max_scores,bins=20,range=(0.0,1.0))
        over_threshold = np.average(max_scores >= 0.85)*100
        print(over_threshold)
        return max_scores,hist,over_threshold


    def load_config(self):
        config = ConfigParser()
        config.optionxform=str
        config.read(self.model_config)
        print(self.model_config.name)
        print(config.sections())
        ret = dict()
        ret['JTVAE'] = JTNNVAE.config_dict(config['JTVAE'])
        ret['PEAK_ENCODER'] = ms_peak_encoder_cnn.config_dict(config['PEAK_ENCODER'])

        return ret
    

def _proc(args):
    true_finger = args['true_finger']
    pred_smiles = args['pred_smiles']
    pred = Chem.MolFromSmiles(pred_smiles)
    pred_finger = AllChem.GetMACCSKeysFingerprint(pred)
    maccs_score =  DataStructs.TanimotoSimilarity(true_finger,pred_finger)
    ret = {
        'maccs_score':maccs_score
    }
    return ret

def analysisModel(trained_result_path,eval_mode=True):
    AnalysisModel(trained_result_path,eval_mode)()

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = ArgumentParser()

    parser.add_argument('trained_result_path',type=str)

    args = parser.parse_args()

    proc = AnalysisModel(**args.__dict__)
    proc()

if __name__=="__main__":
    main()