
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from tarfile import TarFile
import time,datetime
import shutil
from configparser import ConfigParser
import pickle
from glob import glob

from pcemg.scripts.utils import is_env_notebook
if is_env_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import re
import csv
import openpyxl
import numpy as np
import torch

import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import torch

from multiprocessing import Pool

from torch_jtnn import *
from pcemg.datautil import dataset_load
from pcemg.model.ms_encoder import ms_peak_encoder_cnn as ms_peak_encoder_cnn

import logging
from logging import getLogger

class AnalysisModel():
    def __init__(self,trained_result_path,eval_mode,logger=None):
        self.__trained_result_path = trained_result_path
        self.__eval_mode  = eval_mode
        self.__logger = logger
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

            self.__logger.info(f"Temp path:{temp_path}")
            log = (temp_path/"log").open('w')

            self.copy_require_file(tempdir)
            config = self.load_config()

            with open(self.vali_file,'rb') as f:
                vali_dataset = pickle.load(f)
            vali_dataset.shuffle = False

            with open(self.vocab_path,'r') as f:
                vocab,_ = zip(*[(x.split(',')[0].strip('\n\r'),int(x.split(',')[1].strip('\n\r'))) for x in f ])
                
            vocab = Vocab(vocab,_)

            dec_model = JTNNVAE(vocab=vocab,**config['JTVAE']).to('cuda')
            
            enc_model = ms_peak_encoder_cnn(vali_dataset.max_spectrum_size,**config['PEAK_ENCODER'],varbose=False).to('cuda')
            
            print(f"Structuer of Encoder\n {enc_model}")
            print(f"Structuer of Decoder\n {dec_model}")
            enc_model_path,dec_model_path = AnalysisModel.select_iter(self.model_list,self.iter_list)
            self.__logger.info(f"{enc_model_path},{dec_model_path}")
            

            enc_model.load_state_dict(torch.load(enc_model_path,map_location='cuda'))
            dec_model.load_state_dict(torch.load(dec_model_path,map_location='cuda'))

            sample_rate_list = [
                [0.0,1],
                [1.0,5],
                [3.0,10]
            ]

            def evaluation(eval_mode):
                """ Predict molecular structure from spectrum in validation datasets.
                
                Args:
                    eval_mode (bool):
                
                Returns:
                    List_SMILES (list of string): 
                    mean (ndarray): 
                    log_var (ndarray):

                """
                result = []
                true_smiles = []

                total_data = vali_dataset.batch_size * len(vali_dataset)

                rate_list = list()
                total_iter = 0
                for rate,times in sample_rate_list:
                    rate_list.extend([rate for _ in range(times)])
                total_iter = len(rate_list)
                self.__logger.info(f"rate_list :{rate_list}")

                mean = list()
                log_var = list()

                if eval_mode:
                    enc_model.eval()
                    dec_model.eval()

                # 正解分子構造の取り出し
                for batch in tqdm(vali_dataset,desc="Encoding"):
                    x_batch, _, _, _,x,y = batch
                    true_smiles.extend([[Chem.MolToSmiles(Chem.MolFromSmiles(x_data.smiles),True)] for x_data in x_batch])

                    x = x.to('cuda')
                    y = y.to('cuda')

                    b_mean,b_log_var = enc_model(x,y,sample=False)
                    mean.append(b_mean)
                    log_var.append(b_log_var)
                
                mean = torch.cat(mean,dim=0)
                log_var = torch.cat(log_var,dim=0)
                self.__logger.info(f"mean shape: {mean.shape}")

                with tqdm(total=total_iter) as q:
                    for rate in rate_list:
                        q.set_description(desc="sample rate: "+str(rate))
                        epsilon = torch.randn_like(mean)
                        z = mean + torch.exp(log_var/2)*epsilon*rate
                        length = z.size(0)
                        point_split = int(z.size(1) / 2)
                        tree_vec = z[:,:point_split]
                        mol_vec = z[:,point_split:]

                        for num,(tree,mol) in tqdm(enumerate([(tree_vec[b:b+1],mol_vec[b:b+1]) for b in range(length)]),leave=False,total=length):
                            predict_smiles = dec_model.decode(tree,mol,False)
                            predict_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(predict_smiles),True)
                            true_smiles[num].append(predict_smiles)
                            
                        q.update()


                """
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
                """
                return true_smiles,mean.to('cpu').detach().numpy(),log_var.to('cpu').detach().numpy()


            with torch.no_grad():

                result,mean,log_var = evaluation(self.__eval_mode)

                print(len(result))
                print(len(result[0]))
            
            log_path = temp_path / "mean_table.csv"
            np.savetxt(str(log_path),mean,delimiter=',',fmt='%.5e')
            shutil.copy(log_path,"./mean_table.csv")

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
            
            log_path = temp_path / "area_sample_result.xlsx"
            #self.save_result_to_xlsx(sample_rate_list,result,log_path)

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

        self.model_list,self.iter_list = self.find_trained_model(temp_path)
        self.model_config = self.find_config_file(temp_path)
        self.vocab_path = self.find_vocab_file(temp_path)
        self.vali_file = self.find_validata(temp_path)

    def find_trained_model(self,temp_path):
        
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
    
    def find_config_file(self,temp_path):
        model_config = list(temp_path.glob('./**/model_config.ini'))[0]
        self.__logger.info(f"Model Configuration file path:{model_config}")
        return model_config
    
    def find_vocab_file(self,temp_path):
        vocab = list(temp_path.glob('./**/MS_vocab.txt'))[0]
        self.__logger.info(vocab)
        return vocab
        
    def find_validata(self,temp_path):
        vali = list(temp_path.glob('./**/vali_data.pkl'))[0]
        self.__logger.info(vali)
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
        with self.model_config.open('r') as f:
            print(f.read())

        config = ConfigParser()
        config.optionxform=str
        config.read(self.model_config)
        self.__logger.info(f"{config.sections()}")
        ret = dict()
        ret['JTVAE'] = JTNNVAE.config_dict(config['JTVAE'])
        #ret['PEAK_ENCODER'] = ms_peak_encoder_cnn.config_dict(config['PEAK_ENCODER'])
        ret['PEAK_ENCODER'] = config['PEAK_ENCODER']

        return ret

    def save_result_to_xlsx(self,sample_rate_list,result,file_name="area_sample_result.xlsx"):
        wb = openpyxl.Workbook()
        ws1 = wb.active
        ws1.append(['rate','number of times'])
        ws1.append(sample_rate_list)
        ws1.append(result)

        wb.save(file_name)

    
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

def analysisModel(trained_result_path,eval_mode=True,logger=None,loglevel=logging.WARN):
    if logger is None:
        rv = logging._checkLevel(loglevel)
        logger = getLogger(__name__)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(rv)
        stream_handler.setFormatter(logging.Formatter("[%(levelname)s]:%(message)s"))
        logger.addHandler(stream_handler)
        logger.setLevel(rv)

    AnalysisModel(trained_result_path,eval_mode,logger=logger)()

def main():
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = ArgumentParser()

    parser.add_argument('trained_result_path',type=str)
    parser.add_argument('--loglevel',default='WARN')

    args = parser.parse_args()

    proc = AnalysisModel(**args.__dict__)
    proc()

if __name__=="__main__":
    main()