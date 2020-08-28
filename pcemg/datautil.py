#!/bin/bash
""" 質量スペクトルのデータセットライブラリ

"""

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import os, random,math
from tqdm import tqdm
import pickle
import rdkit
from torch_jtnn.mol_tree import MolTree
from torch_jtnn.datautils import tensorize,set_batch_nodeID

#import mysql
#from mysql import connector
from getpass import getpass,getuser

import warnings
from multiprocessing import Pool

class MS_Dataset(object):
    
    QUERY= """select smiles,file_path from massbank where ms_type="MS" and instrument_type="EI-B"; """
    """
    """
    def __init__(self,vocab,host,database,batch_size,user=None,passwd=None,port=3306,
                 save="./MS_Dataset.pkl"):
        if os.path.exists(save):
            with open(save,"rb") as f:
                terget_list = pickle.load(f)
        else:
            terget_list = self.dataload(host,database,batch_size,user,passwd,port)
            with open(save,"wb") as f:
                pickle.dump(terget_list,f)
        self.max_spectrum_size = max([len(one[0]) for one in terget_list])
        self.vocab = vocab
        self.dataset = terget_list
        self.batch_size = batch_size
        self.shuffle = True
        
    def dataload(self,host,database,batch_size,user=None,passwd=None,port=3306):
        terget_list = []
        try:
            if not isinstance(user,str):
                user = raw_input("user")
            if not isinstance(passwd,str):
                passwd = getpass()
            connect = connector.connect(host=host,user=user,password=passwd,port=port,database=database)
            cursor = connect.cursor()
            cursor.execute(MS_Dataset.QUERY)
            data_list = cursor.fetchall()
        except mysql.connector.Error as e:
            print("Something went wrong: {}".format(e))
            sys.exit(1)
        finally:
            if connect: connect.close()
            if cursor: cursor.close()
        
        succes = 0
        fault = 0
        max_spectrum_size = 0
        for one in tqdm(data_list):
            ret = get_spectrum(*one)
            if ret is not None:
                max_spectrum_size = max(len(ret[0]),max_spectrum_size)
                terget_list.append(ret)
                succes+=1
            else:
                fault += 1
        print("success {},fault {}".format(succes,fault))
        return terget_list,max_spectrum_size
    def __len__(self):
        return len(self.dataset)
        
    def __iter__(self):
        if self.shuffle: 
            random.shuffle(self.dataset) #shuffle data before batch
            
        batches = [zip(*self.dataset[i : i + self.batch_size]) for i in range(0, len(self.dataset), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()
        dataset = MS_subDataset(batches,self.vocab,self.max_spectrum_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
        for b in dataloader:
            yield b
        del batches, dataset, dataloader

def is_select(one):
    """ データの選定処理(default)
    Args:
        one (dict): 
    
    Returns:
        bool: 与えられたデータが許可された場合はtrue、そうでなければfalse
    
    Note:
        条件式は順序に影響します。

    """
    return one["smiles"]!="N/A" and one["ms_type"]=="MS" and one["instrument_type"]=="EI-B" and "ionization_energy" in one and one["ionization_energy"]=="70 eV" and one["ion_mode"]=="POSITIVE" and "2H" not in one["smiles"] #and round(rdMolDescriptors._CalcMolWt(Chem.MolFromSmiles(one['smiles']))) in one['peak_x'].astype(int) 

def data_catch(one):
    """
    """
    mol = molfromsmiles(one["smiles"])
    return (one["peak_x"],one["peak_y"],mol)

def dataset_load(path,vocab,batch_size,train_validation_rate,select_fn=is_select,save="./MS_Dataset.pkl"):
    """ データセットロード関数
        pickle圧縮された生データセットから、不要なデータの除去と成型、データセットサブクラス（MS_Dataset_pickle）の生成

    Args:
        path (str):
        vocab(str):
        batch_size (int):
        train_validation_rate (float):
        select_fn (function):
    """
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    if os.path.exists(save):
        with open(save,"rb") as f:
            dataset = pickle.load(f)
    else:
        with open(path,"rb") as f:
            fullset = pickle.load(f)
            fullset = [one for one in fullset if select_fn(one)]
        
        dataset = []
        p = Pool()
        try:
            dataset = p.map(data_catch,fullset)
        finally:
            p.close()
            p.join()
            del p
                
        with open(save,"wb") as f:
            pickle.dump(dataset,f)
        
    max_spectrum_size = max([len(one[0]) for one in dataset])
    
    total_size = len(dataset)
    train_size = int(total_size * train_validation_rate)
    random.shuffle(dataset)
    train_dataset = MS_Dataset_pickle(dataset[:train_size],vocab,batch_size,max_spectrum_size)
    valid_dataset = MS_Dataset_pickle(dataset[train_size:],vocab,batch_size,max_spectrum_size)
    return (train_dataset,valid_dataset)
        
class MS_Dataset_pickle(object):
    def __init__(self,dataset,vocab,batch_size,max_spectrum_size,shuffle=True):
        
        self.dataset = dataset
        self.max_spectrum_size = max_spectrum_size
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __len__(self):
        return math.floor(len(self.dataset)/self.batch_size)
    
    def __iter__(self):
        if self.shuffle: 
            random.shuffle(self.dataset) #shuffle data before batch
        batches = [list(zip(*self.dataset[i : i + self.batch_size])) for i in range(0, len(self.dataset), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()
        dataset = MS_subDataset(batches,self.vocab,self.max_spectrum_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
        for b in dataloader:
            yield b
        del batches, dataset, dataloader
        
class MS_subDataset(Dataset):
    def __init__(self,datasets,vocab,max_spectrum_size):
        self.datasets = datasets
        self.vocab = vocab
        self.max_spectrum_size=max_spectrum_size
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        spec_x = [np.pad(one,(0,self.max_spectrum_size-len(one)),"constant",constant_values=0) for one in self.datasets[idx][0]]
        spec_y = [np.pad(one,(0,self.max_spectrum_size-len(one)),"constant",constant_values=0) for one in self.datasets[idx][1]]
        return tensorize(self.datasets[idx][2], self.vocab, assm=True)+(torch.tensor(spec_x),)+(torch.tensor(spec_y),)
    
def molfromsmiles(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree
# 
def get_spectrum(smiles,path):
    x_list=[]
    y_list=[]
    try:
        mol = molfromsmiles(smiles)
            
    except AttributeError as e:
        warnings.warn("Entered An SMILES that does not meet the rules")
        return None
    
    with open(path,"r") as f:
        lines = f.read().split("\n")
        num = [i for i,one in enumerate(lines) if one.split(": ")[0] == "PK$PEAK"][0] # Perhaps it is faster to use for.
        for one in lines[num+1:-2]:
            x,y,y2 = one.split(" ")[2:]
            x_list.append(float(x))
            y_list.append(float(y))
    return np.asarray(x_list,dtype=np.float32),np.asarray(y_list,dtype=np.float32),mol
if __name__=="__main__":
    print("test")