import os
from glob import glob
from pathlib import Path
from multiprocessing import Pool
from rdkit import Chem
import pickle
import numpy as np

class default_copy(object):
    def __init__(self,key=None):
        self.key = key
    
    def __call__(self,_dict,_data):
        assert self.key is not None
        _dict[self.key] = _data

class smiles_copy(default_copy):
    def __call__(self,_dict,_data):
        smiles = _data

        if smiles != 'N/A':
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)

        _dict['smiles'] = smiles

class massspectrometry_copy(default_copy):
    def __call__(self,_dict,_data):
        __massspectrometry_captions={
            "MS_TYPE":                  default_copy('ms_type'),
            "ION_MODE":                 default_copy('ion_mode'),
            "IONIZATION_ENERGY":        default_copy('ionization_energy'),
            "COLLISION_ENERGY":         default_copy('collision_energy'),
            "COLLISION_GAS":            default_copy('collision_gas'),
        }

        _caption,_data = _data.split(' ',1)
        proc = __massspectrometry_captions.get(_caption)
        if proc is not None:
            proc(_dict,_data)

class MassBankParser(object):
    def __init__(self):
        pass

    def __call__(self,path_list):
        with Pool() as p:
            ret = p.map(self.parse,path_list)
        return ret

    def parse(self,path):
        __mass_bank_captions={
            "AUTHORS":                  default_copy('authors'),
            "AC$INSTRUMENT":            default_copy('instrument'),
            "AC$INSTRUMENT_TYPE":       default_copy('instrument_type'),
            "CH$NAME":                  default_copy('name'),
            "CH$SMILES":                smiles_copy(),
            "AC$MASS_SPECTROMETRY":     massspectrometry_copy(),
        }

        ret = {}
        with open(path,'r') as f:
            _temp = f.read().split('\n')
        
        for num,line in enumerate(_temp):
            _caption = line.split(': ',1)
            proc = __mass_bank_captions.get(_caption[0])
            if proc is not None:
                proc(ret,_caption[1])
        return ret