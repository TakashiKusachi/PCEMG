import os
from pathlib import Path
from multiprocessing import Pool
from rdkit import Chem
import pickle
import numpy as np

class GNPSParser(object):
    def __init__(self):
        pass

    def __call__(self,path_list):
        assert len(path_list) == 1
        path = path_list[0]
        with Pool() as p:
            ret = p.map(self.parse,self.sep_data(path))
        return ret

    def sep_data(self,path):
        """
        ファイル内にある複数のデータを一個ずつ取り出すメソッド
        """
        begins = []
        ends = []

        with open(path,'r') as f:
            lines = f.readlines()
        print("liens : "+str(len(lines)))
        for i,line in enumerate(lines):
            if 'BEGIN IONS' in line:
                begins.append(i)
            elif 'END IONS' in line:
                ends.append(i)

        assert len(begins) == len(ends),""
        data_index = zip(begins,ends)
        for begin,end in data_index:
            yield lines[begin+1:end]

    def parse(self,data):
        attr = {}
        mz = []
        inten = []
        for num,line in enumerate(data):
            sped = line.split('=',1)
            if len(sped) == 2:
                attr[sped[0]] = sped[1]
                continue
            else:
                sped = line.split('\t',1)
                mz.append(float(sped[0]))
                inten.append(float(sped[1]))
        attr['x'] = np.asarray(mz)
        attr['y'] = np.asarray(inten)
        return attr