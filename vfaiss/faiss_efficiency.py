#! -*- coding: utf-8 -*-
from math import floor
import os
import sys
import time
import numpy as np
import faiss                   
# make faiss available
import os
import psutil
import mkl
mkl.get_max_threads()

def genTrainDat(dim = 1000, samplenums = 1000000, seed=1234):
    np.random.seed(seed)
    # make reproducible
    xb = np.random.random((samplenums, dim)).astype('float32')
    xb[:, 0] += np.arange(samplenums) / 1000.
    # 随机生成查询数据
    return xb

def genQueryDat(dim = 1000, querynums = 1000, seed=1234):
    np.random.seed(seed)
    xq = np.random.random((querynums, dim)).astype('float32')
    xq[:, 0] += np.arange(querynums) / 1000.
    return xq

# trainDat = genTrainDat()
# queryDat = genQueryDat()

def build_faiss(dim=1000):
    trainDat = genTrainDat(dim=dim)
    queryDat = genQueryDat(dim=dim)
    print('创建索引...')
    start = time.time()
    index = faiss.IndexFlatL2(dim)      # build the index
    print(index.is_trained)
    index.add(trainDat)                 # add vectors to the index
    print(index.ntotal)
    end = time.time()
    total_time = (end - start)*1000
    print('用时:%4f 毫秒' % total_time )

if __name__ == '__main__':
    build_faiss()
