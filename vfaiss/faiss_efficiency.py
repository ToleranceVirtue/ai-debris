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
    st = time.time()
    xb = np.random.random((samplenums, dim)).astype('float32')
    # xb[:, 0] += np.arange(samplenums) / 1000.
    # 随机生成查询数据
    print("gen train data cost:", time.time() - st)
    return xb

def genQueryDat(dim = 1000, querynums = 1000, seed=1234):
    np.random.seed(seed)
    st = time.time()
    xq = np.random.random((querynums, dim)).astype('float32')
    xq[:, 0] += np.arange(querynums) / 1000.
    print("gen search data cost:", time.time() - st)
    return xq

# trainDat = genTrainDat()
# queryDat = genQueryDat()

def build_faiss(dim=1000, nlist=1000, samplenums=int(1e6), querynums=int(1e3), isgpu=False):
    trainDat = genTrainDat(dim=dim, samplenums=samplenums)
    queryDat = genQueryDat(dim=dim, querynums=querynums)
    print('[*S] build index...')
    start = time.time()
    # index = faiss.IndexFlatL2(dim)      # build the index
    # quantizer = faiss.IndexFlatL2(dim)  # the other index
    # faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metrics)，
    # 分别为faiss.METRIC_L2 欧式距离、 faiss.METRIC_INNER_PRODUCT 向量内积
    # here we specify METRIC_L2, by default it performs inner-product search
    # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    
    index = faiss.index_factory(dim, "IVF64,Flat")
    
    if isgpu:
        try:
            print('[*] 尝试使用单GPU进行索引...')
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            index = gpu_index
        except Exception as e :
            print('[*] GPU索引失败:', e)
            pass
    # print(index.ntotal)
    
    assert not index.is_trained
    index.train(trainDat)
    assert index.is_trained
    index.add(trainDat)                 # add vectors to the index
    # print(index.ntotal)  # 加入了多少行数据
    end = time.time()
    btt = (end - start)*1000
    print('[*E] samplenums:%d, dim:%d, nlist:%d, cost:%4f ms' % (samplenums, dim, nlist, btt))
    idx_path = "{}_{}_{}.index".format(samplenums, dim, nlist)
    # You need to re-convert the index back to CPU with index_gpu_to_cpu before storing it.
    faiss.write_index(index, idx_path)
    
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024

    return btt, mem

def increa_faiss(add_num=int(1e4), dim=1000, nlist=1000, samplenums=int(1e6), querynums=int(1e3), isgpu=False):
    xb = genTrainDat(dim=dim, samplenums=add_num)
    idx_path = "{}_{}_{}.index".format(samplenums, dim, nlist)
    index = faiss.read_index(idx_path)
    start = time.time()
    index.add(xb)
    end = time.time()
    increa_time = (end - start)*1000
    print("increa:", index.ntotal, "cost:", increa_time)
    return increa_time   


def update_faiss(up_num=int(1e3), dim=1000, nlist=1000, samplenums=int(1e6), querynums=int(1e3), isgpu=False):
    idx_path = "{}_{}_{}.index".format(samplenums, dim, nlist)
    index = faiss.read_index(idx_path)
    indices = np.random.randint(index.ntotal, size=up_num)
    nxb = genTrainDat(dim=dim, samplenums=up_num)
    start = time.time()
    index.make_direct_map()
    index.update_vectors(np.array(indices).astype('int64'), nxb)
    end = time.time()
    up_time = (end - start)*1000
    print("total:", index.ntotal, "update:", up_num, "cost:",up_time)
    return up_time

def query_faiss(dim=1000, nlist=1000, k=100, samplenums=int(1e6), querynums=int(1e3)):
    queryDat = genQueryDat(dim=dim, querynums=querynums)
    print('[*S] query index...')
    idx_path = "{}_{}_{}.index".format(samplenums, dim, nlist)
    index = faiss.read_index(idx_path)
    start = time.time()
    D, I = index.search(queryDat, k)     # actual search
    end = time.time()
    qt = (end - start)*1000
    print('[*E] querynums:%d, dim:%d, nlist:%d, cost:%4f ms' % (querynums, dim, nlist, qt))
    return qt


def plot_idx_meas():
    """构建索引耗时"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    cidx_tts = []
    cmms = []
    samplenums = int(1e6)
    nlist = 1000
    for i, d in enumerate(range(100, 1100, 100)):
        cbtt, cm = build_faiss(dim=d, nlist=nlist, samplenums=samplenums, querynums=int(1e3), isgpu=False)
        xs.append(i)
        xlabes.append(d)
        cidx_tts.append(cbtt)
        cmms.append(cm)

    # print(xs)
    # print(idx_tts)
    plt.figure(1, figsize=(20,20))
    plt.subplot(2, 1, 1)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, cidx_tts, color="b", marker='o', linestyle="-", label="cpu cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.title('build cost time')
    plt.xlabel('dim')
    plt.ylabel('ms')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f M'))
    plt.plot(xs, cmms, color="b", marker='o', linestyle="-", label="cpu mems")   # 绘制曲线 y1
    plt.xticks(xs, xlabes)
    plt.title('build mm')
    plt.xlabel('dim')
    plt.ylabel('M')
    plt.grid()
    # plt.show()
    xpu = "cpu"
    pltname = "{}_build_idx_mem_sample{}_nlist{}.png".format(xpu, samplenums, nlist) 
    plt.savefig(pltname)

def plot_query_meas(querynums=1000):
    """检索耗时"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    tts = []
    
    for i, d in enumerate(range(100, 1100, 100)):
        qt = query_faiss(dim=d, nlist=1000, k=1000, samplenums=int(1e6), querynums=querynums)
        xs.append(i)
        xlabes.append(d)
        tts.append(qt)

    # print(xs)
    # print(idx_tts)
    plt.figure(figsize=(20,20))
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, tts, color="blue", marker='o', linestyle="-", label="query cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.title('query cost time')
    plt.xlabel('dim')
    plt.ylabel('ms')
    plt.grid()
    # plt.show()
    plt.savefig("dim_query1.png") 

def plot_add_meas():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    tts = []

    for i, n in enumerate(range(int(1e4), int(1.2e6), int(5e4))):
        rt = increa_faiss(n, dim=1000, nlist=1000, samplenums=int(1e6), querynums=int(1e3), isgpu=False)
        xs.append(i)
        xlabes.append(n)
        tts.append(rt)

    # print(xs)
    # print(idx_tts)
    plt.figure(figsize=(20,20))
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, tts, color="blue", marker='o', linestyle="-", label="add dat cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.title('faiss add cost time')
    plt.xlabel('dim')
    plt.ylabel('ms')
    plt.grid()
    # plt.show()
    plt.savefig("add_faiss.png")

def plot_update_meas():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    tts = []

    for i, n in enumerate(range(int(1e4), int(1.1e6), int(5e4))):
        rt = update_faiss(n, dim=1000, nlist=1000, samplenums=int(1e6), querynums=int(1e3), isgpu=False)
        xs.append(i)
        xlabes.append(n)
        tts.append(rt)

    # print(xs)
    # print(idx_tts)
    plt.figure(figsize=(20,20))
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, tts, color="blue", marker='o', linestyle="-", label="update dat cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.title('faiss update cost time')
    plt.xlabel('dim')
    plt.ylabel('ms')
    # plt.show()
    plt.savefig("up_faiss.png")

if __name__ == '__main__':
    # 构建faiss
    rt,mem = build_faiss(dim=1000, samplenums=int(1e6))
    # print(rt, mem)
    
    # query
    # rt = query_faiss()
    # print(rt)
    
    # 构建索引耗时
    #plot_idx_meas()

    #plot_query_meas(querynums=1)
    #rt = increa_faiss(10000)
    #print(rt)
    #rt = update_faiss(10000)
    #print(rt)
    
    #plot_add_meas()
    #plot_update_meas()
    pass
