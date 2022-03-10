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

def measure(d = 768, nb = 1000000, nq = 1000, nlist = 100, k = 5):
    # print('faiss-cpu 速度测试'.center(40,'-'))
    # print('正在生成模拟数据...')
    # dimension
    # database size
    # nb of queries
    # nlist 聚类中心的个数
    # we want to see 4 nearest neighbors
    np.random.seed(1234)
    # make reproducible
    # 生成随机数据
    xb = np.random.random((nb, d)).astype('float32')
    # 然后...?这是干嘛呢 
    # 注：让第一列有点规律性，这样才能比较相似度；
    xb[:, 0] += np.arange(nb) / 1000.

    # 随机生成查询数据
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # print('数据总条数:%d, 维度:%d, 查询个数：%d' % (nb, d, nq) )
    # 保存数据
    #np.save('dat.npy',xb)
    #sys.exit()
    # print('创建索引...')
    start = time.time()
    index = faiss.IndexFlatL2(d)   # build the index
    # print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    # print(index.ntotal)
    end = time.time()
    total_time = (end - start)*1000
    # print('用时:%4f 毫秒' % total_time )

    # 查看当前进程使用的内存情况
    process = psutil.Process(os.getpid())
    # print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')


    
    # print('开始查询(Top %d)...' % k)
    start = time.time()
    D, I = index.search(xq, k)     # actual search
    # 询返回两个numpy array对象D和I。
    # D表示与相似向量的距离(distance)，维度，I表示相似用户的ID。
    #print(I.shape)
    #print(D.shape)
    # print(I[:5]) # neighbors of the 5 first queries
    # print(D[:5]) # neighbors of the 5 last queries

    end = time.time()
    total_time = (end - start)*1000
    # print('用时:%4f 毫秒' % total_time )
    # print('-'*40)
    #-----------------------------------------

    # print('IndexIVFFlat方法对比...')
    start = time.time()
                      
    # print('开始索引，聚类中心个数：%d ......' % nlist)
    quantizer = faiss.IndexFlatL2(d)  # the other index
    # faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metrics)，
    # 分别为faiss.METRIC_L2 欧式距离、 faiss.METRIC_INNER_PRODUCT 向量内积
    # here we specify METRIC_L2, by default it performs inner-product search
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 尝试使用gpu
    try:
        # print('尝试使用单GPU进行索引...')
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        index = gpu_index
    except Exception as e :
        # print('GPU索引失败:', e)
        pass


    assert not index.is_trained
    index.train(xb)
    assert index.is_trained
    index.add(xb)                  # add may be a bit slower as well

    end = time.time()
    index_total_time = (end - start) * 1000
    # print('索引用时:%4f 毫秒' % index_total_time)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    # print('Used Memory:', mem, 'MB')
    # print('开始查询...')
    start = time.time()

    D, I = index.search(xq, k)     # actual search
    # print(I[:5])                 # neighbors of the 5 last queries

    #index.nprobe = 10              # default nprobe is 1, try a few more
    #D, I = index.search(xq, k)
    #print(I[:5])                  # neighbors of the 5 last queries

    end = time.time()
    total_time = (end - start) * 1000
    # print('用时:%4f 毫秒' % total_time)
    return floor(index_total_time), floor(total_time), floor(mem)
#-----------------------------------------

def plot_idx_meas():
    """构建索引耗时"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    idx_tts = []
    tts = []
    mms = []

    for i, d in enumerate(range(100, 1000, 100)):
        idx_tt, tt, mm = measure(d = d, nb = 100000, nq = 1000, nlist = 100)
        print(i, idx_tt, tt, mm)
        xs.append(i)
        xlabes.append(d)
        idx_tts.append(idx_tt)
        tts.append(tt)
        mms.append(mm)

    # print(xs)
    # print(idx_tts)
    plt.figure()
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, idx_tts, color="blue", linestyle="-", label="build idx cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.show()


def plot_query_meas():
    """检索耗时"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    idx_tts = []
    tts = []
    mms = []

    for i, nq in enumerate(range(100, 1000, 100)):
        idx_tt, tt, mm = measure(d = 1000, nb = 1000000, nq = nq, nlist = 100)
        print(i, idx_tt, tt, mm)
        xs.append(i)
        xlabes.append(nq)
        idx_tts.append(idx_tt)
        tts.append(tt)
        mms.append(mm)

    # print(xs)
    # print(idx_tts)
    plt.figure()
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, tt, color="blue", linestyle="-", label="query idx cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.show()

def plot_nlist_meas():
    """构建节点耗时"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs = []
    xlabes = []
    idx_tts = []
    tts = []
    mms = []

    for i, nlist in enumerate(range(100, 1000, 100)):
        idx_tt, tt, mm = measure(d = 1000, nb = 1000000, nq = 1000, nlist = nlist)
        print(i, idx_tt, tt, mm)
        xs.append(i)
        xlabes.append(nlist)
        idx_tts.append(idx_tt)
        tts.append(tt)
        mms.append(mm)

    # print(xs)
    # print(idx_tts)
    plt.figure()
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f ms'))
    plt.plot(xs, tt, color="blue", linestyle="-", label="query nlist cost")   # 绘制曲线 y1
    # plt.plot(xs, tts, label="query idx cost")     # 绘制曲线 y2
    plt.xticks(xs, xlabes)
    plt.show()

if __name__ == '__main__':
    plot_idx_meas()
    # plot_query_meas()
    # plot_nlist_meas()
