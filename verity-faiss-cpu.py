#! -*- coding: utf-8 -*-

from ast import For
import faiss
import numpy as np

d = 1000
# 向量维度
nb = int(1e4)
# 待索引向量size

nq = 10
# 查询向量size

index = faiss.IndexFlatL2(d)
# 建立索引
print(index.is_trained)        
# 输出true

# 随机种子确定
np.random.seed(1234)
Ntime = int(1e1)

for _ in range(Ntime):
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    index.add(xb)


#为了使随机产生的向量有较大区别进行人工调整向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


# 索引中添加向量
print(index.ntotal)            
# 输出100000