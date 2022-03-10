#! -*- coding: utf-8 -*-
# SimCSE 中文测试
from utils import *
import sys
import numpy as np
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import jieba
jieba.initialize()

## 加载模型
from bert4keras.backend import keras
encoder = keras.models.load_model('test.model', compile=False)

## 加载数据
kkk_all = np.load(r"\t**内容1.npy")

# 基本参数
model_type, pooling, task_name, dropout_rate = "BERT", "first-last-avg", "ATEC", 0.3
dropout_rate = float(dropout_rate)

# bert配置
model_name = {
    'BERT': 'chinese_L-12_H-768_A-12',
    'RoBERTa': 'chinese_roberta_wwm_ext_L-12_H-768_A-12',
    'WoBERT': 'chinese_wobert_plus_L-12_H-768_A-12',
    'NEZHA': 'nezha_base_wwm',
    'RoFormer': 'chinese_roformer_L-12_H-768_A-12',
    'BERT-large': 'uer/mixed_corpus_bert_large_model',
    'RoBERTa-large': 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16',
    'NEZHA-large': 'nezha_large_wwm',
    'SimBERT': 'chinese_simbert_L-12_H-768_A-12',
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = r'D:\t**\simbert\%s\bert_config.json' % model_name
if model_type == 'NEZHA':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-691689' % model_name
elif model_type == 'NEZHA-large':
    checkpoint_path = '/root/kg/bert/%s/model.ckpt-346400' % model_name
else:
    checkpoint_path = r'D:\t**\simbert\%s\bert_model.ckpt' % model_name
dict_path = r'D:\t**\simbert\%s\vocab.txt' % model_name
print(dict_path)

# 建立分词器
if model_type in ['WoBERT', 'RoFormer']:
    tokenizer = get_tokenizer(
        dict_path, pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
else:
    tokenizer = get_tokenizer(dict_path)

### 数据embedding保存
if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64
datas_all = [ i[1] for i in kkk_all.tolist()]
# 测试相似度效果
data = datas_all
a_token_ids, b_token_ids, labels = [], [], []
texts = []

for d in data:
#     d=d[1]
    token_ids = tokenizer.encode(d, maxlen=maxlen)[0]
    a_token_ids.append(token_ids)
#     token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
#     b_token_ids.append(token_ids)
#     labels.append(d[2])
    texts.append(d)

a_token_ids = sequence_padding(a_token_ids)
# b_token_ids = sequence_padding(b_token_ids)
a_vecs = encoder.predict([a_token_ids, np.zeros_like(a_token_ids)],
                         verbose=True)
# b_vecs = encoder.predict([b_token_ids, np.zeros_like(b_token_ids)],
#                          verbose=True)
# labels = np.array(labels)

a_vecs = a_vecs / (a_vecs**2).sum(axis=1, keepdims=True)**0.5
np.save(r"simcse_datas_chinese.npy",a_vecs)

##测试效果


def most_similar(text, topn=10):
    """检索最相近的topn个句子
    """
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    vec = encoder.predict([[token_ids], [segment_ids]])[0]
    vec /= (vec**2).sum()**0.5
    sims = np.dot(a_vecs, vec)
    return [(kkk_all[i], sims[i]) for i in sims.argsort()[::-1][:topn]]
    
    

kk=["妲己不是坏女人","百变巴士","最美青春"]
mmm = []
for i in kk:
    results = most_similar(i, 10)
    mmm.append([i,results])
    print(i,results)
    titles = []
    pics = []
    for ii in results:
        titles.append(ii[0][1])
        pics.append(ii[0][2])
    print(titles, pics)
