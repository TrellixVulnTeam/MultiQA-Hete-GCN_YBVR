import spacy
import json
import collections
import dgl
from dgl import DGLGraph
import dgl.function as fn
import torch
import numpy as np
import os

import networkx as nx
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

json_filename = 'C:/Users/Administrator/MultiQA-Hete-GCN/data/hotpot-all/train.json'
questions = []
maxlen=0


with open(json_filename) as f:
    dict_data = json.load(f)
    length = len(dict_data['data'])
    for i in range(length):
        question = dict_data['data'][i]['paragraphs'][0]['qas'][0]['question']#读取数据里的questions
        questions.append(question)

def transfer_n_e(nodes, edges):

    num_nodes = len(nodes)
    new_edges = []
    for e in edges:
        new_edges.append(e) 
    return num_nodes, new_edges

all_graphs = []
parser = spacy.load('en_core_web_sm')

for i, sent in enumerate(questions):
    sent = questions[i]
    doc = parser(sent)
    sent = ' '.join([token.text for token in doc])
    parse_rst = doc.to_json()
    nodes = collections.OrderedDict()
    edges = []
    edge_type = []
    for i_word, word in enumerate(parse_rst['tokens']):
        if i_word not in nodes:#添加自环
            nodes[i_word] = len(nodes) 
            edges.append( [i_word, i_word] )
            edge_type.append(0)
        if word['head'] not in nodes:
            nodes[word['head']] = len(nodes) 
            edges.append( [word['head'], word['head']] )
            edge_type.append(0)
        #添加双向依赖关系
        edges.append( [word['head'], word['id']] )
        edge_type.append(1)
        edges.append( [word['id'], word['head']] )
        edge_type.append(2)

    num_nodes, tran_edges = transfer_n_e(nodes, edges)
    #建图
    G = dgl.DGLGraph()
    G = G.to('cuda:0')
    G.add_nodes(num_nodes)
    G.add_edges(list(zip(*tran_edges))[0],list(zip(*tran_edges))[1])

    edge_norm = []
    for e1, e2 in tran_edges:
        if e1 == e2:
            edge_norm.append(1)
        else:
            edge_norm.append(1/(G.in_degree(e2) - 1))


    edge_type = torch.from_numpy(np.array(edge_type)).cuda()
    edge_norm = torch.from_numpy(np.array(edge_norm)).unsqueeze(1).float().cuda()

    G.edata.update({'rel_type': edge_type,})
    G.edata.update({'norm': edge_norm})
    all_graphs.append(G)


#可视化第一个句子的依存分析图
plt.figure(figsize=(14, 6))
g = all_graphs[0]
g = g.to('cpu')#数据放在CPU上才能可视化
nx.draw(g.to_networkx(), with_labels=True)
plt.show()