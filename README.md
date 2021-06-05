# MultiQA-Hete-GCN
Dr WU's Hete-GCN to solve Multi-hop Question

requirements:
torch
dgl-cul111#图神经网络的包，注意按照自己的cuda版本来下载
matplotlib
networkx
spacy#注意要下载en_core_web_sm这个模型

在hotpotqa官网下载数据集后，先运行convert-qa-into-squad.py转换成squad形式的数据集。
然后运行Build_Graph

python convert-qa-into-squad.py
python Build_Graph.py
