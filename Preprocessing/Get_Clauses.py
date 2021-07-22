import nltk
from nltk import Tree
import benepar
import spacy
nlp = spacy.load("./en_core_web_sm-3.1.0")

def connect(whnp,subtexts):#把单独的从句代词和后面的部分连在一起
    length = len(subtexts)
    for wp in whnp:
        i=0
        while(i<length-1):
            if (subtexts[i].strip() in wp or wp in subtexts[i].strip()) and len(subtexts[i].split())<=2:#避免出现单独的疑问词被拿出来
                subtexts[i+1]=subtexts[i]+subtexts[i+1]
                del(subtexts[i])
                length = len(subtexts)
            i = i+1
    return subtexts

def not_in_entity(Entity_list,clause):
    for ent in Entity_list:
        if clause in str(ent):
            return 0
    return 1

def clause(sentence):
    clause_level_list = ["S","SBAR", "SINV", "SQ"]
    parser = benepar.Parser("./benepar_en3")
    parse_tree = str(parser.parse(sentence))#得到解析树
    
    # nlp = spacy.load("en_core_web_sm", disable=['parser'])
    doc = nlp(sentence)
    Entity_list = list(doc.ents)#抽取句子中的实体
    
    t = nltk.tree.ParentedTree.fromstring(parse_tree)
    #t.pretty_print()
    subtexts = []
    for subtree in t.subtrees():
        if subtree.label()in clause_level_list and not_in_entity(Entity_list,' '.join(subtree.leaves())):
            #从句可以包含实体，实体决不能包含从句
            subtexts.append(' '.join(subtree.leaves()))
    for i in reversed(range(len(subtexts)-1)):
        subtexts[i] = subtexts[i][0:subtexts[i].find(subtexts[i+1])]
        if subtexts[i] =='':
            del subtexts[i]
    whnp = ['which','what','that','who','where','whom']
    clauses = connect(whnp,subtexts)
    for clause in clauses:
        sentence = sentence.replace(clause,"")
    if len(sentence)>0: 
        clauses.append(sentence)
    return clauses