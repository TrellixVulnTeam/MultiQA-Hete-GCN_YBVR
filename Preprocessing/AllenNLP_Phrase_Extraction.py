from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import datetime
import spacy

starttime = datetime.datetime.now()
predictor = Predictor.from_path("E:/Hetero_QA_data&model/models/elmo-constituency-parser-2020.02.10.tar.gz")#也可以下载到本地
nlp = spacy.load("en_core_web_sm", disable=['parser'])

def Phrase_in_Entity(phrase,Entity_list):#判断短语是否是实体或者是否是实体的一部分
    for entity in Entity_list:
        if phrase==str(entity) or phrase in str(entity):
            return 1
    return 0

def extract(word,Phrases,Entity_list,sentence):#抽取出短语
    Labels = ['NP','VP','PP','WHNP','S','SBAR','ADJP','DP','QP','NR','NT','SQ','ADVP']#待抽取的短语集合
    for child in word['children']:
        if child['nodeType'] in Labels and len(child['word'].split())>1 and len(child['word'].split())<len(sentence.split()):
            #找到所有包含1个单词以上的短语
            if child['word']not in Phrases :#并且该短语不属于实体
                Phrases.append(child['word'])
            extract(child,Phrases,Entity_list,sentence)#由于是树形结构，所以递归调用


def Phrase_Extraction(sentence):
    prediction = predictor.predict(sentence)
    Phrases = []
    doc = nlp(sentence)#实体识别
    Entity_list = doc.ents
    extract(prediction['hierplane_tree']['root'],Phrases,Entity_list,sentence)
    endtime = datetime.datetime.now()
    print (endtime - starttime)
    return Phrases
    
