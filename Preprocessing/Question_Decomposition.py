import json
import pandas
import datetime
import string
from Get_sub_questions import Clauses_Extraction, Conjunctions_Extraction, Phrases_Extraction
punctuation_string = string.punctuation#remove the punctuation

input_data = json.load(open('./data/hotpot_dev_distractor_v1.json', 'r',encoding='utf-8'))
#input_data = json.load(open('E:/Hetero_QA_data&model/data/new 1.json', 'r',encoding='utf-8'))
Extracted, Questions,ids = [], [], []
sents, labels, titles = [], [], []
wrong_result = 0

for i, data in enumerate(input_data):
    gold_paras = [para for para , _  in data['supporting_facts']]
    print(i,'/',len(input_data))
    starttime = datetime.datetime.now()
    ##get the complex question and remove the punctuation
    for title, sentences in data['context']:
        _id = data['_id']
        label = int(title in gold_paras)#标题是否在support fact里
        context = " ".join(sentences)
        question = data['question']
        question = ' '.join(question.split())
        for i in punctuation_string:#remove the punctuation
            question = question.replace(i, '')
        sentence = question
        #Decompose the question based on different types
        if data['type']=='comparison':
            Sub_Questions = Conjunctions_Extraction(question)#Get conjunctions
            if len(Sub_Questions)==1:#if we can not find the conjunctions then decompose the question into Phrases.
                Sub_Questions = []
                Phrases = Phrases_Extraction(question)
                for i,phrase in enumerate(Phrases):
                    sentence = sentence.replace(phrase,"[Answer of Question{}]".format(i))
                    Sub_Questions.append(phrase)
                Sub_Questions.append(sentence.strip())
        else:
            Clauses = Clauses_Extraction(question)#Get clauses
            Sub_Questions = []
            for i,clause in enumerate(Clauses):
                sentence = sentence.replace(clause,"[Answer of Question{}]".format(i))
                Sub_Questions.append(clause)
            Sub_Questions.append(sentence.strip())
            Sub_Questions = list(set(Sub_Questions))
            if len(Sub_Questions) == 1:#if we can not find the clauses then decompose the question into Phrases.
                Sub_Questions = []
                Phrases = Phrases_Extraction(question)
                for i,phrase in enumerate(Phrases):
                    sentence = sentence.replace(phrase,"[Answer of Question{}]".format(i))
                    Sub_Questions.append(phrase)
                Sub_Questions.append(sentence.strip())
        for sent in sentences:
            for sub_question in Sub_Questions:
                labels.append(label)
                titles.append(title)
                sents.append(sent)
                Extracted.append(sub_question)
                Questions.append(question)
                ids.append(_id)
        endtime = datetime.datetime.now()
        print (endtime - starttime)
df = pandas.DataFrame({'id':ids,'Title': titles, 'Sentences': sents, 'Label': labels,'Sub_question':Extracted, 'Question':Questions})
df.to_csv('./result/Question_Decoposition.csv')