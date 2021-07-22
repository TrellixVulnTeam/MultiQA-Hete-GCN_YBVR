import datetime
import string
from Get_Conjunctions import conjunctions
from Get_Clauses import clause
import json
import pandas as pd

punctuation_string = string.punctuation#remove the punctuation

output_file = './result/sub_sentences.csv'

input_data = json.load(open('./data/hotpot_dev_distractor_v1.json', 'r',encoding='utf-8'))

types,phra,questions,docs = [], [], [],[]
sents, labels, titles= [], [], []
ids = []
all_docs = []
total = len(input_data)
for i,data in enumerate(input_data):
    gold_paras = [para for para , _  in data['supporting_facts']]
    _id = data['_id']
    print(i,"/",total)
    starttime = datetime.datetime.now()           
    for title, sentences in data['context']:#get all the sentences
        for sent in sentences:
            all_docs.append(sent)
    for title, sentences in data['context']:
        label = int(title in gold_paras)#Check whether the tile is in the supporting_facts or not
        context = " ".join(sentences)
        question = data['question']
        for i in punctuation_string:#remove the punctuation
            question = question.replace(i, '')
        question = question.strip()
        sub_sentences1 = conjunctions(question)#split conjunctions
        sub_sentences2 = clause(question)#extract clauses
        
        sub_sentences = sub_sentences1+sub_sentences2
        for sent in sentences:
            for sub_sent in sub_sentences:
                if sub_sent == question:#comtimes a whole sentence is also a clause but we do not need it
                    continue
                labels.append(label)
                titles.append(title)
                sents.append(sent)
                questions.append(question)
                phra.append(sub_sent)
                ids.append(_id)
                docs.append(all_docs)
    
    endtime = datetime.datetime.now()
    print (endtime - starttime)

df = pd.DataFrame({'id':ids,'Title': titles, 'Sentences': sents, 'Label': labels,'Sub_question':phra, 'Question':questions,"Context":docs})
df.to_csv(output_file)