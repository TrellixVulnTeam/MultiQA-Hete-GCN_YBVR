from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas
import json

input_data = 'E:/Hetero_QA_data&model/data/sub_sentences.csv'
df = pandas.read_csv(input_data)

sentences = []

for (i, row) in df.iterrows():
    sentence1 = row['Sub_question']
    sentence2 = row['Sentences']
    score = fuzz.partial_ratio(sentence1,sentence2)
    if score>=50:
        sentences.append({'id':row['id'],"Sub_question":row['Sub_question'],"Context":[row['Title'],row['Sentences']]})

print(len(sentences))
file_name = 'fuzzy_match.json' #通过扩展名指定文件存储的数据为json格式
with open(file_name,'w') as file_object:
    json.dump(sentences,file_object)