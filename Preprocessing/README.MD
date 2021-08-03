DATA:
hotpotQA

Models:
# 短语抽取，现在不需要了
https://allennlp.s3.amazonaws.com/models/elmo-constituency-parser-2020.02.10.tar.gz

BERT pretrained model for text match:
https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz

For extract clauses and conjunctions:
Berkeley Neural Parser
Spacy Model en_core_web_sm

Run:
python Get_sub_questions.py
# extract all the conjunctions and clauses from the complex questions

python Sentence_rank.py
# 将sentence_phrase_pairs.csv的数据按照"[CLS]" + phrase + "[SEP]" + sentence + "[SEP]"的形式送入bert预训练模型中进行文本匹配，找出文章中与问题中短语语义相似度较大的句子
# 作为candidate sentence进行下一步工作
