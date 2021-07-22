import argparse
import glob
import logging
import os
import random
import numpy as np
import pandas
import torch
import json
from os.path import join
from collections import Counter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

# This line must be above local package reference
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from feature_extraction import convert_examples_to_features,HotpotQAProcessor

logger = logging.getLogger(__name__)



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(args, model, tokenizer,prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset,features = load_and_cache_examples(args, 'hotpotqa', tokenizer, evaluate=True)

    eval_batch_size = 32
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    #eval_sampler = ExampleDataset(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % 32)

    eval_loss, eval_accuracy = 0.0, 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    preds = None
    out_label_ids = None
    predictions = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluation"):
        model.eval()
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        predictions.append(logits)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        # writer.write("%s = %s\n" % (key, str(result[key])))

    logger.info("***** Writting Predictions ******")
    logits0 = np.concatenate(predictions, axis=0)[:, 0]
    logits1 = np.concatenate(predictions, axis=0)[:, 1]
    ground_truth = [fea.label_id for fea in features]
    score = pandas.DataFrame({'logits0': logits0, 'logits1': logits1, 'label': ground_truth})
    score.to_csv('pred_score.csv')
    return score, eval_loss, eval_accuracy


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def rank_sentences(data, pred_score):
    sentences = []
    logits = np.array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]
    for (i, row) in data.iterrows():
        score = pred_score.loc[i, 'prob'].item()
        if score>0.5:
            sentences.append({'id':row['id'],"Sub_question":row['Sub_question'],"Context":[row['Title'],row['Sentences']]})
   
    file_name = 'selected_sentences.json'
    with open(file_name,'w') as file_object:
        json.dump(sentences,file_object)
    print(len(sentences))


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = HotpotQAProcessor()
    output_mode = "classification"
    label_list = processor.get_labels()
    examples = processor.get_examples(args.input_data)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    #elif output_mode == "regression":
        #all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset,features

def get_dev_paras(data, pred_score):
    logits = np.array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]
    Paragraphs = dict()
    cur_ptr = 0
    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr
        all_paras = []
        selected_paras = []
        while cur_ptr < tem_ptr + len(case['context']):
            score = pred_score.loc[cur_ptr, 'prob']
            all_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            if score >= 0.5:  
                selected_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            cur_ptr += 1
        sorted_all_paras = sorted(all_paras, key=lambda x: x[0], reverse=True)
        sorted_selected_paras = sorted(selected_paras, key=lambda x: x[0], reverse=True)
        Paragraphs[key] = [p[1] for p in sorted_selected_paras]
        while len(Paragraphs[key]) < 3:
            if len(Paragraphs[key]) == len(all_paras):
                break
            Paragraphs[key].append(sorted_all_paras[len(Paragraphs[key])][1])

    Selected_paras_num = [len(Paragraphs[key]) for key in Paragraphs]
    print("Selected Paras Num:", Counter(Selected_paras_num))

    file_name = 'Devset.json' #通过扩展名指定文件存储的数据为json格式
    with open(file_name,'w') as file_object:
        json.dump(Paragraphs,file_object)


def get_train_paras(source_data, score):
    # + Negative Sample.
    logits = np.array([score['logits0'], score['logits1']]).transpose()
    score['prob'] = softmax(logits)[:, 1]
    score = np.array(score['prob'])
    Paragraphs = dict()
    ptr = 0
    for case in tqdm(source_data):
        key = case['_id']
        Paragraphs[key] = []
        para_ids = []
        gold = set([para[0] for para in case['supporting_facts']])

        for i, para in enumerate(case['context']):
            if para[0] in gold:
                Paragraphs[key].append(para)
                para_ids.append(i)

        tem_score = score[ptr:ptr + len(case['context'])]
        ptr += len(case['context'])
        sorted_id = sorted(range(len(tem_score)), key=lambda k: tem_score[k], reverse=True)

        for i in sorted_id:
            if i not in para_ids:
                Paragraphs[key].append(case['context'][i])
                break
    
    file_name = 'Trainset.json' #通过扩展名指定文件存储的数据为json格式
    with open(file_name,'w') as file_object:
        json.dump(Paragraphs,file_object)


def set_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_ckpt", default='./model/pytorch_model.bin', type=str, help="evaluation checkpoint")
    parser.add_argument("--raw_data", default='./data/hotpot_dev_distractor_v1.json', type=str, help="raw data for processing")
    parser.add_argument("--input_data", default='./result/sub_sentences.csv', type=str, help="source data for processing")#处理过的数据
    parser.add_argument("--data_dir", default = './data', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
    parser.add_argument("--split", default='', type=str)
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,)
    parser.add_argument("--do_lower_case", action='store_true',help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()

    args.device = torch.device("cuda")

    processor = HotpotQAProcessor()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
   
    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load('./model/pytorch_model.bin')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=model_state_dict)
    model.cuda()
    model = torch.nn.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

    score, eval_loss, eval_accuracy = evaluate(args, model, tokenizer,prefix="")
    print("loss:",eval_loss, "acc:",eval_accuracy)
    input_data = pandas.read_csv('./result/sub_sentences.csv')
    #raw_data = json.load(open('E:/Hetero_QA_data&model/data/new 1.json', 'r',encoding='utf-8'))
    
    #if args.split == 'dev':
        #get_dev_paras(raw_data, score)
    #elif args.split == 'train':
        #get_train_paras(raw_data, score)
    # load source data
    
    rank_paras_dict = rank_sentences(raw_data,input_data, score)
    #json.dump(rank_paras_dict, open(join(args.data_dir, 'sent_ranking.json'), 'w'))