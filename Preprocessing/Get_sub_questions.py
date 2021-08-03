import nltk
from nltk import Tree
import benepar
import spacy
from allennlp.predictors.predictor import Predictor
# import allennlp_models.structured_prediction
import itertools
import spacy

nlp = spacy.load("./en_core_web_sm-3.1.0")
parser = benepar.Parser("./benepar_en3")

def Extract_Phrases(parse_tree,sentence):#抽取出短语
    subtexts = []
    Labels = ['NP','VP','ADVP','NML']
    for subtree in parse_tree.subtrees():
        subtext = ' '.join(subtree.leaves())
        sentence_length = len(sentence.split())
        subtext_length = len(subtext.split())
        if subtree.label() in Labels and (subtext_length > 3 and sentence_length - subtext_length >= 3):
            subtexts.append(subtext)
            break
    for i in reversed(range(len(subtexts)-1)):
        subtexts[i] = subtexts[i][0:subtexts[i].find(subtexts[i+1])]
        if subtexts[i] =='':
            del subtexts[i]
    return subtexts

def Phrases_Extraction(sentence):
    parse_tree = parser.parse(sentence)
    Phrases = Extract_Phrases(parse_tree,sentence)
    return Phrases


def Extract_Clauses(parse_tree,sentence):#抽取出从句
    subtexts = []
    Labels = ['S','SBAR','SQ','SINV']
    for subtree in parse_tree.subtrees():
        subtext = ' '.join(subtree.leaves())
        sentence_length = len(sentence.split())
        subtext_length = len(subtext.split())
        if subtree.label() in Labels and (subtext_length>3 and sentence_length - subtext_length >= 4):
            subtexts.append(subtext)
            break
    for i in reversed(range(len(subtexts)-1)):
        subtexts[i] = subtexts[i][0:subtexts[i].find(subtexts[i+1])]
        if subtexts[i] =='':
            del subtexts[i]
    return subtexts


def Clauses_Extraction(sentence):
    parse_tree = parser.parse(sentence)#得到解析树
    Clauses = Extract_Clauses(parse_tree,sentence)
    return Clauses

def generate_trees(root):#分解并列句
    """
    Yield all conjuncted variants of subtrees that can be generated from the given node.
    A subtree here is just a set of nodes.
    """
    prev_result = [root]
    if not root.children:
        yield prev_result
        return
    children_deps = {c.dep_ for c in root.children}
    if 'conj' in children_deps:
        # generate two options: subtree without cc+conj, or with conj child replacing the root
        # the first option:
        good_children = [c for c in root.children if c.dep_ not in {'cc', 'conj'}]
        for subtree in combine_children(prev_result, good_children):
            yield subtree 
        # the second option
        for child in root.children:
            if child.dep_ == 'conj':
                for subtree in generate_trees(child):
                    yield subtree
    else:
        # otherwise, just combine all the children subtrees
        for subtree in combine_children([root], root.children):
            yield subtree

def combine_children(prev_result, children):
    """ Combine the parent subtree with all variants of the children subtrees """
    child_lists = []
    for child in children:
        child_lists.append(list(generate_trees(child)))
    for prod in itertools.product(*child_lists):  # all possible combinations
        yield prev_result + [tok for parts in prod for tok in parts]

def Conjunctions_Extraction(sent):
    doc = nlp(sent)
    sentences = list(doc.sents)
    sub_sentences = []
    for sentence in sentences:
        for tree in generate_trees(sentence.root):
            subtext = ' '.join([token.text for token in sorted(tree, key=lambda x: x.i)])
            sub_sentences.append(subtext)
    return sub_sentences