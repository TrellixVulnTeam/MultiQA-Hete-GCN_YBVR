import itertools
import spacy
nlp = spacy.load("./en_core_web_sm-3.1.0")

def generate_trees(root):
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

def conjunctions(sent):
    doc = nlp(sent)
    sentence = list(doc.sents)[0]
    sentences = []
    for tree in generate_trees(sentence.root):
        sentences.append(' '.join([token.text for token in sorted(tree, key=lambda x: x.i)]))
    return sentences