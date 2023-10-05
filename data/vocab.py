import os 
import pickle 
from collections import defaultdict
import torch
from config import *
from nltk.corpus import wordnet as wn
import numpy as np 

class Vocab:
    """ classname indexed vocab """
    def __init__(self, vocab):
        self.vocab = vocab
        self.indexing()
        ### {global_synset: global_names}
        self.mapping_ids_names = dict(zip(vocab['ids'], vocab['names']))
        ### {local_idx: local_names}
        self.mapping_idx_names = dict(zip(range(len(self.classnames)), self.classnames))
        ### {local_names: local_idx}
        self.mapping_names_idx = dict(zip(self.classnames, range(len(self.classnames))))
        ### {global_synset: global_idx}
        self.mapping_ids_global_idx = dict(zip(vocab['ids'], range(len(vocab['ids']))))
        self.mapping_global_idx_ids = dict(zip(range(len(vocab['ids'])), vocab['ids']))
        
        ### {global_names: [global_idx]}
        self.vocab_mapping_names_idx = defaultdict(list)
        for k, v in zip(vocab['names'], range(len(vocab['names']))):
            self.vocab_mapping_names_idx[k].append(v)
            
        ### {local_idx: global_idx}
        self.mapping_idx_global_idx = {}
        for i in range(len(self)):
            self.mapping_idx_global_idx[i] = self.vocab_mapping_names_idx[self.mapping_idx_names[i]]

        return
    
    def __len__(self):
        return len(self.classnames)
    
    def __getitem__(self, idx):
        return self.classnames[idx]
    
    def indexing(self):
        self.classnames = sorted(list(set(self.vocab['names'])))
        return
    

def get_vocab_from_file():
    """
    Args:
        vocab: {`names`: list, `ids`: synset ids, `parents`: [{synset ids}]}
    """
    with open(f'{PROJECT_HOME}/wordnet_nouns.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def get_subsample_vocab(sample_synset_id: set):
    vocab = get_vocab_from_file()
    index = np.array([ i for i in range(len(vocab['synsets'])) if vocab['synsets'][i] in sample_synset_id ]).astype(np.int32)
    for k in vocab.keys():
        vocab[k] = np.array(vocab[k])[index].tolist()
    return vocab

def read_imagenet21k_classes():
    with open(f'{PROJECT_HOME}/imagenet21k_wordnet_ids.txt', 'r') as f:
        data = f.read()
        data = list(filter(lambda x: len(x), data.split('\n')))
    return data


def get_vocab(mode='in21k'):
    """
    ARGS:
        mode: ['wordnet', 'in21k']
    """
    print(f'get_vocab {mode}')
    if mode == 'in21k':
        classes = read_imagenet21k_classes() + os.listdir(f'{HOME}/dataset/imagenet-img/')
        classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
        classes = set(classes)
        vocab = get_subsample_vocab(classes)
        vocab = Vocab(vocab=vocab)
    elif mode == 'in21k-L':
        classes = read_imagenet21k_classes() + os.listdir(f'{HOME}/dataset/imagenet-img/')
        classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
        classes = set(classes)
        vocab = get_subsample_vocab(classes)
        vocab = Vocab(vocab=vocab)
    else:
        raise NotImplementedError()
    return vocab


def get_classifier(args):
    """ extracted from CLIP """
    print(f'get_classifier {args.vocab_name}')
    if args.vocab_name == 'in21k':
        classifier = torch.load(f'{PROJECT_HOME}/ipynb/cache/vocabulary_classifier.pth')
    elif args.vocab_name == 'in21k-L':
        classifier = torch.load(f'{PROJECT_HOME}/ipynb/cache/vocabulary_classifier_L.pth')
    else:
        raise NotImplementedError()
    classifier = classifier.to(args.device)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    return classifier