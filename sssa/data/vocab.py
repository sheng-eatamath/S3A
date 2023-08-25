import os 
import pickle 
from collections import defaultdict
import torch
from config import *
from nltk.corpus import wordnet as wn
import numpy as np 

class Vocab:
    """ 
    [deprecated]
    classname indexed vocab 
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.indexing()
        ### {global_synset: global_names}
        self.mapping_ids_names = dict(zip(vocab['ids'], vocab['names']))
        # self.mapping_names_ids = dict(zip(vocab['names'], vocab['ids']))
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
        ### global index retained
        # self.classnames = list(self.vocab['names'])
        return
    
    
class Vocab2:
    """ classname indexed vocab """
    def __init__(self, vocab_classnames):
        self.vocab_classnames = vocab_classnames
        self.indexing()
        ### {global_synset: global_names}
        self.all_wnsynsets = [s for s in wn.all_synsets(pos='n')]
        offset_to_wnid = lambda offset: "n{:08d}".format(offset)
        self.mapping_ids_names = dict([
            (offset_to_wnid(s.offset()), s.name().split('.')[0]) 
            for s in self.all_wnsynsets
            ])
        ### {local_idx: local_names}
        self.mapping_idx_names = dict(zip(range(len(self.classnames)), self.classnames))
        ### {local_names: local_idx}
        self.mapping_names_idx = dict(zip(self.classnames, range(len(self.classnames))))
        ### {global_synset: global_idx}
        self.mapping_ids_global_idx = dict(zip(
            [offset_to_wnid(s.offset()) for s in self.all_wnsynsets], 
            range(len(self.all_wnsynsets))
            ))
        self.mapping_global_idx_ids = dict(zip(
            range(len(self.all_wnsynsets)),
            [offset_to_wnid(s.offset()) for s in self.all_wnsynsets]
            ))
        
        self.all_wnnames = [s.name().split('.')[0] for s in wn.all_synsets(pos='n')]
        ### {global_names: [global_idx]}
        self.vocab_mapping_names_idx = defaultdict(list)
        for k, v in zip(self.all_wnnames, range(len(self.all_wnnames))):
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
        self.classnames = sorted(list(set(self.vocab_classnames)))
        return
    
    
def read_vocab():
    """
    Args:
        vocab: {`names`: list, `ids`: synset ids, `parents`: [{synset ids}]}
    """
    with open(vocab_fpath, 'rb') as f:
        vocab = pickle.load(f)
        
    # with open('/home/sheng/dataset/wordnet_nouns_no_abstract.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    return vocab

def get_vocab_from_file():
    """
    Args:
        vocab: {`names`: list, `ids`: synset ids, `parents`: [{synset ids}]}
    """
    with open('/home/sheng/dataset/wordnet_nouns_with_synset_4.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def get_subsample_vocab(sample_synset_id: set):
    vocab = get_vocab_from_file()
    index = np.array([ i for i in range(len(vocab['synsets'])) if vocab['synsets'][i] in sample_synset_id ]).astype(np.int32)
    for k in vocab.keys():
        vocab[k] = np.array(vocab[k])[index].tolist()
    return vocab

def read_imagenet21k_classes():
    with open('/home/sheng/dataset/imagenet21k/imagenet21k_wordnet_ids.txt', 'r') as f:
        data = f.read()
        data = list(filter(lambda x: len(x), data.split('\n')))
    return data


def get_vocab(mode='wordnet'):
    """
    [deprecated]
    ARGS:
        mode: ['wordnet', 'in21k']
    """
    print(f'get_vocab {mode}')
    if mode == 'wordnet':
        vocab = read_vocab()
        vocab = Vocab(vocab=vocab)
    elif mode == 'in21k':
        classes = read_imagenet21k_classes() + os.listdir('/home/sheng/dataset/imagenet-img/')
        classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
        classes = set(classes)
        vocab = get_subsample_vocab(classes)
        vocab = Vocab(vocab=vocab)
    elif mode == 'in21k-L':
        classes = read_imagenet21k_classes() + os.listdir('/home/sheng/dataset/imagenet-img/')
        classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
        classes = set(classes)
        vocab = get_subsample_vocab(classes)
        vocab = Vocab(vocab=vocab)
    else:
        raise NotImplementedError()
    return vocab


def get_vocab_with_classnames(mode='in21k'):
    """
    ARGS:
        mode: ['in21k', 'concat3', 'concat3+lvis']
    """
    print(f'get_vocab {mode}')
    if mode == 'in21k':
        with open(f'{vocab_dir}/vocab_dataset=1.pkl', 'rb') as f:
            classnames = pickle.load(f)
        vocab = Vocab2(vocab_classnames=classnames)
    elif mode == 'concat3':
        raise ValueError()
        with open(f'{vocab_dir}/vocab_dataset=3_20088.pkl', 'rb') as f:
            classnames = pickle.load(f)
        vocab = Vocab2(vocab_classnames=classnames)
    elif mode == 'concat3+lvis':
        raise ValueError()
        with open(f'{vocab_dir}/vocab_dataset=3+lvis_20239.pkl', 'rb') as f:
            classnames = pickle.load(f)
        vocab = Vocab2(vocab_classnames=classnames)
    elif mode == 'concat2':
        with open(f'{vocab_dir}/vocab_dataset=2_20079.pkl', 'rb') as f:
            classnames = pickle.load(f)
        vocab = Vocab2(vocab_classnames=classnames)
    elif mode == 'concat2+lvis':
        with open(f'{vocab_dir}/vocab_dataset=2+lvis_20232.pkl', 'rb') as f:
            classnames = pickle.load(f)
        vocab = Vocab2(vocab_classnames=classnames)
    else:
        raise NotImplementedError()
    return vocab


def get_classifier_wordnet(device):
    """ classifier is normalized
    [deprecated]
    """
    classifier = torch.load('./data/wordnet_classifier.pth')
    classifier = classifier.to(device)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    return classifier


def get_classifier_concat2(device):
    """ classifier is normalized
    [deprecated]
    """
    classifier = torch.load(f'{HOME}/sssa/ipynb/cache/classifier_3d-concat2.pth')
    classifier = classifier.to(device)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    return classifier


def get_classifier(args):
    """ extracted from CLIP """
    print(f'get_classifier {args.vocab_name}')
    if args.vocab_name == 'wordnet':
        raise NotImplementedError()
        classifier = torch.load('/home/sheng/sssa/ipynb/cache/wordnet_classifier.pth')
    elif args.vocab_name == 'in21k':
        classifier = torch.load('/home/sheng/sssa/ipynb/cache/wordnet_classifier_in21k_word.pth')
    elif args.vocab_name == 'in21k-L':
        classifier = torch.load('/home/sheng/sssa/ipynb/cache/wordnet_classifier_in21k_word_L.pth')
    elif args.vocab_name == 'concat2':
        raise NotImplementedError()
        classifier = torch.load(f'{HOME}/sssa/ipynb/cache/classifier_3d-concat2.pth')
    else:
        raise NotImplementedError()
    classifier = classifier.to(args.device)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    return classifier