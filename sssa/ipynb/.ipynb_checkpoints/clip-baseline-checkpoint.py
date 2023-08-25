import sys
sys.path.append('/home/sheng/sssa/')
sys.path.append('/home/sheng/sssa/')
# sys.path.append('/home/sheng/sssa/CLIP/')

import os
import json
import re
import time
import pickle
from typing import Union, List
from pprint import pprint
from tqdm import tqdm
from copy import deepcopy
import random
import itertools
import numpy as np
from functools import reduce, partial
from itertools import zip_longest
import seaborn as sns
from collections import Counter, defaultdict, OrderedDict
import matplotlib.pyplot as plt
import heapq
from wordnet_utils import *
import scipy.io
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder

from ipynb_utils import get_hier_datasets, get_classifier, MCMF_assign_labels
# import clip
import model as clip
from data.datasets import build_transform, get_hier_datasets, Vocab
from data.imagenet_datasets import get_datasets_oszsl


class Config:
    device = 'cuda:1'
    arch = 'ViT-L/14'
    dataset = 'imagenet'
    n_sampled_classes = 100
    input_size = 224
    estimate_k = 252
    
    batch_size = 16
    use_def = False
    clip_checkpoint = None
    # f_classifier = './cache/wordnet_classifier_in21k_word.pth'
    f_classifier = './cache/wordnet_classifier_in21k_word-L.pth'
    templates_name = 'templates_small'
    seed = 0
    
args = Config()


def load_templates(args):
    with open(f'../{args.templates_name}.json', 'rb') as f:
        templates = json.load(f)['imagenet']
    return templates

def get_vocab():
    """
    Args:
        vocab: {`names`: list, `ids`: synset ids, `parents`: [{synset ids}]}
    """
    with open('/home/sheng/dataset/wordnet_nouns_with_synset_4.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def get_subsample_vocab(sample_synset_id: set):
    vocab = get_vocab()
    index = np.array([ i for i in range(len(vocab['synsets'])) if vocab['synsets'][i] in sample_synset_id ]).astype(np.int32)
    for k in vocab.keys():
        vocab[k] = np.array(vocab[k])[index].tolist()
    return vocab

def read_imagenet21k_classes():
    with open('/home/sheng/dataset/imagenet21k/imagenet21k_wordnet_ids.txt', 'r') as f:
        data = f.read()
        data = list(filter(lambda x: len(x), data.split('\n')))
    return data

""" from MUST """
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def build_classifier(args, model, templates, vocab_classnames, parent_classnames=None):
    batch_size = 64
    with torch.no_grad():
        zeroshot_weights = []
        assert parent_classnames is None
        with tqdm(total=len(vocab_classnames)//batch_size) as pbar:
            for classname_set in np.array_split(vocab_classnames, len(vocab_classnames)//batch_size):
                texts = [template.format(classname) for classname in classname_set for template in templates] #format with class
                texts = tokenize(texts).to(args.device) #tokenize
                class_embeddings = model.encode_text(texts).float() #embed with text encoder
                class_embeddings = class_embeddings.view(-1, len(templates), class_embeddings.size(-1))
                class_embeddings = F.normalize(class_embeddings, dim=-1)
                class_embedding = class_embeddings.mean(dim=1)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                zeroshot_weights.append(class_embedding.cpu())
                pbar.update(1)
    classifier = torch.cat(zeroshot_weights, dim=0)
    return classifier


def load_clip(args):
    model, preprocess = clip.load(args.arch)
    if args.clip_checkpoint:
        model.load_state_dict({k[len('model.'):]:v for k, v in torch.load(args.clip_checkpoint, map_location='cpu')['model_ema'].items()}, strict=False)
    model.to(args.device).eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model, preprocess

def load_clip2(args):
    model = clip.load(args.arch, device=args.device)
    if args.clip_checkpoint:
        model.load_state_dict({k[len('model.'):]:v for k, v in torch.load(args.clip_checkpoint, map_location='cpu')['model_ema'].items()}, strict=False)
    model.to(args.device).eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model

def load_mixture_clip(args, decay=1.0):
    model1 = clip.load(args.arch)
    if args.clip_checkpoint:
        model1.load_state_dict({k[len('model.'):]:v for k, v in torch.load(args.clip_checkpoint, map_location='cpu')['model_ema'].items()}, strict=False)
    model1.to(args.device).eval()
    model2 = clip.load(args.arch)
    model2.to(args.device).eval()
    with torch.no_grad():
        msd = model1.state_dict()
        for k, ema_v in model2.state_dict().items():
            # if needs_module:
            #     k = 'module.' + k
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)
    return model2

def topk_acc(all_pred_voc_topk, all_gt_voc):
    acc = []
    ### topK accuracy
    for i in range(all_pred_voc_topk.size(1)):
        vec = torch.zeros(all_pred_voc_topk.size(0)).bool()
        for j in range(i+1):
            vec |= (all_pred_voc_topk[:, j]==all_gt_voc)
        print(f'k={i} acc={vec.float().mean()}')
        acc.append(vec.float().mean().item())
    return acc

def semantic_acc(y_pred, y_true, metrics={}):
    """ compute soft semantic acc for @y_pred and @y_true """
    assert len(metrics)>0
    assert y_pred.size(0)==y_true.size(0)
    scores = {m:[] for m in metrics.keys()}
    with tqdm(total=y_pred.size(0)) as pbar:
        for i in range(y_pred.size(0)):
            syn_pred = mapping_vocidx_to_synsets(y_pred[i].item(), vocab)
            syn_true = mapping_vocidx_to_synsets(y_true[i].item(), vocab)
            pairs = list(itertools.product(range(len(syn_pred)), range(len(syn_true))))
            for m_name, m in metrics.items():
                scores[m_name].append( max([ m(syn_pred[p[0]], syn_true[p[1]]) for p in pairs ]) )
            pbar.update(1)
    for m_name in metrics.keys():
        scores[m_name] = np.array(scores[m_name]).mean()
    return scores
    
""" from MUST """
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


templates = load_templates(args)
vocab = get_vocab()
nouns = [ wn.synset(s) for s in vocab['synsets'] ]
classnames = vocab['names']
parents = vocab['parents']
defs = vocab['def']


""" prepare dataset and load CLIP """
classes = read_imagenet21k_classes() + os.listdir('/home/sheng/dataset/imagenet-img/')
classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
classes = set(classes)
vocab = get_subsample_vocab(classes)
vocab = Vocab(vocab=vocab)


for dataset_name in ['make_entity13']:
    print('='*30)
    print(dataset_name)
    args.dataset = dataset_name
    transform_val = build_transform(is_train=False, args=args, train_config=None)
    model = load_clip2(args)

    dataset = get_datasets_oszsl(args, None, is_train=True, transform=transform_val, seed=0)
    idx_to_class = dict(zip(dataset.labels_transformed, dataset.labels))
    classifier = build_classifier(args, model, templates, [idx_to_class[i] for i in range(dataset.num_classes)]).to(args.device)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    loader_val = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=args.batch_size, shuffle=False)

    amp_autocast = torch.cuda.amp.autocast
    all_pred_voc = []
    all_gt_voc = []
    all_gt_clu = []
    all_pred_voc_topk = []
    all_vfeatures = []
    with tqdm(total=len(loader_val)) as pbar:
        model.eval()
        for idx_batch, batch in enumerate(loader_val):
            images, label_voc, label_clu, idx_img = batch
            images = images.to(args.device)
            with amp_autocast():
                with torch.no_grad():
                    logits = model.visual.extract_features(images)
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = model.logit_scale.exp() * logits @ classifier.t()
                    prob = similarity.softmax(-1)
                    all_pred_voc.append(deepcopy(prob.argmax(dim=-1).cpu()))
                    all_gt_voc.append(deepcopy(label_voc))
                    all_pred_voc_topk.append(deepcopy(prob.topk(k=5, dim=-1).indices.cpu()))
                    all_vfeatures.append(deepcopy(logits.cpu().numpy()))
                    all_gt_clu.append(deepcopy(label_clu))
            pbar.update(1)

    all_pred_voc = torch.cat(all_pred_voc, dim=0)
    all_gt_voc = torch.cat(all_gt_voc, dim=0)
    all_pred_voc_topk = torch.cat(all_pred_voc_topk, dim=0)
    all_vfeatures = np.concatenate(all_vfeatures)
    all_gt_clu = torch.cat(all_gt_clu, dim=0)

    print(f'acc={(all_pred_voc == all_gt_voc).float().mean()}')
    print(f'acc={(all_gt_clu == all_gt_voc).float().mean()}')
