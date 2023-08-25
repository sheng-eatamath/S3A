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

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from ipynb_utils import get_hier_datasets, get_classifier, MCMF_assign_labels
import model as clip
from data.datasets import build_transform, get_hier_datasets, Vocab
from data.imagenet_datasets import get_datasets_oszsl
from sklearn.cluster import KMeans, MiniBatchKMeans
from my_util_package_oszsl.evaluation import cluster_acc


class Config:
    device = 'cuda:0'
    arch = 'ViT-B/16'
    dataset = 'make_nonliving26'
    n_sampled_classes = 100
    input_size = 224
    seed = 0
    
    batch_size = 512
    use_def = False
    clip_checkpoint = None
    f_classifier = './cache/wordnet_classifier_in21k_word.pth'
    templates_name = 'templates_small'
    
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

def read_lvis_imagenet21k_classes():
    with open('/home/sheng/dataset/imagenet21k/lvis_imagenet21k_wordnet_ids.txt', 'r') as f:
        data = f.read()
        data = list(filter(lambda x: len(x), data.split('\n')))
        # data = list(map(lambda x: x.split('.')[0], data))
    return data

def load_clip2(args):
    model = clip.load(args.arch, device=args.device)
    if args.clip_checkpoint:
        model.load_state_dict({k[len('model.'):]:v for k, v in torch.load(args.clip_checkpoint, map_location='cpu')['model'].items()}, strict=False)
    model.to(args.device).eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    return model

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
if args.dataset == 'lvis':
    classes = read_lvis_imagenet21k_classes()
    classes = set(classes)
vocab = get_subsample_vocab(classes)
vocab = Vocab(vocab=vocab)

transform_val = build_transform(is_train=False, args=args, train_config=None)
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

""" load dataset """
transform_f = transforms.Compose([
    transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=torch.tensor(mean),
        std=torch.tensor(std))
])


from my_util_package.dino import vision_transformer as vits
from config import dino_pretrain_path
from sklearn.cluster import KMeans, MiniBatchKMeans
from my_util_package_oszsl.evaluation import cluster_acc

dino = vits.vit_base()
dino.load_state_dict(torch.load(dino_pretrain_path, map_location='cpu'))
dino = dino.to(args.device)

for dataset_name in ['imagenet', 'make_living17', 'make_nonliving26', 'make_entity13', 'make_entity30', 'imagenet1k', 'sdogs', 'imagenet21k_1']:
    print('='*30)
    print(dataset_name)
    args.dataset = dataset_name
    dataset = get_datasets_oszsl(args, vocab, is_train=True, transform=transform_f, seed=0)
    loader_val = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=256, shuffle=False)
    print('dataset size', len(dataset))

    all_vfeatures = []
    all_clu_label = []
    with tqdm(total=len(loader_val)) as pbar:
        dino.eval()
        for idx_batch, batch in enumerate(loader_val):
            images, label_voc, label_clu, idx_img = batch
            images = images.to(args.device)
            with torch.no_grad():
                features = dino(images)
                features = F.normalize(features, dim=-1)
                all_vfeatures.append(deepcopy(features.cpu().numpy()))
                all_clu_label.append(deepcopy(label_clu.numpy()))
            pbar.update(1)

    all_vfeatures = np.concatenate(all_vfeatures)
    all_clu_label = np.concatenate(all_clu_label)

    kmeans = KMeans(n_clusters=dataset.num_classes, random_state=0, n_init=10, max_iter=1000, verbose=1).fit(all_vfeatures)
    preds = kmeans.labels_
    acc = cluster_acc(all_clu_label, preds)

    with open(f'./cache/dino/dino-{args.dataset}.pkl', 'wb') as f:
        pickle.dump({
            'all_vfeatures': all_vfeatures,
            'all_clu_label': all_clu_label,
            'acc': acc,
        }, f)