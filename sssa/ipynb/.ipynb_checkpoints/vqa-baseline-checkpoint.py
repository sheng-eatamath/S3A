
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
    device = 'cuda:3'
    arch = 'ViT-B/16'
    dataset = 'imagenet'
    n_sampled_classes = 100
    input_size = 224
    estimate_k = 252
    
    batch_size = 256
    use_def = False
    clip_checkpoint = None
    # clip_checkpoint = '/home/sheng/MUST-output/make_nonliving26/baseline-04_22_1/checkpoint-current.pth'
    # clip_checkpoint = '/home/sheng/MUST-output/make_nonliving26/chatgpt_init-warmup=2/checkpoint-current.pth'
    # clip_checkpoint = '/home/sheng/MUST-output/make_entity13/sssa/checkpoint-current.pth'
    f_classifier = './cache/wordnet_classifier_in21k_word.pth'
    # f_classifier = './cache/wordnet_classifier_in21k_word_L.pth'
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

transform_val = build_transform(is_train=False, args=args, train_config=None)



from lavis.models import load_model_and_preprocess
from my_util_package.evaluate import measure_similarity_bert

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=args.device)
for dataset_name in ['sdogs', 'imagenet', 'make_nonliving26', 'make_living17', 'make_entity13', 'make_entity30', 'imagenet1k', 'imagenet21k_1']:
    print('='*30)
    print(dataset_name)
    args.dataset = dataset_name
    
    dataset_raw = get_datasets_oszsl(args, vocab, is_train=True, transform=None, seed=0)
    loader_r = torch.utils.data.DataLoader(dataset_raw, num_workers=4, batch_size=20, shuffle=False)

    with open(f'./cache/vlm-{args.dataset}-scd.pkl', 'rb') as f:
        res = pickle.load(f)
    topk_all_clu_pred = res['topk_all_clu_pred']
    pred_kmeans_t = res['pred_kmeans_t']
    all_clu_pred = res['all_clu_pred']

    to_name = lambda x: x.name().split('.')[0]
    cluster_row_synsets = []
    for row in topk_all_clu_pred:
        row_synsets = [vocab.mapping_idx_names[voc_idx.item()] for voc_idx in row]
        cluster_row_synsets.append(row_synsets)


    bsize = 2
    q = lambda c: f"which category does the object in the photo belong to? a. {c[0]}\nb. {c[1]}\nc. {c[2]}."
    q = lambda c: f"what is category name of the dog in the photo?  {c[0]}, {c[1]}, or {c[2]}?"
    with tqdm(total=len(dataset_raw)) as pbar:
        batch_label_voc, batch_label_clu = [], []
        batch_img, batch_q = [], []
        all_ans = []
        all_voc, all_clu = [], []
        for idx, item in enumerate(dataset_raw):
            image, label_voc, label_clu, idx_img = item[:4]
            image = vis_processors["eval"](image)
            question = txt_processors["eval"](q(cluster_row_synsets[pred_kmeans_t[idx_img]]))
            batch_label_voc.append(label_voc)
            batch_label_clu.append(label_clu)
            batch_img.append(image)
            batch_q.append(question)

            if idx%bsize==(bsize-1):
                ans = model.predict_answers(samples={"image": torch.stack(batch_img, dim=0).to(args.device), 
                                                     "text_input": batch_q}, inference_method="generate")
                all_ans.append(ans)
                all_voc.extend(batch_label_voc)
                all_clu.extend(batch_label_clu)

                batch_label_voc, batch_label_clu = [], []
                batch_img, batch_q = [], []

            pbar.update(1)

    c_gt = np.array([vocab.mapping_idx_names[x] for x in all_voc])
    c_pred = np.array(reduce(lambda x, y: x+y, all_ans))
    score = measure_similarity_bert(vocab, torch.tensor(all_voc), c_pred, device=args.device)
    print(f'score={score}')
    res = {
        'all_voc': all_voc,
        'c_gt': c_gt,
        'c_pred': c_pred,
    }
    with open(f'./cache/vqav2-{args.dataset}-train.pkl', 'wb') as f:
        pickle.dump(res, f)