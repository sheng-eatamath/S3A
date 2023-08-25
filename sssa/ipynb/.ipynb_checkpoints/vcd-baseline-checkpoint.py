import sys
sys.path.append('/home/sheng/MUST')
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
    device = 'cuda:0'
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
    all_tokens = [[sot_token] + _tokenizer.encode(text)[:75] + [eot_token] for text in texts]
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


def agg_by_pred_cluster(args, pred_kmeans, all_topk_voc, voc_size):
    """
    Args:
        pred_kmeans: np.array([N])
        all_topk_voc: np.array([N x K])
        voc_size: int
    Returns:
        all_clu_pred: tensor([C x V])
    """
    print('agg_by_pred_cluster')
    all_clu_pred = []
    n_count = []
    for i in np.unique(pred_kmeans):
        selected = (pred_kmeans==i)
        n_count.append( selected.sum().item() )
        counter_voc_ind, counter_val = np.unique((all_topk_voc[selected, :]).ravel(), return_counts=True)
        # counter_val = counter_val/(n_count+1e-20) # L1 norm
        clu_pred = torch.zeros(args.num_voc) # cluster-wise prob
        clu_pred[torch.from_numpy(counter_voc_ind).long()] = torch.from_numpy(counter_val).float()
        # clu_pred = F.normalize(all_topk_voc[selected].sum(dim=0), dim=-1, p=1)
        all_clu_pred.append(clu_pred)
    all_clu_pred = torch.stack(all_clu_pred, dim=0).cpu()
    n_count = torch.tensor(n_count).cpu()
    
    # all_clu_pred = setdiff_assignment(all_clu_pred)
    
    all_clu_pred = all_clu_pred/(n_count.view(-1, 1) + 1e-20)
    
    print('is mutex assignment::', all_clu_pred.argmax(dim=-1).size(0)==all_clu_pred.argmax(dim=-1).unique().size(0))
    print('assignment collision num::', len(list(filter(lambda x: x>1, Counter(all_clu_pred.argmax(dim=-1).numpy()).values()))))
    return all_clu_pred

def linear_assign(all_clu_pred, pred_kmeans, all_gt_voc, return_results=False):
    print('linear_assign')
    cost_mat = all_clu_pred.cpu().numpy()
    print(f'assignment shape={cost_mat.shape}')
    res_ass = linear_assignment(cost_mat.max() - cost_mat)
    label_voc_kmeans = torch.tensor([res_ass[1][x.item()] for x in pred_kmeans])
    inst_acc = (label_voc_kmeans==all_gt_voc).float().mean().item()
    print('instance label acc::', inst_acc)
    if return_results:
        return label_voc_kmeans, res_ass, inst_acc
    return label_voc_kmeans, res_ass

def reassign_by_pred_cluster(label_voc_kmeans, model, classifier, device, 
                             all_prob=None, 
                             instance_selected=None, 
                             classifier_selected=None):
    """
    Args:
        classifier_selected: tensor([C2])
    """
    print('reassign_by_pred_cluster')
    amp_autocast = torch.cuda.amp.autocast
    label_voc_kmeans = label_voc_kmeans.to(device)
    if all_prob is None:
        cluster_ind = []
        with tqdm(total=len(loader_f)) as pbar:
            if hasattr(model, 'eval'):
                model.eval()
            for idx_batch, batch in enumerate(loader_f):
                images, label_voc, label_clu, idx_img = batch[:4]
                images = images.to(device)
                if (instance_selected is not None) and ((~instance_selected[idx_img]).all()):
                    continue
                with amp_autocast():
                    with torch.no_grad():
                        if (instance_selected is not None):
                            logits = model.visual(images[instance_selected[idx_img]])
                        else:
                            logits = model.visual(images)
                            
                        logits = logits/logits.norm(dim=-1, keepdim=True)
                        if classifier_selected is not None:
                            similarity = 100 * logits @ classifier[classifier_selected].t()
                            prob = classifier_selected[similarity.softmax(-1)]
                            cluster_ind.append(deepcopy(prob.cpu().argmax(dim=-1)))
                        else:
                            similarity = 100 * logits @ classifier.t()
                            prob = similarity.softmax(-1)
                            cluster_ind.append(deepcopy(prob[:, label_voc_kmeans].cpu().argmax(dim=-1)))
                pbar.update(1)
        cluster_ind = torch.cat(cluster_ind, dim=0)
    else:
        all_prob = all_prob[:, label_voc_kmeans]
        cluster_ind = all_prob.argmax(dim=-1)
        
    if classifier_selected is not None:
        cluster_ind_voc = classifier_selected[cluster_ind]
    else:
        cluster_ind_voc = label_voc_kmeans[cluster_ind]
    mapping_ind = dict(zip(cluster_ind.unique().numpy(), torch.arange(cluster_ind.unique().size(0)).numpy()))
    cluster_ind = torch.tensor([mapping_ind[x.item()] for x in cluster_ind])
    return cluster_ind, cluster_ind_voc


def reassign_by_pred_cluster(label_voc_kmeans, loader_f, model, classifier, device, 
                             preextracted_vfeatures=None):
    """ given vocab label set @label_voc_kmeans, 
    Args:
        label_voc_kmeans: cluster-assigned label on vocab
        ...
        preextracted_vfeatures: np.array([N x D])
    Returns:
        cluster_ind: tensor([N]): re-ordered cluster assignment
        cluster_ind_voc: tensor([N]): cluster assignment indiced by vocab
    """
    print('reassign_by_pred_cluster')
    amp_autocast = torch.cuda.amp.autocast
    label_voc_kmeans = label_voc_kmeans.to(device).unique()
    cluster_ind = []
    with tqdm(total=len(loader_f)) as pbar:
        if hasattr(model, 'eval'):
            model.eval()
        if preextracted_vfeatures is not None:
            N = len(loader_f.dataset)
            batch_size = 10000
            indices = np.array_split(np.arange(N), N//batch_size)
            with torch.no_grad():
                for group in indices:
                    logits = torch.from_numpy(preextracted_vfeatures[group]).float()
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = 100 * logits@classifier.t().cpu()
                    prob = similarity.softmax(-1)
                    cluster_ind.append(deepcopy(prob[:, label_voc_kmeans.cpu()].argmax(dim=-1)))
        else:
            for idx_batch, batch in enumerate(loader_f):
                images, label_voc, label_clu, idx_img = batch[:4]
                images = images.to(device)
                with amp_autocast():
                    with torch.no_grad():
                        if preextracted_vfeatures is not None:
                            logits = torch.from_numpy(preextracted_vfeatures[idx_img.cpu().numpy()]).float().to(device)
                        else:
                            logits = model.ema.extract_vfeatures(images)
                        logits = logits/logits.norm(dim=-1, keepdim=True)
                        similarity = 100 * logits @ classifier.t()
                        prob = similarity.softmax(-1)
                        cluster_ind.append(deepcopy(prob[:, label_voc_kmeans].cpu().argmax(dim=-1)))
                pbar.update(1)
    cluster_ind = torch.cat(cluster_ind, dim=0)
    cluster_ind_voc = label_voc_kmeans[cluster_ind]
    mapping_ind = dict(zip(cluster_ind.unique().numpy(), torch.arange(cluster_ind.unique().size(0)).numpy()))
    cluster_ind = torch.tensor([mapping_ind[x.item()] for x in cluster_ind])
    return cluster_ind, cluster_ind_voc


@torch.no_grad()
def computation_reassign_by_pred_cluster(row, idx, args, model, classifier, candidate_classifier_ind):
    """
    candidate_classifier_ind = label_voc_kmeans.unique().to(args.device)
    """
    images, label_voc, label_clu, idx_img = row[:4]
    images = images.to(args.device)
    with amp_autocast():
        vfeatures = model.visual(images).float()
        # vfeatures = vfeatures/vfeatures.norm(dim=-1, keepdim=True)
    vfeatures = F.normalize(vfeatures, dim=-1)
    batch_sim = 100*vfeatures@classifier[candidate_classifier_ind].t()
    cluster_ind = batch_sim.argmax(dim=-1)
    cluster_ind_voc = candidate_classifier_ind[cluster_ind].cpu()
    return cluster_ind_voc

def aggregation_reassign_by_pred_cluster(r, candidate_classifier_ind):
    cluster_ind_voc = torch.cat(r, dim=0)
    mapping_ind = dict(zip(cluster_ind_voc.unique().numpy(), torch.arange(cluster_ind_voc.unique().size(0)).numpy()))
    cluster_ind = torch.tensor([mapping_ind[x.item()] for x in cluster_ind_voc])
    return cluster_ind, cluster_ind_voc


@torch.no_grad()
def extract_vfeatures(model, data_loader, device):
    amp_autocast = torch.cuda.amp.autocast
    all_vfeatures = []
    with tqdm(total=len(data_loader)) as pbar:
        if hasattr(model, 'eval'):
            model.eval()
        for idx_batch, batch in enumerate(data_loader):
            images, label_voc, label_clu, idx_img = batch[:4]
            images = images.to(device)
            with amp_autocast():
                vfeatures = model.visual(images).float()
            vfeatures = vfeatures/vfeatures.norm(dim=-1, keepdim=True)
            all_vfeatures.append(vfeatures.cpu().numpy())
            pbar.update(1)
    all_vfeatures = np.concatenate(all_vfeatures)
    return all_vfeatures


@torch.no_grad()
def loop_row_collect_results_nograd(obj_iter, computations={}, aggregations={}):
    """ compute and aggregate results, looping over @obj_iter 
    func_computation(@row, @index_row)
    aggregations(list(@results_computation))
    """
    assert set(list(computations.keys())) == set(list(aggregations.keys()))
    collector = { k:[] for k in computations }
    with tqdm(total=len(obj_iter)) as pbar:
        for i, row in enumerate(obj_iter):
            ### apply computations
            for k, func in computations.items():
                collector[k].append(func(row, i))
            pbar.update(1)
    ### aggregate results
    results = {}
    for k, func_agg in aggregations.items():
        results[k] = func_agg(collector[k])
    return results


import openai
def openai_chatgpt_post(content, parameters={'temperature': 0.7}):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0301",
      messages=[
        {"role": "user", "content": content},
      ],
    **parameters,
    )
    result = completion['choices'][0]['message']['content']
    return result


@torch.no_grad()
def build_classifier_chatgpt(all_row_chatgpt_names, model, all_row_key_name=None):
    """ build classifier for chatgpt
    Args:
        all_row_chatgpt_names: [[names]]
    """
    if all_row_key_name is None: ### single name
        with open('../templates_small.json', 'rb') as f: ### template 1
            templates = json.load(f)['imagenet']
    else:
        with open('../templates_small.json', 'rb') as f: ### template 2
            templates = json.load(f)[f'{args.dataset}-parent-3']
            
    len_t = len(templates)
    row_classifier = []
    with tqdm(total=len(all_row_chatgpt_names)) as pbar:
        for idx, row in enumerate(all_row_chatgpt_names):
            len_row = len(row)
            if all_row_key_name is None:
                row_t = [ t.format(name) for name in row for t in templates ]
            else:
                row_t = [ t.format(pname, name) for pname, name in zip(all_row_key_name[idx], row) for t in templates ]
            row_t = tokenize(row_t).to(args.device)
            features = model.encode_text(row_t)
            features = features.view(len_row, len_t, -1).float()
            features = features/features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=1)
            features = features/features.norm(dim=-1, keepdim=True)
            row_classifier.append(features.cpu())
            
            pbar.update(1)
    return row_classifier
    



from sklearn.cluster import KMeans
from my_util_package_oszsl.evaluation import cluster_acc
from scipy.optimize import linear_sum_assignment as linear_assignment

subset = ['train', 'val'][0]
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

""" prepare dataset and load CLIP """
classes = read_imagenet21k_classes() + os.listdir('/home/sheng/dataset/imagenet-img/')
classes = [wn.synset_from_pos_and_offset('n', int(x[1:])).name() for x in classes]
classes = set(classes)
vocab = get_subsample_vocab(classes)
vocab = Vocab(vocab=vocab)

model = load_clip2(args)


for dataset_name in ['sdogs', 'imagenet', 'make_nonliving26', 'make_living17', 'make_entity30', 'imagenet1k', 'imagenet21k_1']:
    print('='*30)
    print(f'dataset={dataset_name}')
    args.dataset = dataset_name
    # dataset_f = get_datasets_oszsl(args, vocab, is_train=False, transform=transform_f)
    if subset == 'train':
        dataset_f = get_datasets_oszsl(args, vocab, is_train=True, transform=transform_f, seed=0)
    elif subset == 'val':
        dataset_f = get_datasets_oszsl(args, vocab, is_train=False, transform=transform_f, seed=0)
    args.nb_classes = dataset_f.num_classes
    loader_f = torch.utils.data.DataLoader(dataset_f, num_workers=4, batch_size=args.batch_size, shuffle=False)

    loader_f = torch.utils.data.DataLoader(dataset_f, num_workers=4, batch_size=args.batch_size, shuffle=False)
    classifier = get_classifier(args)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    args.num_voc = classifier.size(0)
    amp_autocast = torch.cuda.amp.autocast
    ### collect variables
    prob_k = 1
    all_topk_voc = []
    all_gt_voc = []
    all_label_clu = []
    all_vfeatures = []
    with tqdm(total=len(loader_f)) as pbar:
        if hasattr(model, 'eval'):
            model.eval()
        for idx_batch, batch in enumerate(loader_f):
            images, label_voc, label_clu, idx_img = batch[:4]
            images = images.to(args.device)
            with amp_autocast():
                with torch.no_grad():
                    logits = model.visual.extract_features(images)
                    # logits = model.extract_vfeatures(images)
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = 100 * logits @ classifier.t()
                    prob = similarity.softmax(-1)
                    prob_topk_ind = prob.topk(k=prob_k, dim=-1).indices
                    all_topk_voc.append(deepcopy(prob_topk_ind.cpu().numpy()))
                    all_gt_voc.append(deepcopy(label_voc))
                    all_label_clu.append(deepcopy(label_clu))
                    all_vfeatures.append(deepcopy(logits.cpu().numpy()))
            pbar.update(1)

    all_topk_voc = np.concatenate(all_topk_voc)
    all_gt_voc = torch.cat(all_gt_voc, dim=0)
    all_label_clu = torch.cat(all_label_clu, dim=0)
    all_vfeatures = np.concatenate(all_vfeatures)


    # pred_kmeans = torch.from_numpy(np.load(f'./pred_clu-{args.dataset}-train-clip.npy'))
    pred_kmeans = torch.from_numpy(np.load(f'./cache/cluster/kmeans-{args.dataset}.npy'))
    # pred_kmeans = torch.from_numpy(np.load('/home/sheng/MUST-output/make_nonliving26/baseline-04_22_1/pred_kmeans_t.npy'))
    # pred_kmeans = torch.from_numpy(np.load('/home/sheng/MUST-output/make_nonliving26/chatgpt_init-warmup=2/pred_kmeans_t.npy'))
    pred_kmeans_t = pred_kmeans
    history_set_pred = []
    for t in range(3):
        record_pred_kmeans_t = pred_kmeans_t
        all_clu_pred = agg_by_pred_cluster(args, pred_kmeans_t.numpy(), all_topk_voc, voc_size=args.num_voc)
        label_voc_kmeans, res_ass = linear_assign(all_clu_pred, pred_kmeans_t, all_gt_voc)
        pred_kmeans_t, cluster_ind_voc = reassign_by_pred_cluster(label_voc_kmeans, loader_f, model, classifier, args.device, preextracted_vfeatures=all_vfeatures)
        set_pred = set(res_ass[1].tolist())
        set_gt = set(all_gt_voc.unique().numpy().tolist())
        print('missing label::', len(set_gt - set_pred))
        print('cluster acc', cluster_acc(y_true=all_label_clu.numpy(), y_pred=pred_kmeans_t.numpy()))
        history_set_pred.append(set_pred)

        set_pred = set(res_ass[1].tolist())
        set_gt = set(all_gt_voc.unique().numpy().tolist())
        n_inter = all_gt_voc[cluster_ind_voc.cpu()==all_gt_voc].unique().shape[0]
        n_union = torch.cat([cluster_ind_voc.cpu(), all_gt_voc]).unique().shape[0]
        iou_voc = n_inter/n_union
        n_missing_label = all_gt_voc.unique().shape[0] - n_inter
        print('missing label::', n_missing_label)
        print('iou voc::', iou_voc)


    all_clu_gt_voc = []
    for c in record_pred_kmeans_t.unique():
        select = (record_pred_kmeans_t==c)
        all_clu_gt_voc.append(all_gt_voc[select].mode().values)
    all_clu_gt_voc = torch.tensor(all_clu_gt_voc)
    k_1 = 3
    topk_all_clu_pred = all_clu_pred.topk(k=k_1).indices
    cluster_is_correct = torch.zeros(topk_all_clu_pred.size(0)).bool()
    for i in range(k_1):
        cluster_is_correct |= (topk_all_clu_pred[:, i]==all_clu_gt_voc)
    print(f'recall@{k_1} = {cluster_is_correct.float().mean()}')
    """ gather concepts """
    to_name = lambda x: [ s.name() + ': ' + s.definition() for s in x ]
    cluster_row_synsets = []
    for row in topk_all_clu_pred:
        row_synsets = [to_name(mapping_vocidx_to_synsets(voc_idx.item(), vocab)) for voc_idx in row]
        cluster_row_synsets.append(row_synsets)

    """ generate concept requests """
    concept_request = []
    for row in cluster_row_synsets:
        ccpts = reduce(lambda x, y: x+y, row)
        ccpts = list(map(lambda x: "'"+x+".'", ccpts))
        ccpts = ', '.join(ccpts)
        concept_request.append(ccpts)

    """ generate concept templates """
    template_1 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all alternative concept names for each visual concept. List in the format \"{concept name}: {list of names separated by ';'}.\""
    with open('/home/sheng/OSZSL/templates_chatgpt.json', 'r') as f:
        template_chatgpt = json.load(f)
    template_2 = lambda concept_list: template_chatgpt['pictionary-long'].format(concept_list)
    template_3 = lambda concept_list: template_chatgpt['pictionary-short'].format(concept_list)
    template_4 = lambda concept_list: template_chatgpt['direct'].format(concept_list)
    template_5 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all synonym concept names for each visual concept. List in the format \"{concept name}: {list of names separated by ';'}.\""
    template_6 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all category names for each visual concept. List in the format \"{concept name}: {list of names separated by ';'}.\""
    template_7 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all parent-type category names for each visual concept. List in the format \"{concept name}: {list of names separated by ';'}.\""
    template_8 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all possible descriptive phrases of image captions for each visual concept. List in the format \"{concept name}: {all phrases deliminated by semicolons}.\" for each concept. No duplication."
    template_9 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all possible visiual descriptive phrases for each visual concept without duplication. List in the format \"{concept name}: {all phrases deliminated by semicolons}.\" for each concept. No duplication."
    # template_10 = lambda concept_list: "Given visual concepts: "+ concept_list + "List all possible visiual descriptive phrases for each visual concept without duplication. List in the format \"{concept name}: {all phrases deliminated by semicolons}.\" for each concept. No duplication."
    template_13 = lambda concept_list: "Given visual concepts: "+ concept_list + "Please list all possible adjective phrases of visual descriptions for each visual concept without duplication. Please list in the format \"{concept name}: {all phrases deliminated by semicolons}.\" for each concept. No duplication."
    template_9_1 = lambda concept_list: "Given visual concepts: "+ concept_list + "Please list all possible visual descriptive phrases for each visual concept without duplication. Please list in the format \"{concept name}: {all phrases deliminated by semicolons}.\" for each concept. No duplication."


    template_in_use = template_9_1
    concept_templates = []
    for row in concept_request:
        concept_templates.append(template_in_use(row))

    n_repeat = 3


    request = lambda c: f"what are useful visual features for distinguishing a {c} in a photo? please list descriptive phrases in bullet points."

    try:
        with open('./cache/openai/vcd-total.pkl', 'rb') as f:
            all_cate_descriptions = pickle.load(f)
    except Exception as e:
        print(e)
        all_cate_descriptions = {}

    with tqdm(total=len(cluster_row_synsets)*n_repeat) as pbar:
        for row in cluster_row_synsets:
            for cate in row:
                cate_name = cate[0].split('.')[0]
                if cate_name in all_cate_descriptions.keys():
                    continue
                else:
                    all_cate_descriptions.setdefault(cate_name, [])
                    for i in range(n_repeat):
                        while 1:
                            try:
                                response = openai_chatgpt_post(request(cate_name))
                            except Exception as e:
                                print(e)
                            break
                        all_cate_descriptions[cate_name].append(response)

                        pbar.update(1)

    all_cate_descriptions_new = {}
    for k, v in all_cate_descriptions.items():
        all_cate_descriptions_new[k] = \
            list(
                map(lambda x: 
                    list(map(lambda y: y.strip(' ').strip('\n'), x.split('- ')))[1:], 
                    v)
            )

    for k, v in all_cate_descriptions_new.items():
        all_cate_descriptions_new[k] = reduce(lambda x, y: x+y, all_cate_descriptions_new[k])

    with open(f'./cache/openai/vcd-total.pkl', 'wb') as f:
        pickle.dump(all_cate_descriptions, f)

    with open(f'./cache/openai/vcd-clean-total.pkl', 'wb') as f:
        pickle.dump(all_cate_descriptions_new, f)


    all_row_chatgpt_names = []
    all_row_key_name = []
    all_row_weight = []
    all_row_i_syn = []
    for row in cluster_row_synsets:
        row_chatgpt_names = []
        row_key_name = []
        row_i_syn = []
        for i, cate in enumerate(row):
            cate_name = cate[0].split('.')[0]
            desc = all_cate_descriptions_new[cate_name]
            if len(desc)==0:
                print(f'desc=0 for {cate_name}')
                desc = [cate[0].split(':')[-1]]
            row_key_name.extend([cate_name]*len(desc))
            row_chatgpt_names.extend(desc)
            row_i_syn.extend([i]*len(desc))
        all_row_chatgpt_names.append(row_chatgpt_names)
        all_row_key_name.append(row_key_name)
        all_row_weight.append([1]*len(row_chatgpt_names))
        all_row_i_syn.append(row_i_syn)

    all_row_classifier = build_classifier_chatgpt(all_row_chatgpt_names, model, all_row_key_name=all_row_key_name)

    # vfeatures = np.load(f'./cache/features/vfeatures-{args.dataset}.npy')
    """
    Args:
        all_clu_pred: shape
        all_row_classifier
        record_pred_kmeans_t
        all_row_weight
        all_row_i_syn
        vfeatures
    """
    vfeatures = all_vfeatures
    all_clu_pred_chatgpt = torch.zeros_like(all_clu_pred)
    is_correct = []
    all_name_gt = []
    k_2 = 3
    enable_weight = True
    instance_pred_voc = torch.zeros_like(record_pred_kmeans_t)
    for c in range(len(all_row_classifier)):
        select = (record_pred_kmeans_t==c)
        row_classifier = all_row_classifier[c]
        sim = torch.from_numpy(vfeatures[select, ...]).to(args.device)@row_classifier.to(args.device).t()
        sim_topk = sim.topk(k=k_2)
        ind, val = sim_topk.indices.flatten().cpu().unique(return_counts=True)
        count_names = torch.zeros(row_classifier.size(0)).long()
        count_names[ind] = val ### count of each name
        count_smask = []
        smask = np.array(all_row_i_syn[c]) ### partition mask
        for s in np.unique(smask):
            if enable_weight:
                row_weight = torch.tensor(all_row_weight[c]).float()
                row_weight[smask==s] = row_weight[smask==s] / row_weight[(smask==s)].sum()
                row_weight /= row_weight.sum()
                count_smask.append((row_weight[smask==s]*count_names[smask==s]).sum().item())
            else:
                count_smask.append(count_names[smask==s].sum())
        # name_pred = all_row_mapping_idx_synset_name[c][np.argmax(count_smask)]
        name_gt = all_gt_voc[select].mode().values
        name_gt = vocab.mapping_idx_names[name_gt.item()]
        all_name_gt.append(name_gt)
        # is_correct.append(name_pred==name_gt)
        # instance_pred_voc[select] = vocab.mapping_names_idx[name_pred]

        val_count = torch.tensor(count_smask)
        ind_count = topk_all_clu_pred[c]
        # ind_count = torch.tensor([vocab.mapping_names_idx[xx] for xx in ind_count])
        all_clu_pred_chatgpt[c, ind_count] = val_count

    
    classifier = get_classifier(args)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    args.num_voc = classifier.size(0)
    a, res_ass = linear_assign(all_clu_pred_chatgpt, record_pred_kmeans_t, all_gt_voc)
    r_pred_kmeans_t, r_cluster_ind_voc = reassign_by_pred_cluster(a, loader_f, model, classifier, args.device, preextracted_vfeatures=all_vfeatures)
    set_pred = set(res_ass[1].tolist())
    set_gt = set(all_gt_voc.unique().numpy().tolist())
    n_inter = all_gt_voc[r_cluster_ind_voc.cpu()==all_gt_voc].unique().shape[0]
    n_union = torch.cat([r_cluster_ind_voc.cpu(), all_gt_voc]).unique().shape[0]
    iou_voc = n_inter/n_union
    n_missing_label = all_gt_voc.unique().shape[0] - n_inter
    print('missing label::', n_missing_label)
    print('iou voc::', iou_voc)
    print('cluster acc', cluster_acc(y_true=all_label_clu.numpy(), y_pred=r_pred_kmeans_t.numpy()))
    n_inter = all_gt_voc[cluster_ind_voc.cpu()==all_gt_voc].unique().shape[0]
    n_union = torch.cat([cluster_ind_voc.cpu(), all_gt_voc]).unique().shape[0]
    iou_voc = n_inter/n_union
    n_missing_label = all_gt_voc.unique().shape[0] - n_inter
    print('missing label::', n_missing_label)