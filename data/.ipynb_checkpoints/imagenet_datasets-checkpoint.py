import os
import numpy as np
from functools import reduce
from collections import Counter, defaultdict
import scipy.io
from PIL import Image
from pathlib import Path

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode 
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100
from timm.data import create_transform

from robustness.tools import breeds_helpers
from robustness.tools.breeds_helpers import ClassHierarchy

from config import *
from data.vocab import get_vocab
from nltk.corpus import wordnet as wn


def get_hier_datasets(dataset_name='make_entity30'):
    hier = ClassHierarchy(info_dir)
    dataset_hier = getattr(breeds_helpers, dataset_name)(info_dir, split=None)
    return dataset_hier, hier

in100_subsample_classes = [
        993, 859, 298, 553, 672, 971,  27, 231, 306, 706, 496, 558, 784,
       239, 578,  55, 906, 175,  14,  77,  31, 481, 310, 311, 883, 788,
        45, 103, 760,   1, 823, 710, 614, 790, 408, 736, 957, 366, 918,
       267, 230, 996, 635, 698, 251, 783, 819, 141, 316, 587, 331, 295,
       262, 432, 862, 582, 272, 270, 987, 319, 569, 643, 142, 202, 413,
       196, 264, 531, 252, 576, 738, 299, 740, 247, 926, 412, 389, 796,
       601, 654, 261, 456, 386, 982, 909, 693, 236, 501, 497, 874, 452,
       494, 923, 279, 638, 485, 568, 108, 367, 644
]

def get_datasets_rzsc(args, vocab=None, is_train=False, transform=None, seed=0, **kwargs):
    if is_train:
        root_dataset = root_imagenet
    else:
        root_dataset = root_imagenet_val
        
    np.random.seed(seed)
    if args.dataset in ['imagenet']:
        args.sub_classes = np.random.choice(range(1000), size=(args.n_sampled_classes), replace=False)
        dataset = ImageNetDataset(root=root_dataset, vocab=vocab, transform=transform, sub_classes=args.sub_classes, hier_dataset_name=None)
        dataset.subset_classes(args.sub_classes.tolist())
        dataset.preprocess(None, None)
        dataset.str_align = None
    elif args.dataset in ['make_entity13', 'make_living17', 'make_nonliving26', 'make_entity30']:
        dataset = ImageNetDataset(root=root_dataset, vocab=vocab, transform=transform, sub_classes=None, hier_dataset_name=args.dataset)
        dataset_hier, hier = get_hier_datasets(dataset_name=args.dataset)
        dataset.subset_classes(None)
        dataset.preprocess(dataset_hier, hier)
        dataset.str_align = None
    elif args.dataset in ['imagenet1k']:
        args.sub_classes = np.random.choice(range(1000), size=(1000), replace=False)
        dataset = ImageNetDataset(root=root_dataset, vocab=vocab, transform=transform, sub_classes=args.sub_classes, hier_dataset_name=None)
        dataset.subset_classes(args.sub_classes.tolist())
        dataset.preprocess(None, None)
        dataset.str_align = None
    elif args.dataset == 'sdogs':
        dataset = SDOGS(root=f'{HOME}/dataset/StanfordDogs', train=is_train, transform=transform, vocab=vocab)
        dataset.str_align = None
    elif args.dataset == 'cifar100':
        args.oov_dataset = True
        dataset = CustomCIFAR100(root=f'{HOME}/dataset/CIFAR100/', train=is_train, transform=transform, vocab=vocab)
        dataset.str_align = None
    elif args.dataset == 'caltech101':
        args.oov_dataset = True
        dataset = CaltechDataset(root=f'{HOME}/dataset/Caltech101/caltech-101/caltech101/101_ObjectCategories/', split='train' if is_train else 'test', transform=transform, vocab=vocab, **kwargs)
        dataset.preprocess()
        dataset.str_align = None
    elif args.dataset == 'pet':
        args.oov_dataset = True
        dataset = OxfordIIITPet(root=f'{HOME}/dataset/pet', split='trainval' if is_train else 'test', transform=transform, vocab=vocab)
    else:
        raise NotImplementedError()
    return dataset


class ImageNetDataset(datasets.ImageFolder):
    def __init__(self, root, vocab=None, transform=None, sub_classes=None, hier_dataset_name=None):
        """ 
        Args:
            hier_dataset_name [BREEDS, None=IN-1k]
        """
        super(ImageNetDataset, self).__init__(root, transform)
        self.vocab = vocab
        return
    
    def subset_classes(self, sub_classes=None):
        if sub_classes is None:
            return
        else:
            sub_classes = set(sub_classes)
            self.samples = list(filter(lambda x: x[1] in sub_classes, self.samples))
        return 
    
    def preprocess(self, dataset_hier=None, hier=None):
        """
        [tested]
            1. filtering @self.labels, @self.samples
            2. @label to @vocab_idx; @label to @label_transformed
        """
        self.labels = list(map(lambda x: self.classes[x[1]], self.samples))
        
        if (dataset_hier is not None) and (hier is not None):
            _, (subclasses, _), _ = dataset_hier
            # convert subclasses idx to @synsetid
            idx_subclasses = list(reduce(lambda x, y: x+y, subclasses))
            synset_subclasses = list(map(lambda x: hier.LEAF_IDS[x], idx_subclasses))
            set_synset_subclasses = synset_subclasses # speedup
            is_valid = list(map(lambda x: x in set_synset_subclasses, self.labels))
            # subseting
            self.labels = np.array(self.labels)[is_valid].tolist()
            self.samples = np.array(self.samples)[is_valid].tolist()
        
        if self.vocab is not None:
            self.labels = list(map(lambda x: self.vocab.mapping_names_idx[self.vocab.mapping_ids_names[x]], self.labels)) ### to @vocab_idx
            # self.labels = list(map(lambda x: self.vocab.mapping_ids_global_idx[x], self.labels))
            # self.labels = list(map(lambda x: self.vocab.mapping_names_idx[self.vocab.mapping_ids_names[x]] if x in self.vocab.mapping_ids_names else -1, self.labels))
        else:   
            self.labels = [ wn.synset_from_pos_and_offset('n', int(x[1:])).name().split('.')[0] for x in self.labels]
        self.num_classes = len(set(self.labels))
        self.label_transform = {}
        for c, i in zip(sorted(set(self.labels)), range(self.num_classes)):
            self.label_transform[c] = i
        self.labels_transformed = list(map(lambda x: self.label_transform[x], self.labels))
        self.idx_imgs = np.array(range(len(self.samples)))
        return
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        label_voc = self.labels[idx]
        label_clu = self.labels_transformed[idx]
        idx_img = self.idx_imgs[idx]
        result = [img, label_voc, label_clu, idx_img]
        if self.str_align is not None:
            result.append(self.str_align[idx])
        return result
    
    @property
    def len_output(self):
        return 4 if self.str_align is None else 5


class SDOGS(ImageFolder):
    folder = 'StanfordDogs'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 vocab=None,
                ):
        super(SDOGS, self).__init__(root=os.path.join(root, 'Images'))
        
        self.vocab = vocab
        self.root = root
        self.target_transform = None
        self.classnames = list(map(lambda x: x.split('-')[1], self.classes))
        self.classes = list(map(lambda x: x.split('-')[0], self.classes))
        self.num_classes = len(set(self.classes))
        
        self.train = train
        self.transform = transform

        split = self.load_split()
        sub_samples, sub_targets = [], []
        for i, x in enumerate(self.samples):
            cond = x[0].split('/')[-1].split('.')[0] in split
            if cond:
                sub_samples.append(self.samples[i])
                sub_targets.append(self.targets[i])
        self.samples = sub_samples
        self.targets = sub_targets

        self.preprocess()
        return
    
    def preprocess(self):
        self.labels = list(map(lambda x: self.classes[x[1]], self.samples))
        if self.vocab is not None:
            self.labels = list(map(lambda x: self.vocab.mapping_ids_names[x], self.labels))
            self.labels = list(map(lambda x: self.vocab.mapping_names_idx[x], self.labels)) ### to @vocab_idx
        else:
            self.labels = list(map(lambda x: wn.synset_from_pos_and_offset('n', int(self.classes[x[1]][1:])).name().split('.')[0], self.samples))
            self.labels_transformed = []
        self.idx_imgs = np.array(range(len(self.samples)))
        return
    
    def __getitem__(self, idx):
        fimg, label_clu = self.samples[idx]
        image = Image.open(fimg).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_voc = self.labels[idx]
        idx_img = idx
        result = [image, label_voc, label_clu, idx_img]
        if self.str_align is not None:
            result.append(self.str_align[idx])
        return result
    
    def __len__(self):
        return super().__len__()
    
    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']
        split = [item[0][0].split('/')[-1] for item in split]
        return set(split)
    
    def get_img_name(self):
        return list(map(lambda x: x[0].split('/')[-1].split('.')[0], self.samples))
    




class CaltechDataset(ImageFolder):
    def __init__(self, root, vocab=None, transform=None, split='train', **kwargs):
        self.root = root
        self.vocab = vocab
        self.transform = transform
        assert split in ['train', 'test']
        self.split = split
        self.split_ratio = 0.8
        
        self.category_mapping_caltech101 = {
        }
        self.category_remove_caltech101 = ['yin_yang', 'background_google']
        self.parse_files()
        self.map_classes()
        self.random_split()
        return
    
    def parse_files(self):
        samples = []
        targets = []
        folder_path = Path(self.root)
        for p in folder_path.glob('**/*/*'):
            p = str(p)
            if '.ipynb_checkpoints' not in p.split('/'):
                samples.append(p)
                targets.append(p.split('/')[-2])
        self.samples = samples
        self.targets = targets
        return
    
    def map_classes(self):
        new_targets = []
        valid_inds = []
        for i, c in enumerate(self.targets):
            c = c.lower()
            if c in self.category_mapping_caltech101.keys():
                new_targets.append(self.category_mapping_caltech101[c])
                valid_inds.append(i)
            elif c in self.category_remove_caltech101:
                pass
            else:
                new_targets.append(c)
                valid_inds.append(i)
        self.targets = np.array(new_targets)
        self.samples = np.array(self.samples)[np.array(valid_inds)]
        return
    
    def random_split(self):
        np.random.seed(0)
        all_select = np.zeros(len(self.targets)).astype(np.bool)
        for c in np.unique(self.targets):
            select = (self.targets==c)
            position = select.nonzero()[0]
            sampled_ind = np.random.choice(position, int(self.split_ratio*select.sum()), replace=False)
            select = np.zeros_like(select)
            select[sampled_ind] = True
            all_select |= select
        if self.split == 'train':
            self.samples = self.samples[all_select]
            self.targets = self.targets[all_select]
        else:
            self.samples = self.samples[~all_select]
            self.targets = self.targets[~all_select]
        return 
    
    def preprocess(self):
        """
        [tested]
            1. filtering @self.labels, @self.samples
            2. @label to @vocab_idx; @label to @label_transformed
        """
        
        # if self.vocab is not None:
        #     self.targets = list(map(lambda x: self.vocab.mapping_names_idx[x], self.targets)) ### to @voc_ind
        self.num_classes = len(set(self.targets))
        self.label_transform = {}
        for c, i in zip(sorted(set(self.targets)), range(self.num_classes)):
            self.label_transform[c] = i
        self.labels_transformed = list(map(lambda x: self.label_transform[x], self.targets))
        self.idx_imgs = np.array(range(len(self.samples)))
        return
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img = self.samples[idx]
        img = Image.open(img).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label_voc = self.targets[idx]
        label_clu = self.labels_transformed[idx]
        idx_img = self.idx_imgs[idx]
        result = [img, label_voc, label_clu, idx_img]
        if self.str_align is not None:
            result.append(self.str_align[idx])
        return result
    
    @property
    def len_output(self):
        return 4 if self.str_align is None else 5


class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, transform=None, vocab=None, **kwargs):
        super(CustomCIFAR100, self).__init__(root=root, train=train, transform=transform, **kwargs)
        self.vocab = vocab
        self.uq_idxs = np.array(range(len(self)))
        self.category_mapping = {
        }
        self.classes = [self.category_mapping[c] if c in self.category_mapping.keys() else c for c in self.classes]
        # self.class_to_idx = dict([(category_mapping[k], v) if k in category_mapping else (k,v) for k, v in self.class_to_idx.items()])
        self.num_classes = len(set(self.classes))
        self.label_voc = list(map(lambda x: self.classes[x], self.targets))
        return

    def __getitem__(self, item):
        img, label_clu = super().__getitem__(item)
        label_voc = self.label_voc[item]
        idx_img = self.uq_idxs[item]
        result = [img, label_voc, label_clu, idx_img]
        if self.str_align is not None:
            result.append(self.str_align[item])
        return result

    def __len__(self):
        return len(self.targets)
    
    

    
class OxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    def __init__(self, root, split='trainval', transform=None, vocab=None, **kwargs):
        super(OxfordIIITPet, self).__init__(root=root, split=split, transform=transform, **kwargs)
        self.vocab = vocab
        self.uq_idxs = np.array(range(len(self)))
        self.classes = dict([(self.class_to_idx[c], c.lower()) for c in self.classes])
        self.str_align = None
        self.num_classes = len(set(self._labels))
        return

    def __getitem__(self, item):
        img, label_clu = super().__getitem__(item)
        label_voc = self.classes[label_clu]
        idx_img = self.uq_idxs[item]
        result = [img, label_voc, label_clu, idx_img]
        if self.str_align is not None:
            result.append(self.str_align[item])
        return result
    
    
    
    
    
### ============================================================
###     VOCAB
### ============================================================
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