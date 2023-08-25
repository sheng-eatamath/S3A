import os
import numpy as np
from functools import reduce

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import networkx as nx


from robustness.tools import breeds_helpers
from robustness.tools.breeds_helpers import ClassHierarchy


info_dir = '/home/sheng/OSZSL/breeds_hier/modified'

def get_hier_datasets(dataset_name='make_entity30'):
    hier = ClassHierarchy(info_dir)
    dataset_hier = getattr(breeds_helpers, dataset_name)(info_dir, split=None)
    return dataset_hier, hier


class ImageNetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, vocab, transform, hier_dataset_name=None):
        """ 
        Args:
            hier_dataset_name [BREEDS, None=IN-1k]
        """
        super(ImageNetDataset, self).__init__(root, transform)
        self.vocab = vocab
        if hier_dataset_name is None:
            dataset_hier = hier = None
        else:
            dataset_hier, hier = get_hier_datasets(dataset_name=hier_dataset_name)
        self.preprocess(dataset_hier, hier)
        return
    
    def preprocess(self, dataset_hier=None, hier=None):
        """
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
        
        self.labels = list(map(lambda x: self.vocab.mapping_ids_idx[x], self.labels)) ### to @vocab_idx
        self.num_classes = len(set(self.labels))
        self.label_transform = {}
        for c, i in zip(set(self.labels), range(self.num_classes)):
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
        return img, label_voc, label_clu, idx_img
    

def get_classifier(args):
    if args.use_def:
        classifier = torch.load('./cache/wordnet_classifier_def.pth')
    else:
        classifier = torch.load('./cache/wordnet_classifier.pth')
    if args.f_classifier is not None:
        classifier = torch.load(args.f_classifier)
    classifier = classifier.to(args.device)
    return classifier



def MCMF_assign_labels(M, K):
    C, V = M.shape

    print('Create the directed graph with C+V+2 nodes')
    G = nx.DiGraph()

    print('Add the source node (node 0)')
    G.add_node(0)

    print('Add the sink node (node C+V+1)')
    G.add_node(C+V+1)

    print('Add the nodes for the C classes (nodes 1 to C)')
    G.add_nodes_from(range(1, C+1))

    print('Add the nodes for the V vocabulary words (nodes C+1 to C+V)')
    G.add_nodes_from(range(C+1, C+V+1))

    print('Add the edges between the classes and the vocabulary words')
    x_ind, y_ind = M.nonzero()
    for i in range(len(x_ind)):
        G.add_edge(x_ind[i]+1, y_ind[i]+1+C, capacity=1, weight=-int(M[x_ind[i], y_ind[i]]*10000))
    # for i in range(1, C+1):
    #     for j in range(C+1, C+V+1):
    #         if M[i-1][j-C-1]!=0:
    #             G.add_edge(i, j, capacity=1, weight=-M[i-1][j-C-1]*1000)

    print('Add the edges between the vocabulary words and the sink')
    for j in range(C+1, C+V+1):
        G.add_edge(j, C+V+1, capacity=1, weight=0)

    print('Add the edges between the source and the classes')
    for i in range(1, C+1):
        G.add_edge(0, i, capacity=K, weight=0)

    print('Find the minimum cost maximum flow in the graph')
    flow_dict = nx.max_flow_min_cost(G, 0, C+V+1)

    print('Extract the labels assigned to each class')
    O = np.zeros((C, V), dtype=np.int8)
    for i in range(1, C+1):
        for j in flow_dict[i]:
            if j != 0 and j != C+V+1:
                if flow_dict[i][j] == 1:
                    O[i-1][j-C-1] = 1
    print(f'O={np.unique(O.sum(0)), O.shape}')
    class_topk_assignment = torch.from_numpy(O.nonzero()[1].reshape(-1, K))
    return class_topk_assignment


