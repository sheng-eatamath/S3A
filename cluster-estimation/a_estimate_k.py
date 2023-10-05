import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans

import torch
from torch.utils.data import DataLoader


# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_kmeans_for_scipy(K, data_dict, args=None, verbose=False):
    
    K = int(K)
    vfeatures = data_dict['vfeatures']
    labels = data_dict['labels']
    
    # -----------------------
    # K-MEANS
    # -----------------------
    print(f'Fitting K-Means for K = {K}...')
    if args.method_kmeans == 'minibatch-kmeans':
        kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=0, max_iter=300).fit(vfeatures)
    elif args.method_kmeans == 'kmeans':
        kmeans = KMeans(n_clusters=K, random_state=0).fit(vfeatures)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    print(f'K = {K}')
    sil = silhouette_score(vfeatures, preds)
    print(f'sil={sil}')
    
    if args.save_prediction is not None:
        np.save(f'./cache/{args.save_prediction}-clustering_pred-{args.K}.npy', preds)
    return -sil


def binary_search(data_dict, args):

    min_classes = args.min_classes

    # Iter 0
    big_k = args.max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    acc_big = - test_kmeans_for_scipy(big_k, data_dict, args)
    acc_small = - test_kmeans_for_scipy(small_k, data_dict, args)
    acc_middle = - test_kmeans_for_scipy(middle_k, data_dict, args)

    print(f'Iter 0: BigK {big_k}, Score {acc_big:.4f} | MiddleK {middle_k}, Score {acc_middle:.4f} | SmallK {small_k}, Score {acc_small:.4f} ')
    all_accs = [acc_small, acc_middle, acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if acc_big > acc_small:

            best_acc = max(acc_middle, acc_big)

            small_k = middle_k
            acc_small = acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            best_acc = max(acc_middle, acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            acc_big = acc_middle

        acc_middle = - test_kmeans_for_scipy(middle_k, data_dict, args)

        print(f'Iter {i}: BigK {big_k}, Score {acc_big:.4f} | MiddleK {middle_k}, Score {acc_middle:.4f} | SmallK {small_k}, Score {acc_small:.4f} ')
        all_accs = [acc_small, acc_middle, acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Score so far {best_acc_so_far:.4f} at K {best_acc_at_k}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_classes', default=1000, type=int)
    parser.add_argument('--min_classes', default=10, type=int)
    parser.add_argument('--data_fpath', type=str, default=None)
    parser.add_argument('--vfeatures_fpath', type=str, default=None)
    parser.add_argument('--ratio_rand_sample', type=float, default=0.1)
    parser.add_argument('--min_rand_sample', type=int, default=100*300)
    parser.add_argument('--method_kmeans', type=str, default='kmeans', choices=['kmeans', 'minibatch-kmeans'])
    parser.add_argument('--save_prediction', type=str, default=None)
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    print(args)
    device = args.device
    # assert args.data_fpath is not None
    assert args.vfeatures_fpath is not None

    # ----------------------
    # GLOBAL VARIABLES
    # ----------------------
    cluster_accs = {}

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')
    if args.data_fpath is not None:
        features = torch.load(args.data_fpath)
        labels = features['all_label_clu'].cpu()
    else:
        labels = None
    vfeatures = np.load(args.vfeatures_fpath)
    
    np.random.seed(43)
    N = vfeatures.shape[0]
    sampled_indices = np.random.permutation(N)[:max(args.min_rand_sample, int(N*args.ratio_rand_sample))]
    print(f'randomly sample {len(sampled_indices)} samples')
    labels = labels[sampled_indices] if labels is not None else None
    vfeatures = vfeatures[sampled_indices]
    data_dict={
        'labels': labels,
        'vfeatures': vfeatures,
    }
    binary_search(data_dict, args)

    