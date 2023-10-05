import numpy as np 
import torch 
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn import metrics

def cluster_acc(y_true, y_pred, return_confm=False, eps=1e-10, convert_ind=False, return_pred=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    try:
        assert y_pred.size == y_true.size
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T ### [idx, mapping_idx]
    
    ### >>> confusion matrix
    m = w[np.argsort(ind[:, 1])]
    m = m/(m.sum(axis=1, keepdims=True)+eps)
    ### <<<
    
    ### >>> pairwise confusion matrix
    assigned_y_pred = np.take(ind[:, 1], y_pred)
    pconfm = pair_confusion_matrix(y_true, assigned_y_pred)
    pairwise_recall = (pconfm/pconfm.sum(axis=1, keepdims=True))[1,1]
    pairwise_precision = (pconfm/pconfm.sum(axis=0, keepdims=True))[1,1]
    ### <<<
    
    if convert_ind:
        y_pred = np.array(list(map(lambda x: ind[x, 1], y_pred)))

    if return_confm:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, m, pairwise_recall, pairwise_precision
    else:
        if return_pred:
            return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, pred
        else:
            return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
        
        
        
def measure_similarity_bert(vocab, all_pred_voc, all_gt_voc, base=['nli-bert-large', 'all-mpnet-base-v2'], device='cuda:0'):
    from sentence_transformers import SentenceTransformer, util
    all_scores = {}
    for b in base:
        model = SentenceTransformer(b, device=device)

        set_all_gt_voc = list(set(all_gt_voc))
        embeddings_gt = torch.tensor(model.encode(set_all_gt_voc))
        mapping_set_all_gt_voc = {}
        for i, w in enumerate(set_all_gt_voc):
            mapping_set_all_gt_voc[w] = embeddings_gt[i]

        if isinstance(all_pred_voc[0], str):
            all_pred_voc_names = all_pred_voc
        else:
            all_pred_voc_names = [vocab.mapping_idx_names[x.item()] for x in all_pred_voc]
        set_all_pred_voc = list(set(all_pred_voc_names))
        embeddings_pred = torch.tensor(model.encode(set_all_pred_voc))
        mapping_set_all_pred_voc = {}
        for i, w in enumerate(set_all_pred_voc):
            mapping_set_all_pred_voc[w] = embeddings_pred[i]

        scores = []
        for pred, gt in zip(all_pred_voc_names, all_gt_voc):
            sim = util.cos_sim(mapping_set_all_pred_voc[pred], mapping_set_all_gt_voc[gt])
            scores.append(sim.item())
            
        all_scores[b] = np.mean(scores)

    return all_scores
    
    