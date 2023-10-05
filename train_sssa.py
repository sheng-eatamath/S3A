from misc import str2bool
from config import output_dir

import os, time
import datetime
import json
import random
from pathlib import Path
from collections import OrderedDict
import argparse
from contextlib import suppress
import warnings

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from ema import ModelEma
from data.build_dataset import build_transform
from data.vocab import get_vocab
from data.imagenet_datasets import get_datasets_rzsc
from self_training_sssa import *
from model import clip_classifier
from my_util_package import misc


def get_args():
    parser = argparse.ArgumentParser('MUST training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_freq', default=1, type=int) 
    
    # CLIP parameters
    parser.add_argument('--clip_model', default='ViT-B/16', help='pretrained clip model name') 
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
  
    # training parameters
    parser.add_argument("--train_config", default='train_configs.json', type=str, help='training configurations') 
    parser.add_argument('--model_ema_decay', type=float, default=0.9998, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # GENERAL
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--devices', type=str, default=None, help='list of devices for data parallelism, sep by `,`')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)

    # [NOTE: deprecated] model parallelism distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dist_on_itp', action='store_true')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    
    
    """ GENERAL """
    ### param for setup
    parser.add_argument('--exp_id', type=str, default=None, help="experiment name")
    parser.add_argument('--uk', type=str2bool, default=False, help="whether the K is unknown")
    parser.add_argument('--estimate_k', type=int, default=None, help="no need to specify; determined by estimation results")
    parser.add_argument('--total_iter', type=int, default=-1, help="total iter number for training")
    parser.add_argument('--n_iter_cluster_vote', type=int, default=3, help="iteration for iterative clustering-voting")
    parser.add_argument('--use_resume', type=str2bool, default=False, help="resume from checkpoint")
    parser.add_argument('--resume_ckpt', type=str, default=None, help="checkpoint path")
    
    ### dataset
    parser.add_argument('--dataset', default='imagenet', type=str, 
                        choices=['imagenet','make_entity13','make_living17','make_nonliving26','make_entity30','imagenet1k','sdogs','cifar100','caltech101','pet'], 
                        help='dataset name')
    parser.add_argument('--oov_dataset', type=str2bool, default=False)
    parser.add_argument('--n_sampled_classes', type=int, default=100, help="number of randomly sampled classes for ImageNet dataset")
    parser.add_argument('--nb_classes', default=0, type=int, help="class number; determined by dataset; no need to specify")
    parser.add_argument('--vocab_name', type=str, default='in21k', help="vocabulary name")
    
    ### method
    parser.add_argument('--use_chatgpt', type=str2bool, default=True)
    parser.add_argument('--epoch_init_warmup', type=int, default=2, help="number of epochs using offline structural pseudo-alignment without updating.")
    parser.add_argument('--w_ins', type=float, default=1.0, help="loss weight for instance semantic alignment loss")
    parser.add_argument('--w_str', type=float, default=0.25, help="loss weight for structural semantic alignment loss")
    parser.add_argument('--suffix', type=str, default="", help="for loading ablation prompting files")

    return parser.parse_args()


def main(args):
    if args.devices is not None:
        args.devices = [int(x) for x in args.devices.split(',')]
    device = torch.device(args.device)
    misc.seed_torch(args.seed)
    cudnn.benchmark = True
    
    # read config file
    train_configs = json.load(open(args.train_config,'r'))
    train_config = train_configs[args.dataset+'_'+args.clip_model]
    # create exp folder
    args.output_dir = os.path.join(output_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.exp_id)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = utils.TensorboardLogger(log_dir=args.output_dir)
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(dict(args._get_kwargs())) + "\n")
            
    print(args)
    print(train_config)

    
    ### model
    model = clip_classifier(args)
    args.nb_classes = model.num_classes
    model = model.to(args.device)
    model = module_to_device(args, model)
    vocab = get_vocab(args.vocab_name)
    args.num_voc = len(vocab)
    
    ### dataset
    transform_train = build_transform(is_train=True, args=args, train_config=train_config)
    transform_val = build_transform(is_train=False, args=args, train_config=None)
    dataset_train = get_datasets_rzsc(args, vocab=vocab, is_train=True, transform=transform_train)
    dataset_trainval = get_datasets_rzsc(args, vocab=vocab, is_train=True, transform=transform_val)
    dataset_val = get_datasets_rzsc(args, vocab=vocab, is_train=False, transform=transform_val)
    if args.uk: ### estimated cluster number for each datasets
        estimate_k = {'make_entity13': 252, 'make_living17': 73, 'make_nonliving26': 101, 'make_entity30': 206}
        try:
            args.estimate_k = estimate_k[args.dataset]
        except Exception as e:
            print(e)
            print(f'dataset={args.dataset} not implemented for estimated cluster numbers.')
        
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    data_loader_trainval = torch.utils.data.DataLoader(
        dataset_trainval,
        batch_size=8*args.batch_size//len(args.devices) if args.devices else 8*args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=False,
        shuffle=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=8*args.batch_size//len(args.devices) if args.devices else 8*args.batch_size,
        num_workers=args.num_workers,
        # pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    ### teacher
    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        resume='',
        device=args.device,
        devices=args.devices,
        )
    print("Using EMA with decay = %.5f" % (args.model_ema_decay) )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    
    ### control total iteration steps
    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train) if args.total_iter==-1 else min(len(data_loader_train), args.total_iter//train_config['epochs'])
    print(f'total_iter={args.total_iter} num_training_steps_per_epoch={num_training_steps_per_epoch}')

    args.lr = train_config['lr'] * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.epochs = train_config['epochs']
    args.eval_freq = train_config['eval_freq']
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_train))

    num_layers = model.module.model.visual.transformer.layers if hasattr(model, 'module') else model.model.visual.transformer.layers
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None
    if assigner is not None:
        print("LR Assigned values = %s" % str(assigner.values))
        

    ### setup schedule
    optimizer = create_optimizer(
        args, model.module if hasattr(model, 'module') else model,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    if args.amp:
        loss_scaler = NativeScaler()
        amp_autocast = torch.cuda.amp.autocast
    else:
        loss_scaler = None
        amp_autocast = suppress

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
        
        
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    epoch_clustering = None
    
    if args.use_resume:
        ckpt = torch.load(f'{args.resume_ckpt}', map_location='cpu')
        epoch_resume = ckpt['epoch']
        model_resume = ckpt['model']
        model_ema_resume = ckpt['model_ema']
        optimizer_resume = ckpt['optimizer']
        optimizer.load_state_dict(optimizer_resume)
        model.load_state_dict(model_resume)
        model.train()
        model_ema.ema.load_state_dict(model_ema_resume)
        model_ema.ema.eval()
        if args.oov_dataset:
            ssl_clustering_func = compute_ssl_clustering_simple_label
        else:
            ssl_clustering_func = compute_ssl_clustering_simple
        data = load_chatgpt_precomputed_results(args)
        epoch_clustering = current_epoch_clustering = data['current_epoch_clustering']
        current_epoch_clustering, cluster_ind_voc = ssl_clustering_func(
            args, model_ema.ema, None, data_loader_trainval, 
            epoch=epoch_resume, log_writer=log_writer, 
            pred_kmeans_t=epoch_clustering.cpu(),
            return_details=False,
            load_chatgpt=True,
            )
        print(f'recover from epoch={epoch_resume}')

    
    for epoch in range(args.start_epoch, args.epochs):
        log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        if args.use_resume and epoch<=epoch_resume:
            print(f'resume skip epoch={epoch}')
            continue
            
        ### use ssl-cluster supervision
        if args.oov_dataset:
            ssl_clustering_func = compute_ssl_clustering_simple_label
        else:
            ssl_clustering_func = compute_ssl_clustering_simple
        if args.use_chatgpt:
            ### warmup epoch
            if epoch < args.epoch_init_warmup:
                print('use chatgpt clustering')
                data = load_chatgpt_precomputed_results(args)
                epoch_clustering = current_epoch_clustering = data['current_epoch_clustering']
                cluster_ind_voc = data['cluster_ind_voc']
            else:
                current_epoch_clustering, cluster_ind_voc = ssl_clustering_func(
                    args, model_ema.ema, None, data_loader_trainval, 
                    epoch=epoch, log_writer=log_writer, 
                    pred_kmeans_t=epoch_clustering.cpu(),
                    return_details=False,
                    load_chatgpt=True,
                    )
        else:
            raise NotImplementedError()
        
        ### update structural pseudo-alignment
        if hasattr(dataset_train, 'dataset'):
            a = torch.zeros(len(dataset_train.dataset)).long()
            a[torch.tensor(dataset_train.indices).long()] = cluster_ind_voc.cpu().long()
            dataset_train.dataset.str_align = a
        else:
            dataset_train.str_align = cluster_ind_voc.cpu()
            dataset_trainval.str_align = cluster_ind_voc.cpu()

        train_stats = train_one_epoch(
            model, args, train_config,
            data_loader_train, optimizer, amp_autocast, device, epoch, loss_scaler, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            model_ema=model_ema,
            other_params={
            },
        )
        
        utils.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, post_fix='current')
        
        if args.output_dir and (epoch + 1) % args.eval_freq == 0:
            if args.oov_dataset:
                evaluate_func = evaluate_label
            else:
                evaluate_func = evaluate
            ### evalute on trainval
            trainval_stats, all_prj_features_ema = evaluate_func(data_loader_trainval, model, device, model_ema=model_ema, args=args, 
                                  other_params={
                                      'vocab': vocab,
                                  }, 
                                  log_writer=log_writer,
                                  )
            ### evaluate on test
            test_stats, _ = evaluate_func(data_loader_val, model, device, model_ema=model_ema, args=args, 
                                  other_params={
                                      'vocab': vocab,
                                  }, 
                                  log_writer=log_writer,
                                  )
            
            ### logging
            if args.oov_dataset:
                keyname = 'sim_bert'
            else:
                keyname = 'acc1'
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats[keyname]}%")
            if max_accuracy < test_stats[keyname]:
                max_accuracy = test_stats[keyname]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, post_fix='best')

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(**test_stats, head="test", step=epoch)
                log_writer.update(**trainval_stats, head="val", step=epoch)
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'val_{k}': v for k, v in trainval_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    return


if __name__ == '__main__':
    opts = get_args()
    main(opts)