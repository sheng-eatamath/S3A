
OMP_NUM_THREADS=2
DATASET=sdogs
EXP_ID=test

export TOKENIZERS_PARALLELISM=false
# EXP_ID=soft_semicl-tau_self=1.0-tau_sup=0.07-w_cl=0.5-v2
echo -e $(LC_TIME=en_US date)'\t'${DATASET}/${EXP_ID} >> /home/sheng/sssa/cache/exp_track.log

# nohup \
python -m train_lvscd_0722 \
--dataset ${DATASET} \
--device 'cuda:2' \
--exp_id ${EXP_ID} \
--clip_model ViT-B/16 \
--vocab_name 'in21k' \
--total_iter 30000 \
--batch_size 128 \
--model_ema_decay 0.9998 \
--n_step_scd 3 \
--w_loss_st 1.0 \
--uk 'False' \
--use_epoch_clustering 'False' \
--start_epoch_clustering 0 \
--per_epoch_clustering 10 \
--use_chatgpt 'True' \
--epoch_chatgpt_warmup 2 \
--llm_method 'vde' \
--use_kmeans 'False' \
--w_ssl_clu 0.25
# --suffix "pos=0.4" \
# >> cache/${DATASET}-${EXP_ID}.log 2>&1 &
# --use_resume 'True' \
# --resume_ckpt /home/sheng/MUST-output/${DATASET}/${EXP_ID}/checkpoint-current.pth
# --suffix "pos=0.05" \

# --fpath_preclustering '/home/sheng/MUST-output/make_nonliving26/baseline-04_22_1/pred_kmeans_t.npy' \

# --fpath_preclustering_chatgpt '' \
# --fpath_preclustering '/home/sheng/MUST-output/make_nonliving26/chatgpt_init-warmup=2/pred_kmeans_t.npy'
