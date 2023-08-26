
OMP_NUM_THREADS=2
DATASET=sdogs
EXP_ID=test

export TOKENIZERS_PARALLELISM=false
echo -e $(LC_TIME=en_US date)'\t'${DATASET}/${EXP_ID} >> /home/sheng/sheng-eatamath/S3A/sssa/cache/exp_track.log

# nohup \
python -m train_lvscd \
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
# >> cache/${DATASET}-${EXP_ID}.log 2>&1 &
# --use_resume 'True' \
# --suffix "pos=0.4" \
# --resume_ckpt /home/sheng/MUST-output/${DATASET}/${EXP_ID}/checkpoint-current.pth
