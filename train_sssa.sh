
OMP_NUM_THREADS=2
DATASET=sdogs
EXP_ID=test

export TOKENIZERS_PARALLELISM=false

# nohup \
python -m train_sssa \
--dataset ${DATASET} \
--device 'cuda:3' \
--exp_id ${EXP_ID} \
--clip_model ViT-B/16 \
--vocab_name 'in21k' \
--total_iter 30000 \
--batch_size 128 \
--model_ema_decay 0.9998 \
--n_iter_cluster_vote 3 \
--w_ins 1.0 \
--uk 'False' \
--use_chatgpt 'True' \
--epoch_init_warmup 2 \
--w_str 0.25
# >> cache/${DATASET}-${EXP_ID}.log 2>&1 &
# --use_resume 'True' \
# --suffix "pos=0.4" \
# --resume_ckpt /home/sheng/MUST-output/${DATASET}/${EXP_ID}/checkpoint-current.pth
