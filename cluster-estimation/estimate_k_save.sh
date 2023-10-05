# Get unique log file
SAVE_DIR=/home/sheng/disk/output-gcd-cluster-estimation/
dataset_name='make_entity13'

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))

python -m a_estimate_k \
--max_classes 476 \
--min_classes 50 \
--vfeatures_fpath /home/sheng/MUST/ipynb/cache/features/vfeatures-${dataset_name}.npy \
--search_mode other \
--min_rand_sample 30000 \
--ratio_rand_sample 0.5 \
--method_kmeans 'kmeans' \
--save_prediction ${dataset_name}
