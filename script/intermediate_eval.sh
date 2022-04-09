export CUDA_VISIBLE_DEVICES=7
python ../spring/bin/predict_amrs.py \
    --datasets /home/cl/AMR_Multitask_Inter/Dataset/DP/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt  \
    --gold-path $1".gold_test.txt_0409" \
    --pred-path $1".pred_test.txt_10_0409" \
    --checkpoint $1 \
    --beam-size 10 \
    --batch-size 2500 \
    --device cuda \
    --penman-linearization --use-pointer-tokens