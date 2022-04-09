python predict_hidden_states.py \
    --datasets /home/cl/AMR_Multitask_Inter/Dataset/DP/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt \
    --gold-path gold.tmp \
    --pred-path pred_dp.tmp \
    --checkpoint $1  \
    --batch-size 1 \
    --device cuda \
    --penman-linearization --use-pointer-tokens