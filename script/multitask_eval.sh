python ../spring/bin/predict_amrs_multitask.py \
    --datasets /home/cl/AMR_Multitask_Inter/Dataset/DP/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt  \
    --gold-path $1".gold_test.txt" \
    --pred-path $1".pred_test.txt_10" \
    --checkpoint $1 \
    --beam-size 10 \
    --batch-size 2048 \
    --device cuda \
    --penman-linearization --use-pointer-tokens

# /home/cl/spring/amr_annotation_3.0/data/amrs/split/test
# /home/cl/AMR_Multitask_Inter/Dataset/DP/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt