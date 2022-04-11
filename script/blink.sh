export CUDA_VISIBLE_DEVICES=7
python ../spring/bin/blinkify.py --datasets $1 --out $1.blink --device cuda --blink-models-dir $2/