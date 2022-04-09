# ATP: AMRize Then Parse! Enhancing AMR Parsing with PseudoAMRs

Hi this is the source code of our paper "ATP: AMRize Then Parse! Enhancing AMR Parsing with PseudoAMRs" accepted by findings of NAACL 2022.

### News
 - 🎈 We have released the trained models and the test scripts. 2022.04.10

### Todos
 - 🎯 We are working on merging our training/preprocessing code with the amrlib repo. 
 - 🎯 Release Camera Ready Paper
## Brief Introduction
TLDR: SOTA AMR Parsing single model using only 40k extra data. **Rank 1st** model on Structrual-Related Scores (SRL and Reentrancy). Paper is coming soon.

As Abstract Meaning Representation (AMR) implicitly involves compound semantic annotations, we hypothesize auxiliary tasks which are semantically or formally related can better enhance AMR parsing. With carefully designed control experiments, we find that 1) Semantic role labeling (SRL) and dependency parsing (DP), would bring much more significant performance gain than unrelated tasks in the text-to-AMR transition. 2) To make a better fit for AMR, data from auxiliary tasks should be properly ``AMRized'' to PseudoAMR before training. 3) Intermediate-task training paradigm outperforms multitask learning when introducing auxiliary tasks to AMR parsing. 

From an empirical perspective, we propose a principled method to choose, reform, and train auxiliary tasks to boost AMR parsing. Extensive experiments show that our method achieves new state-of-the-art performance on in-distribution, out-of-distribution, low-resources benchmarks of AMR parsing.


<div align=center>
<img  src="./img.png"/>
</div>

## Requriments

Build envrionment for Spring
```bash
cd spring
conda create -n spring python=3.7
pip install -r requirements.txt
pip install -e .
# we use torch==1.11.0 and A40 GPU. lower torch version is fine.
```

Build envrionment for BLINK, Note that BLINK has some requirements conflicts with Spring, while the blinking script relies on both repos. So we build it upon Spring.
```bash
conda create -n blink37 -y python=3.7 && conda activate blink37

cd spring
pip install -r requirements.txt
pip install -e .

cd BLINK
pip install -r requirements.txt
pip install -e .
```
## Preprocess and AMRization

coming soon ~

## Training 

(cleaning code and data in progress)

```bash
cd spring/bin
```

- Train ATP-DP Task

```bash
python train.py --direction dp --config /home/cl/AMR_Multitask_Inter/spring/configs/config_dp.yaml
```

- Train ATP-SRL Task
```bash
python train.py --direction dp --config /home/cl/AMR_Multitask_Inter/spring/configs/config_srl.yaml 
# yes, the direction is also dp
```


- Train AMR Task based on intermediate ATP-SRL/DP Model

```
python train.py --direction amr --checkpoint PATH_TO_SRL_DP_MODEL --config ../configs/config.yaml
```

- Train AMR,SRL,DP Task in multitask Manner

```bash
python train.py --direction multi --config ../configs/config_multitask.yaml
```


## Inference

```bash
conda activate spring

cd script
bash intermediate_eval.sh MODEL_PATH 
# it will generate the gold and the parsed amr files, you should the change the path of AMR2.0/3.0 Dataset in the script.

conda activate blink37 
# you should download the blink model according to the readme in blink repo
bash blink.sh PARSED_AMR BLINK_MODEL_DIR

cd ../amr-evaluation
bash evaluation.sh PARSED_AMR.blink GOLD_AMR_PATH
```

## Models Release

You could refer to the inference section and download the models below to reproduce the result in our paper.

- ATP_SRL_AMR2.0 [Google Drive](https://drive.google.com/file/d/1MUJ6tW_0MY9SWAbRVEWXHzWYxx9nDx9W/view?usp=sharing)

```sh
#scores
Smatch -> P: 0.8587, R: 0.8437, F: 0.851
Unlabeled -> P: 0.890, R: 0.874, F: 0.882
No WSD -> -> P: 0.863, R: 0.848, F: 0.855
Concepts -> P: 0.914 , R: 0.895 , F: 0.904
Named Ent. -> P: 0.928 , R: 0.901 , F: 0.914
Negations -> P: 0.756 , R: 0.758 , F: 0.757
Wikification -> P: 0.849 , R: 0.824 , F: 0.836
Reentrancies -> P: 0.756 , R: 0.744 , F: 0.750
SRL -> P: 0.840 , R: 0.830 , F: 0.835
```
