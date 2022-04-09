# ATP: AMRize Then Parse! Enhancing AMR Parsing with PseudoAMRs

Hi this is the source code of our paper "ATP: AMRize Then Parse! Enhancing AMR Parsing with PseudoAMRs" accepted by findings of NAACL 2022.

We have released the trained models and the test scripts. We are working on merging our training/preprocessing code with the amrlib repo. 
## Brief Introduction
SOTA AMR Parsing model using only 40k extra data. **Rank 1st** model on Structrual-Related Scores (SRL and Reentrancy). Paper is coming soon.

<div align=center>
<img width="600" src="./img.png"/>
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

## Inference

```bash
conda activate spring

cd script
bash intermediate_eval.sh MODEL_PATH 
# it will generate the gold and the parsed amr files, you should the change the path of AMR2.0/3.0 Dataset in the script.

conda activate blink37 
# you should download the blink model according to the readme in blink repo
bash blink.sh PARSED_AMR BLINK_MODEL_DIR

cd amr-evaluation
bash evaluation.sh PARSED_AMR.blink GOLD_AMR_PATH
```
## Models Release

- ATP_SRL_AMR2.0 [comming soon](www.baidu.com)

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

- ATP_SRL_Ensemble_AMR2.0 [comming soon](www.baidu.com)
- ATP_SRL_AMR3.0 [comming soon](www.baidu.com)
- ATP_DP_AMR2.0 [comming soon](www.baidu.com)
