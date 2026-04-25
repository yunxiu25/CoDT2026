# DICTRec
# DICTRec: Debiased Intervention and Co-Evolutionary Tokenization for Generative Recommendation

> This repository provides the official implementation of **DICTRec**: an LLM-based generative recommendation framework that integrates IPS causal debiasing and graph-semantic co-evolutionary alignment. It **significantly mitigates** popularity bias to ensure long-tail fairness, while achieving millisecond-level, industrial-grade inference efficiency via MIPS retrieval.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ggiokkll/DICTRec2026/blob/main/LICENSE)



## An example of Implementation

Please download the checkpoints at [Google Drive](https://drive.google.com/drive/folders/12OFUuX7a5v7khx_MZiel04N0x5prkdGy?usp=drive_link), and put them in the path of "checkpoints/".

1. **Full Model**
```
python cd code
python main.py --dataset LastFM --vq --train_vq
```

2. **w/o IPS**
```
python main.py --dataset LastFM --vq --train_vq --no_ips
```

3. **w/o Co-evolution**
```
python main.py --dataset LastFM --vq --train_vq --freeze_codebook
```

4. **w/o Alignment**
```
python main.py --dataset LastFM --vq --train_vq --no_align
```

4. **Evaluation**
```
python main.py --dataset LastFM --no_train
```


