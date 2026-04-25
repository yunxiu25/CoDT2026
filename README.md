# DICTRec
# DICTRec: Debiased Intervention and Co-Evolutionary Tokenization for Generative Recommendation

> This repository provides the official implementation of **DICTRec**: an LLM-based generative recommendation framework that integrates IPS causal debiasing and graph-semantic co-evolutionary alignment. It **significantly mitigates** popularity bias to ensure long-tail fairness, while achieving millisecond-level, industrial-grade inference efficiency via MIPS retrieval.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ggiokkll/DICTRec2026/blob/main/LICENSE)

## 📦 Model Weights Preparation

To reproduce the results, please download the corresponding weights from the links below and place them in their respective directories:

* **DICTRec Checkpoints:** Download the pre-trained checkpoints from [Google Drive](https://drive.google.com/file/d/1aUes_IddRy5jG7VpPx6tcxqUqyovgCi7/view?usp=drive_link) and place them inside the `checkpoints/` directory.
* **t5-small Backbone:** Download the pre-trained LLM weights from [Google Drive](https://drive.google.com/file/d/1zcMJGZJoo1b6VVJcIUKF1Z-42ih0g-ED/view?usp=sharing) and place them inside the `t5-small/` directory.

### 📂 Directory Structure
After downloading, please ensure your project hierarchy is organized exactly as follows:

```text
📦 DICTRec
 ├── 📂 checkpoints/           # -> Put DICTRec pre-trained checkpoints here
 │    ├── 📂 backbone
 │    └── 📂 vq
 ├── 📂 src/              # -> Put t5-small backbone weights here
 │    ├── 📂 lgn
 │    └── 📂 t5-small
 ├── 📂 data/
 ├── 📂 code/
 └── 📄 README.md

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


