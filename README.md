# CoDT: Co-evolution based Debiased Tokenization for LLM-based Recommendation

> This repository provides the official implementation of **CoDT**: an LLM-based recommendation framework that integrates popularity-aware re-weighting and dynamic graph-semantic co-evolutionary alignment. It effectively mitigates popularity bias to improve long-tail representation, while achieving millisecond-level inference efficiency via dense matching retrieval.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)]()
[![License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ggiokkll/DICTRec2026/blob/main/LICENSE)

## 📦 Model Weights Preparation

To reproduce the results, please download the corresponding weights from the links below and place them in their respective directories:

* **CoDT Checkpoints:** Download the pre-trained checkpoints from [Google Drive](https://drive.google.com/file/d/1aUes_IddRy5jG7VpPx6tcxqUqyovgCi7/view?usp=drive_link) and place them inside the `checkpoints/` directory.
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
```

## 🛠️ Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## An example of Implementation

1. **Full Model**
```
python cd code
python main.py --dataset LastFM --vq --train_vq
```

2. **w/o Re-weighting**
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


