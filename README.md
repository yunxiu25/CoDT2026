```markdown
# CoDT: Co-evolving Debiased Tokenization for LLM-based Recommendation

> Official implementation of **CoDT**: an LLM-based recommendation framework that integrates popularity-aware re-weighting and dynamic co-evolutionary alignment to mitigate codebook collapse and improve long-tail recommendation, while achieving millisecond-level inference via dense retrieval.

## Model Weights Preparation

Download the pre-trained checkpoints from [Google Drive](https://drive.google.com/file/d/1aUes_IddRy5jG7VpPx6tcxqUqyovgCi7/view?usp=drive_link) and place them inside `checkpoints/`. Download the t5-small backbone from [Google Drive](https://drive.google.com/file/d/1zcMJGZJoo1b6VVNcIUKF1Z-42ih0g-ED/view?usp=sharing) and place it inside `src/t5-small/`.

## Directory Structure

```text
CoDT
 ├── checkpoints/           # pre-trained CoDT checkpoints
 │    ├── backbone/
 │    └── vq/
 ├── src/
 │    ├── lgn/              # LightGCN embeddings
 │    └── t5-small/         # LLM backbone weights
 ├── data/                  # preprocessed dataset splits
 ├── code/                  # training & evaluation scripts
 └── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Training

1. **Full CoDT**

```bash
cd code
python main.py --dataset LastFM --vq --train_vq
```

2. **Ablation: w/o Re-weighting**

```bash
python main.py --dataset LastFM --vq --train_vq --no_ips
```

3. **Ablation: w/o Co-evolution**

```bash
python main.py --dataset LastFM --vq --train_vq --freeze_codebook
```

4. **Ablation: w/o Alignment**

```bash
python main.py --dataset LastFM --vq --train_vq --no_align
```

## Evaluation

5. **Overall Performance**

```bash
python main.py --dataset LastFM --no_train
```

6. **Group-Normalized Long-Tail Evaluation** (Table 5 in the paper)

```bash
python groupnorm_eval.py --repo . --dataset ML1M --model seq --fair_lambda 0.1 --tag codt --save_lists
python groupnorm_eval.py --repo . --dataset ML1M --model lightgcn --tag lightgcn
```

7. **Codebook Utilization Statistics** (Table 9 in the paper)

```bash
python codebook_util.py
```

```
