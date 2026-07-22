import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go CoDT")

    # ----------------- t5 ----------------
    parser.add_argument('--item_limit', type=int, default=50, help="the maximum number of item list")
    parser.add_argument('--k', type=int, default=10, help="the number of top k setting")
    parser.add_argument('--kernel', type=int, default=32, help="the kernel size of average pooling")
    parser.add_argument('--lr', type=float, default=5e-5, help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-2, help="the weight decay for optimizer")
    parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
    parser.add_argument('--batch', type=int, default=128, help="the batch size of training and test")
    parser.add_argument('--valid_batch', type=int, default=24, help="the batch size of valid set")
    parser.add_argument('--no_data_augment', action='store_true', default=False,
                        help="whether apply the data augmentation process or not")
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help="if shuffle the item list of each sample or not. Default=False.")
    parser.add_argument('--train_from_checkpoint', action='store_true', default=False,
                        help="if training from the checkpoint: True or False (default).")
    parser.add_argument('--source_length', type=int, default=512, help="the maximum length of input tokens")
    parser.add_argument('--similarity', type=str, default='cos', help="the similarity metric. available: [cos, MSE]")
    parser.add_argument('--target_length', type=int, default=20, help="the maximum length of output tokens")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--is_unseen', action='store_true', default=False)
    parser.add_argument('--decoder_prepend', type=str,
                        default="The interaction history shows that the user preference could be")

    # ---------------- vq -----------------------
    parser.add_argument('--n_token', type=int, default=256, help="the token number of each codebook")
    parser.add_argument('--n_book', type=int, default=3, help="the number of codebooks")
    parser.add_argument('--vq', action='store_true', default=False, help="if run vq: True or False (default).")
    parser.add_argument('--train_vq', action='store_true', default=False,
                        help="if training the VQ checkpoint: True or False (default).")
    parser.add_argument('--vq_model', type=str, default='MQ', help="available indexing model: [RQ, MQ]")

    # --------------- CoDT --------------------
    parser.add_argument('--gamma', type=float, default=0.5,  # 默认值：0.5
                        help="Gamma parameter for Inverse Propensity Scoring (IPS) in VQ")
    parser.add_argument('--fair_lambda', type=float, default=0.1,
                        help="Penalty weight for Popularity in Fairness-aware Retrieval")
    parser.add_argument('--co_evolve_weight', type=float, default=0.1,  # 默认值0.1
                        help="Weight for Codebook-T5 Co-evolution alignment loss")
    parser.add_argument('--no_ips', action='store_true', default=False,
                        help="Ablation: Disable IPS causal debiasing in VQ (set pop_weights to None)")
    parser.add_argument('--freeze_codebook', action='store_true', default=False,
                        help="Ablation: Freeze codebook, disable co-evolution updates")
    parser.add_argument('--no_align', action='store_true', default=False,
                        help="Ablation: Disable semantic-graph alignment loss")

    # --------------- general --------------------
    parser.add_argument('--cuda', type=int, default=0, help="the used cuda")
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help="available datasets: [LastFM, ML1M, yelp]")

    return parser.parse_args()
