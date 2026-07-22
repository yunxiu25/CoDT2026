import torch
from torch.utils.data import DataLoader
import random
import utils
import train
import dataset
import test
import vq
from parse import parse_args

args = parse_args()
data_name = args.dataset
print('CoDT is working on', data_name)

use_cuda = True
device = torch.device("cuda:" + str(args.cuda) if use_cuda and torch.cuda.is_available() else "cpu")

if args.vq:
    vq.learning(args)

lgn_dim = 64
model_name = 'lgn'
checkpoint_name = model_name + '-' + data_name + '-' + str(lgn_dim)
user_emb, item_emb = utils.read_cf_embeddings(model_name, checkpoint_name)

train_data_raw, test_data_raw, train_codebook_df, test_codebook_df, item_num, user_num = dataset.read_data(data_name)
pop_weights = utils.get_popularity(data_name, item_emb.shape[0])

train_data = [line.strip() for line in train_data_raw]
test_data = [line.strip() for line in test_data_raw]

train_codebook_data = [f"{str(row['user_cb_id']).strip()} {str(row['item_cb_id']).strip()}" for _, row in train_codebook_df.iterrows()]
test_codebook_data = [f"{str(row['user_cb_id']).strip()} {str(row['item_cb_id']).strip()}" for _, row in test_codebook_df.iterrows()]

valid_data = train_data.copy()
valid_codebook_data = train_codebook_data.copy()

if not args.no_data_augment:
    train_data_aug, train_codebook_data_aug = utils.data_augment(train_data, train_codebook_data, shred=2,
                                                         item_limit=args.item_limit)
else:
    train_data_aug, train_codebook_data_aug = utils.data_construction(train_data, train_codebook_data,
                                                              item_limit=args.item_limit)

valid_data_proc, valid_codebook_data_proc = utils.data_construction(valid_data, valid_codebook_data, item_limit=args.item_limit)
test_data_proc, test_codebook_data_proc = utils.data_construction(test_data, test_codebook_data, item_limit=args.item_limit)

train_rec_dataset = dataset.LLM4RecTrainDataset(train_data_aug, train_codebook_data_aug, args.no_shuffle)
train_rec_loader = DataLoader(train_rec_dataset, batch_size=args.batch, shuffle=True)

valid_rec_dataset = dataset.LLM4RecDataset(valid_data_proc, valid_codebook_data_proc, args.no_shuffle)
valid_rec_loader = DataLoader(valid_rec_dataset, batch_size=args.batch, shuffle=False)

test_rec_dataset = dataset.LLM4RecDataset(test_data_proc, test_codebook_data_proc, no_shuffle=True)
test_rec_loader = DataLoader(test_rec_dataset, batch_size=args.batch, shuffle=False)

if not args.no_train:
    train.backbone(data_name, train_rec_loader, valid_rec_loader, user_emb.to(device), item_emb.to(device), item_num,
                   args, device)

test.backbone(data_name, test_rec_loader, user_emb.to(device), item_emb.to(device), item_num, args, device, pop_weights)
