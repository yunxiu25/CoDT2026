import torch
import pandas as pd
import numpy as np
import model
import train
import dataset
import utils
from torch.utils.data import DataLoader

def learning(args):
    lgn_dim = 64
    codebook_dim = 512
    device = torch.device("cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu")
    data_name = args.dataset
    lgn_name = 'lgn-' + data_name + '-' + str(lgn_dim)
    vq_name = 'MQ-' + lgn_name
    print('Process: VQ is working:', vq_name)

    LightGCN = torch.load('../src/lgn/' + lgn_name + '.pth.tar')
    user_emb = LightGCN['embedding_user.weight']
    item_emb = LightGCN['embedding_item.weight']

    pop_weights = utils.get_popularity(data_name, item_emb.shape[0]).to(device)

    if args.vq_model == 'RQ':
        user_vq = model.ResidualVQVAE(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                                      m_book=args.n_book)
        item_vq = model.ResidualVQVAE(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token,
                                      m_book=args.n_book)
    elif args.vq_model == 'MQ':
        user_vq = model.DQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.DQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)

    item_vq_name = 'item-' + vq_name
    if args.train_vq:
        current_pop = None if args.no_ips else pop_weights
        train.vqvae(item_vq, item_vq_name, device, item_emb, n_embedding=args.n_token, m_book=args.n_book,
                    pop_weights=current_pop, gamma=args.gamma)
    item_vq.load_state_dict(torch.load('../checkpoints/vq/' + item_vq_name + '.pth'))
    item_vq.to(device)

    user_vq_name = 'user-' + vq_name
    if args.train_vq:
        train.vqvae(user_vq, user_vq_name, device, user_emb, n_embedding=args.n_token, m_book=args.n_book,
                    pop_weights=None)
    user_vq.load_state_dict(torch.load('../checkpoints/vq/' + user_vq_name + '.pth'))
    user_vq.to(device)

    for phase in ['train', 'test']:
        file_path = f'../data/{data_name}/{phase}.txt'
        with open(file_path, 'r') as f:
            data = f.readlines()

        item_vq.eval()
        user_vq.eval()
        rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
        rec_loader = DataLoader(rec_dataset, batch_size=256, shuffle=False)

        item_list, user_list = [], []
        for i, sample in enumerate(rec_loader):
            _, user_cb_id, item_cb_id = sample
            user_list += user_cb_id
            item_list += item_cb_id

        df = pd.DataFrame({'user_cb_id': user_list, "item_cb_id": item_list})
        df.to_csv(f'../data/{data_name}/{phase}_codebook.txt')
