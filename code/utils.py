import torch
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import random
import time


def get_popularity(data_name, item_num):
    pop = np.zeros(item_num)
    try:
        with open(f'../data/{data_name}/train.txt', 'r') as f:
            for line in f:
                items = line.strip().split()[1:]
                for item in items:
                    pop[int(item)] += 1
    except:
        pass
    return torch.tensor(pop, dtype=torch.float32)


def reconstruct(model, emb, device):
    model.to(device)
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        emb_hat, _, _ = model(emb)
        loss = mse_loss(emb_hat, emb)
        print('test loss:', loss.item())
    return emb_hat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def item_codebook_to_str(vq_id):
    id_num = vq_id.shape[0]
    codebook_num = vq_id.shape[1]
    sample = []
    for i in range(id_num):
        temp = ['item_']
        for j in range(codebook_num):
            token = "".join(["<", str(j), '-', str(vq_id[i, j].item()), '>'])
            temp.append(token)
        temp = "".join(temp)
        sample.append(temp)
    sample = " ".join(sample)
    return sample


def user_codebook_to_str(vq_id):
    user_head = ['a', 'b', 'c', 'd', 'e']
    id_num = vq_id.shape[0]
    codebook_num = vq_id.shape[1]
    sample = []
    for i in range(id_num):
        temp = ['user_']
        for j in range(codebook_num):
            token = "".join(["<", user_head[j], '-', str(vq_id[i, j].item()), '>'])
            temp.append(token)
        temp = "".join(temp)
        sample.append(temp)
    sample = " ".join(sample)
    return sample


def group_model_params(model1, model2, decay):
    grouped_params = [
        {"params": [p for n, p in model1.named_parameters()], "weight_decay": decay},
        {"params": [p for n, p in model2.named_parameters()], "weight_decay": decay},
    ]
    return grouped_params


def prompt(user_batch, items_batch, is_test=False, is_unseen=False):
    prefix = 'You are an expert at recommending products to users based on their purchase histories. \n'
    if not is_test:
        sentences = [
            prefix + f'Given the following purchase history for the {user}: {items}. Predict the user preferences.' for
            user, items in zip(user_batch, items_batch)]
    else:
        sentences = [
            prefix + f"Given the {user}'s previous interactions with the {items}, what are the user preferences?" for
            user, items in zip(user_batch, items_batch)]
    return sentences


def read_cf_embeddings(model_name, checkpoint_name):
    model = torch.load('../src/' + model_name + '/' + checkpoint_name + '.pth.tar')
    user_emb = model['embedding_user.weight']
    item_emb = model['embedding_item.weight']
    return user_emb, item_emb


def get_target_emb(item_emb, labels):
    return item_emb[labels]


def codebook_tokens(n_book, n_token):
    add_tokens = []
    for i in range(n_book):
        for j in range(n_token):
            add_tokens.append("<" + str(i) + '-' + str(j) + '>')
    user_head = ['a', 'b', 'c', 'd', 'e']
    for i in range(n_book):
        for j in range(n_token):
            add_tokens.append("<" + user_head[i] + '-' + str(j) + '>')
    return add_tokens


def similarity_score(predicts, item_emb, item_id):
    predicts_norm = torch.nn.functional.normalize(predicts, p=2, dim=1)
    item_emb_norm = torch.nn.functional.normalize(item_emb, p=2, dim=1)

    score = torch.matmul(predicts_norm, item_emb_norm.T)

    batch = predicts.shape[0]
    for i in range(batch):
        items = item_id[i].split(" ")
        items = [int(item) for item in items]
        score[i, items] = -1e9
    return score


def MSE_distance(predicts, item_emb):
    score = []
    batch, dim = predicts.shape
    item_num, _ = item_emb.shape
    for i in range(batch):
        temp = predicts[i, :].unsqueeze(0).expand(item_num, -1)
        dis = (temp - item_emb).pow(2).sum(1).sqrt()
        score.append(dis)
    return torch.stack(score, dim=0)


def data_augment(id_list, codebook_id_list, shred=2, item_limit=20):
    num = len(id_list)
    samples, codebook_samples = [], []
    for n in range(num):
        ids = id_list[n].strip('\n').split(" ")
        user_id = ids[0]
        item_id = ids[1:]

        cbs = codebook_id_list[n].strip('\n').split(" ")
        user_codebook_id = cbs[0]
        item_codebook_id = cbs[1:]

        temp_sample, temp_codebook_sample = [user_id], [user_codebook_id]
        for k in range(len(item_id)):
            if k >= item_limit: break
            temp_sample.append(item_id[k])
            temp_codebook_sample.append(item_codebook_id[k])
            if k >= shred:
                samples.append(" ".join(temp_sample))
                codebook_samples.append(" ".join(temp_codebook_sample))
    return samples, codebook_samples


def data_construction(id_list, codebook_id_list, item_limit=100):
    num = len(id_list)
    samples, codebook_samples = [], []
    for n in range(num):
        ids = id_list[n].strip('\n').split(" ")
        user_id = ids[0]
        item_id = ids[1:]

        cbs = codebook_id_list[n].strip('\n').split(" ")
        user_codebook_id = cbs[0]
        item_codebook_id = cbs[1:]

        temp_sample, temp_codebook_sample = [user_id], [user_codebook_id]
        for k in range(len(item_id)):
            if k >= item_limit: break
            temp_sample.append(item_id[k])
            temp_codebook_sample.append(item_codebook_id[k])

        samples.append(" ".join(temp_sample))
        codebook_samples.append(" ".join(temp_codebook_sample))
    return samples, codebook_samples
