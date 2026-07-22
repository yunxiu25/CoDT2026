import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans
from transformers import AutoTokenizer, T5Model
import pandas as pd
import model
import utils
import dataset
import myevaluate
from tqdm import tqdm


def vqvae(model, model_name, device, co_emb, n_embedding, m_book=3, kmean_epoch=50, valid_ratio=0.2, batch_size=512,
          lr=1e-3, n_epochs=1000, pop_weights=None, gamma=0.5):
    idx_all = torch.arange(co_emb.shape[0])
    rand_vals = torch.rand(co_emb.shape[0])
    sh = torch.quantile(rand_vals, valid_ratio)
    valid_mask = rand_vals <= sh
    train_mask = rand_vals > sh

    train_dataset = dataset.VQDataset(idx_all[train_mask], co_emb[train_mask])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = dataset.VQDataset(idx_all[valid_mask], co_emb[valid_mask])
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mse_loss = nn.MSELoss(reduction='none')

    tic = time.time()
    valid_loss_record = float('inf')
    for e in range(n_epochs):
        total_loss = 0
        model.train()
        for i, (idx, x) in enumerate(train_dataloader):
            x = x.to(device)
            if pop_weights is not None:
                w = 1.0 / ((pop_weights[idx] + 1.0) ** gamma)
                w = w.to(device).unsqueeze(1)
            else:
                w = 1.0

            if e % kmean_epoch == 0 and i == 0:
                with torch.no_grad():
                    _, res_dict, _ = model(x)
                    for m in range(m_book):
                        try:
                            _, centers = kmeans(res_dict[m], n_embedding, distance='euclidean', device=device,
                                                iter_limit=30)
                            if not torch.isnan(centers).any():
                                model.codebooks[m].weight.data = centers.to(device)
                        except Exception as ex:
                            pass

            x_hat, res_dict, ce_dict = model(x)

            l_reconstruct = (mse_loss(x_hat, x) * w).mean()
            loss = l_reconstruct
            for m in range(m_book):
                l_embedding = (mse_loss(res_dict[m].detach(), ce_dict[m]) * w).mean()
                loss += l_embedding

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item() * x.shape[0]

        model.eval()
        epoch_loss = 0
        for idx, x in valid_dataloader:
            x = x.to(device)
            x_hat, res_dict, ce_dict = model.valid(x)
            loss = mse_loss(x_hat, x).mean()
            epoch_loss += loss.item() * x.shape[0]

        if epoch_loss < valid_loss_record:
            valid_loss_record = epoch_loss
            torch.save(model.state_dict(), '../checkpoints/vq/' + model_name + '.pth')

        if e % 10 == 0:
            toc = time.time()
            print(f'VQ: epoch {e} valid_loss: {valid_loss_record:.4f} elapsed {(toc - tic):.2f}s')


def backbone(data_name, train_rec_loader, valid_rec_loader, user_emb, item_emb, item_num, args, device):
    linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=args.target_length)
    if args.train_from_checkpoint:
        t5 = T5Model.from_pretrained('../checkpoints/backbone/' + data_name)
        tokenizer = AutoTokenizer.from_pretrained("../checkpoints/backbone/" + data_name, legacy=False)
        linear_projection.load_state_dict(torch.load('../checkpoints/backbone/' + data_name + '/projection.pt'))
    else:
        tokenizer = AutoTokenizer.from_pretrained("../src/t5-small", legacy=False, local_files_only=True)
        t5 = T5Model.from_pretrained("../src/t5-small", local_files_only=True)
        add_tokens = utils.codebook_tokens(args.n_book, args.n_token)
        tokenizer.add_tokens(add_tokens)
        t5.resize_token_embeddings(len(tokenizer))

    t5.to(device)
    linear_projection.to(device)

    item_vq = model.DQ(input_dim=item_emb.shape[1], dim=512, n_embedding=args.n_token, m_book=args.n_book)
    if args.train_from_checkpoint:
        try:
            item_vq.load_state_dict(torch.load('../checkpoints/vq/co-evolved-item-MQ-' + data_name + '.pth'))
        except FileNotFoundError:
            item_vq.load_state_dict(torch.load('../checkpoints/vq/item-MQ-lgn-' + data_name + '-64.pth'))
    else:
        item_vq.load_state_dict(torch.load('../checkpoints/vq/item-MQ-lgn-' + data_name + '-64.pth'))

    item_vq.to(device)
    item_vq.train()

    grouped_params = utils.group_model_params(t5, linear_projection, decay=args.decay)

    if not args.freeze_codebook:
        grouped_params.append({"params": item_vq.parameters(), "weight_decay": args.decay})
    else:
        for param in item_vq.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    loss_func = torch.nn.CosineEmbeddingLoss()

    global_metric = 0

    raw_token_ids = tokenizer.convert_tokens_to_ids(
        utils.codebook_tokens(args.n_book, args.n_token)[:args.n_book * args.n_token])
    item_token_ids = torch.tensor(raw_token_ids, dtype=torch.long, device=device)

    for epoch in range(args.epochs):
        t5.train()
        linear_projection.train()
        for i, sample in enumerate(tqdm(train_rec_loader, desc=f"Training Epoch {epoch}")):
            _, train_target_id, _, user_cb_id, train_item_cb_id, train_target_cb_id, _, _ = sample
            input_sentences = utils.prompt(user_cb_id, train_item_cb_id)
            if not input_sentences: continue

            targets = utils.get_target_emb(item_emb, train_target_id.to(device))

            input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length,
                                       padding="max_length", truncation=True)
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            decoder_input_encoding = tokenizer([args.decoder_prepend] * len(list(train_target_cb_id)),
                                               return_tensors="pt",
                                               max_length=args.target_length, padding="max_length", truncation=True)
            decoder_input_ids = t5._shift_right(decoder_input_encoding.input_ids)

            outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                         decoder_input_ids=decoder_input_ids.to(device))
            predicts = linear_projection(outputs.last_hidden_state)

            current_batch = predicts.shape[0]
            neg_idx = torch.randint(0, item_num, (current_batch,), device=device)
            neg_sample = item_emb[neg_idx, :]

            predicts = torch.cat((predicts, predicts), 0)
            samples = torch.cat((targets, neg_sample), 0)
            labels = torch.ones([2 * current_batch], device=device)
            labels[current_batch:] = -1

            rec_loss = loss_func(predicts, samples, labels)

            if args.no_align:
                loss = rec_loss
            else:
                t5_embeds = t5.get_input_embeddings().weight[item_token_ids]
                vq_embeds = torch.cat([item_vq.codebooks[m].weight for m in range(args.n_book)], dim=0)
                align_loss = 1.0 - F.cosine_similarity(t5_embeds, vq_embeds, dim=-1).mean()
                loss = rec_loss + args.co_evolve_weight * align_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0:
            metrics = torch.zeros([2]).to(device)
            n_batch = 0
            t5.eval()
            linear_projection.eval()
            for sample in tqdm(valid_rec_loader, desc=f"Validating Epoch {epoch}"):
                _, item_id, target_id, user_cb_id, item_cb_id, target_cb_id = sample
                input_sentences = utils.prompt(user_cb_id, item_cb_id)
                input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length,
                                           padding="max_length", truncation=True)
                decoder_input_encoding = tokenizer([args.decoder_prepend] * len(list(target_cb_id)),
                                                   return_tensors="pt",
                                                   max_length=args.target_length, padding="max_length", truncation=True)
                decoder_input_ids = t5._shift_right(decoder_input_encoding.input_ids)

                with torch.no_grad():
                    outputs = t5(input_ids=input_encoding.input_ids.to(device),
                                 attention_mask=input_encoding.attention_mask.to(device),
                                 decoder_input_ids=decoder_input_ids.to(device))
                    predicts = linear_projection(outputs.last_hidden_state)
                    scores = utils.similarity_score(predicts, item_emb, item_id)
                    results = torch.argsort(scores, dim=1, descending=True)

                metr, batch = myevaluate.get_metrics(target_id.to(device), results, device, args.k)
                metrics += metr
                n_batch += batch

            val_metric = torch.mean(metrics / n_batch)

            print(
                f'\n[Epoch {epoch}] valid_hit@{args.k} = {metrics[0].item() / n_batch:.4f}, valid_ndcg@{args.k} = {metrics[1].item() / n_batch:.4f}')

            if val_metric > global_metric:
                global_metric = val_metric
                t5.save_pretrained('../checkpoints/backbone/' + data_name)
                tokenizer.save_pretrained("../checkpoints/backbone/" + data_name)
                torch.save(linear_projection.state_dict(), '../checkpoints/backbone/' + data_name + '/projection.pt')
                torch.save(item_vq.state_dict(), '../checkpoints/vq/co-evolved-item-MQ-' + data_name + '.pth')
                print(f"--> Saved optimal model & co-evolved Tokenizer at Epoch {epoch}\n")
