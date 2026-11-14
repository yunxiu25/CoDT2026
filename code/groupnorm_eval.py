import argparse
import inspect
import json
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to the repo's code directory")
ap.add_argument("--dataset", default="ML1M")
ap.add_argument("--model", default="seq", choices=["seq", "lightgcn"])
ap.add_argument("--fair_lambda", type=float, default=0.0)
ap.add_argument("--tag", default="")
ap.add_argument("--item_limit", type=int, default=50,
                help="history cap; oral repo uses k>limit so pass 49 to align")
ap.add_argument("--save_lists", action="store_true")
cli = ap.parse_args()

os.chdir(cli.repo)
sys.path.insert(0, os.path.abspath(cli.repo))

import dataset as ds_mod  # noqa: E402
import utils as utils_mod  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITEM_LIMIT = 50
BATCH = 64
SOURCE_LENGTH = 512
TARGET_LENGTH = 20
DECODER_PREPEND = "The interaction history shows that the user preference could be"


def build_test_loader(data_name):
    (
        train_raw, test_raw, train_cb, test_cb, item_num, user_num
    ) = ds_mod.read_data(data_name)
    test_data = [line.strip() for line in test_raw]

    # oral TokenRec's data_construction iterates the DataFrame; this repo takes str lines
    if "oral" in cli.repo.replace("\\", "/").lower():
        cb_arg = test_cb
    else:
        cb_arg = [
            f"{str(row['user_cb_id']).strip()} {str(row['item_cb_id']).strip()}"
            for _, row in test_cb.iterrows()
        ]
    test_proc, cb_proc = utils_mod.data_construction(test_data, cb_arg, item_limit=cli.item_limit)
    loader = torch.utils.data.DataLoader(
        ds_mod.LLM4RecDataset(test_proc, cb_proc, no_shuffle=True),
        batch_size=BATCH, shuffle=False)
    return loader, item_num, test_raw


def popularity_groups(data_name, item_num):
    if hasattr(utils_mod, "get_popularity"):
        pop = utils_mod.get_popularity(data_name, item_num).numpy()
    else:  # official TokenRec release lacks this helper; count locally
        pop = np.zeros(item_num)
        with open(f"../data/{data_name}/train.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    for it in parts[1:]:
                        pop[int(it)] += 1
    order = np.argsort(-pop)
    h = int(item_num * 0.20)
    t = h + int(item_num * 0.30)
    grp = np.zeros(item_num, dtype=int)
    grp[order[:h]] = 0
    grp[order[h:t]] = 1
    grp[order[t:]] = 2
    return grp, pop, (h, t - h, item_num - t)


def summarize(rec_lists, targets, grp, item_num, group_sizes):
    rec = np.array(rec_lists)                      # [n_users, 20]
    tg = np.array(targets)
    g_of = grp
    hits = np.array([tg[u] in rec[u] for u in range(len(tg))])

    targets_per_group = np.bincount(g_of[tg], minlength=3)
    hits_per_group = np.array([hits[g_of[tg] == g].sum() for g in range(3)])
    total_hits = hits.sum()

    exposure = np.bincount(rec.reshape(-1), minlength=item_num)
    exposure_per_item_group = [exposure[g_of == g].sum() / group_sizes[g] for g in range(3)]

    unique_recommended = np.unique(rec)
    unique_per_group = [int((g_of[unique_recommended] == g).sum()) for g in range(3)]

    tail_hit_items = np.unique(tg[(g_of[tg] == 2) & hits])

    return {
        "n_test_users": int(len(tg)),
        "overall_HR@20": float(hits.mean()),
        "targets_per_group": targets_per_group.tolist(),
        "hits_per_group": hits_per_group.tolist(),
        "recall@20_per_group": (hits_per_group / np.maximum(targets_per_group, 1)).tolist(),
        "hit_proportions_old_metric": (hits_per_group / max(total_hits, 1)).tolist(),
        "avg_exposure_per_item_group": exposure_per_item_group,
        "catalog_coverage": float(len(unique_recommended) / item_num),
        "unique_recommended_per_group": unique_per_group,
        "group_sizes": list(group_sizes),
        "n_unique_tail_items_hit": int(len(tail_hit_items)),
        "n_unique_tail_targets": int(targets_per_group[2]),
    }


def run_seq():
    from transformers import AutoTokenizer, T5Model
    import model as model_mod

    loader, item_num, _ = build_test_loader(cli.dataset)
    grp, pop, sizes = popularity_groups(cli.dataset, item_num)

    lgn = torch.load(f"../src/lgn/lgn-{cli.dataset}-64.pth.tar", map_location="cpu")
    item_emb = lgn["embedding_item.weight"].float().to(DEVICE)

    proj = model_mod.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=TARGET_LENGTH)
    proj.load_state_dict(torch.load("../checkpoints/backbone/" + cli.dataset + "/projection.pt", map_location="cpu"))
    t5 = T5Model.from_pretrained("../checkpoints/backbone/" + cli.dataset)
    tok = AutoTokenizer.from_pretrained("../checkpoints/backbone/" + cli.dataset, legacy=False)
    t5.to(DEVICE).eval(); proj.to(DEVICE).eval()

    pop_t = torch.tensor(pop, dtype=torch.float32, device=DEVICE)
    pop_log = torch.log(pop_t + 1.0)
    pop_penalty = pop_log / (pop_log.max() + 1e-8)

    rec_lists, targets, user_ids = [], [], []
    with torch.no_grad():
        for sample in loader:
            _, item_id, target_id, user_cb_id, item_cb_id, target_cb_id = sample
            sents = utils_mod.prompt(user_cb_id, item_cb_id, is_test=True, is_unseen=False)
            enc = tok(sents, return_tensors="pt", max_length=SOURCE_LENGTH,
                      padding="max_length", truncation=True)
            dec = tok([DECODER_PREPEND] * len(target_cb_id), return_tensors="pt",
                      max_length=TARGET_LENGTH, padding="max_length", truncation=True)
            dec_ids = t5._shift_right(dec.input_ids)
            out = t5(input_ids=enc.input_ids.to(DEVICE),
                     attention_mask=enc.attention_mask.to(DEVICE),
                     decoder_input_ids=dec_ids.to(DEVICE))
            pred = proj(out.last_hidden_state)
            scores = utils_mod.similarity_score(pred, item_emb, item_id)
            scores = scores - cli.fair_lambda * pop_penalty.unsqueeze(0)
            top20 = torch.argsort(scores, dim=1, descending=True)[:, :20].cpu().numpy()
            rec_lists.extend(top20.tolist())
            targets.extend(target_id.numpy().tolist())

    res = summarize(rec_lists, targets, grp, item_num, sizes)
    res["model"] = cli.tag or f"seq(fair={cli.fair_lambda})"
    if cli.save_lists:
        np.savez(f"rec_lists_{cli.tag}_{cli.dataset}.npz",
                 rec=np.array(rec_lists), targets=np.array(targets), grp=grp)
    return res


def run_lightgcn():
    _, item_num, test_raw = build_test_loader(cli.dataset)
    grp, pop, sizes = popularity_groups(cli.dataset, item_num)

    import scipy.sparse as sp
    lgn = torch.load(f"../src/lgn/lgn-{cli.dataset}-64.pth.tar", map_location="cpu")
    U = lgn["embedding_user.weight"].float()
    I = lgn["embedding_item.weight"].float()
    nu, ni = U.shape[0], I.shape[0]
    rows, cols = [], []
    for line in open(f"../data/{cli.dataset}/train.txt"):
        p = line.split()
        if len(p) < 2:
            continue
        u = int(p[0])
        for it in p[1:]:
            rows.append(u)
            cols.append(int(it))
    R = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(nu, ni)).tocsr()
    du = np.asarray(R.sum(1)).ravel()
    di = np.asarray(R.sum(0)).ravel()
    du[du == 0] = 1
    di[di == 0] = 1
    An = (sp.diags(1 / np.sqrt(du)) @ R @ sp.diags(1 / np.sqrt(di))).tocoo()
    At = An.T.tocoo()
    AnT = torch.sparse_coo_tensor(np.vstack([An.row, An.col]), An.data.astype(np.float32), An.shape).coalesce()
    AtT = torch.sparse_coo_tensor(np.vstack([At.row, At.col]), At.data.astype(np.float32), At.shape).coalesce()
    E = torch.cat([U, I], 0)
    final = E.clone()
    for _ in range(3):  # 3-layer LightGCN propagation
        eu = torch.sparse.mm(AnT, E[nu:])
        ei = torch.sparse.mm(AtT, E[:nu])
        E = torch.cat([eu, ei], 0)
        final = final + E
    final = final / 4.0
    UF, IF = final[:nu].to(DEVICE), final[nu:].to(DEVICE)

    rec_lists, targets = [], []
    with torch.no_grad():
        S = torch.nn.functional.normalize(UF, dim=1) @ torch.nn.functional.normalize(IF).T
        for parts in (l.split() for l in test_raw):
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            seq = items[:cli.item_limit]      # aligned with data_construction(item_limit)
            target = seq[-1]
            hist = seq[:-1]
            s = S[uid].clone()
            s[hist] = -1e9
            top20 = torch.topk(s, 20).indices.cpu().numpy().tolist()
            rec_lists.append(top20)
            targets.append(target)

    res = summarize(rec_lists, targets, grp, item_num, sizes)
    res["model"] = "LightGCN"
    return res
