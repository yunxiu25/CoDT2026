import os
import json
from collections import Counter
import numpy as np
import torch

import model
import utils
import kmeans_pytorch as _kmp


def _torch_kmeans(X, num_clusters, distance="euclidean", **kw):
    device = kw.get("device", X.device)
    X = X.to(device).float()
    n = X.shape[0]
    first = torch.randint(n, (1,), device=device)
    centers = [X[first]]
    closest = ((X - centers[0]) ** 2).sum(1)
    for _ in range(num_clusters - 1):
        probs = (closest + 1e-12) / (closest + 1e-12).sum()
        idx = torch.multinomial(probs, 1)
        centers.append(X[idx])
        closest = torch.minimum(closest, ((X - centers[-1]) ** 2).sum(1))
    C = torch.cat(centers, dim=0)
    for _ in range(30):
        d = torch.cdist(X, C)
        assign = d.argmin(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(num_clusters, device=device)
        newC.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(n, device=device))
        mask = cnt > 0
        newC[mask] = newC[mask] / cnt[mask].unsqueeze(1)
        shift = (newC[mask] - C[mask]).norm(dim=1).max()
        C = newC
        if not torch.isfinite(shift) or shift < 1e-4:
            break
    d = torch.cdist(X, C)
    assign = d.argmin(1)
    return assign, C


_kmp.kmeans = _torch_kmeans
import train  # noqa: E402

CKPT_DIR = os.environ.get("CODT_CKPT_DIR", "../checkpoints/vq")
TOKENREC_DIR = os.environ.get("TOKENREC_CKPT_DIR", "../checkpoints/tokenrec_official")
DATASETS = ["ML1M", "LastFM"]
K_BOOKS = 3
N_TOKEN = 256
SEEDS = [42, 3407, 2026]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def level_metrics(tokens, n_token=N_TOKEN):
    """tokens: [n_items, K] -> per-level dict."""
    out = []
    for k in range(tokens.shape[1]):
        counts = np.bincount(tokens[:, k], minlength=n_token).astype(np.float64)
        used = int((counts > 0).sum())
        p = counts[counts > 0] / counts.sum()
        entropy_bits = float(-(p * np.log2(p)).sum())
        perplexity = float(2 ** entropy_bits)
        out.append({
            "used_codes": used,
            "unused_codes": n_token - used,
            "unused_ratio": 1.0 - used / n_token,
            "entropy_bits": entropy_bits,
            "perplexity": perplexity,
            "max_code_share": float(counts.max() / counts.sum()),
        })
    return out


def sequence_metrics(tokens, item_to_group):
    """Full-sequence collision analysis, overall + per popularity group."""
    seqs = [tuple(t) for t in tokens]
    cnt = Counter(seqs)
    n = len(seqs)
    colliding = sum(1 for s in seqs if cnt[s] > 1)
    res = {
        "n_items": n,
        "n_distinct_sequences": len(cnt),
        "n_colliding_items": colliding,
        "collision_rate": colliding / n,
    }
    for gname, gid in [("head", 0), ("torso", 1), ("tail", 2)]:
        idx = [i for i in range(n) if item_to_group[i] == gid]
        g_n = len(idx)
        g_coll = sum(1 for i in idx if cnt[seqs[i]] > 1)
        res[gname] = {
            "n_items": g_n,
            "n_colliding_items": g_coll,
            "n_distinct_sequences": len({seqs[i] for i in idx}),
            "collision_rate": (g_coll / g_n) if g_n else 0.0,
        }
    return res


def build_group_map(pop, n_items):
    order = np.argsort(-pop)
    h = int(n_items * 0.20)
    t = h + int(n_items * 0.30)
    head_set, torso_set = set(order[:h]), set(order[h:t])
    item_to_group = {
        i: (0 if i in head_set else (1 if i in torso_set else 2))
        for i in range(n_items)
    }
    return item_to_group, {"head": h, "torso": t - h, "tail": n_items - t}


def encode_items(vq_ckpt_path, item_emb):
    mq = model.MQ(input_dim=item_emb.shape[1], dim=512, n_embedding=N_TOKEN, m_book=K_BOOKS)
    sd = torch.load(vq_ckpt_path, map_location="cpu")
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("codebook."):
            new_sd[k.replace("codebook.", "codebooks.")] = v
        elif k.startswith("encoder."):
            suffix = k[len("encoder."):]
            for i in range(K_BOOKS):
                new_sd[f"encoders.{i}.{suffix}"] = v
        else:
            new_sd[k] = v
    mq.load_state_dict(new_sd)
    mq.to(device).eval()
    with torch.no_grad():
        toks = mq.encode(item_emb.to(device)).cpu().numpy()
    return toks


def eval_or_train_variant(variant_tag, data_name, item_emb, pop_weights, seed, default_ckpt):
    """Load seed-specific checkpoint if available; otherwise fall back to default checkpoint or train."""
    seed_ckpt = os.path.join(CKPT_DIR, f"{variant_tag}-{data_name}-s{seed}.pth")
    if os.path.exists(seed_ckpt):
        return encode_items(seed_ckpt, item_emb)
    if default_ckpt and os.path.exists(default_ckpt):
        return encode_items(default_ckpt, item_emb)
    utils.set_seed(seed)
    mq = model.MQ(input_dim=item_emb.shape[1], dim=512, n_embedding=N_TOKEN, m_book=K_BOOKS)
    name = f"{variant_tag}-{data_name}-s{seed}"
    train.vqvae(mq, name, device, item_emb.cpu(), n_embedding=N_TOKEN, m_book=K_BOOKS,
                pop_weights=pop_weights)
    mq.to(device).eval()
    with torch.no_grad():
        return mq.encode(item_emb.to(device)).cpu().numpy()


def summarize_seeds(seed_records):
    used = [sum(l["used_codes"] for l in r["per_level"]) for r in seed_records]
    ent = [np.mean([l["entropy_bits"] for l in r["per_level"]]) for r in seed_records]
    all_c = [r["sequence"]["collision_rate"] * 100 for r in seed_records]
    head_c = [r["sequence"]["head"]["collision_rate"] * 100 for r in seed_records]
    torso_c = [r["sequence"]["torso"]["collision_rate"] * 100 for r in seed_records]
    tail_c = [r["sequence"]["tail"]["collision_rate"] * 100 for r in seed_records]
    return (
        f"used={np.mean(used):.0f}±{np.std(used, ddof=1 if len(used)>1 else 0):.0f}/768 "
        f"ent={np.mean(ent):.2f}±{np.std(ent, ddof=1 if len(ent)>1 else 0):.2f}b "
        f"coll={np.mean(all_c):.1f}±{np.std(all_c, ddof=1 if len(all_c)>1 else 0):.1f}% "
        f"(H {np.mean(head_c):.1f}±{np.std(head_c, ddof=1 if len(head_c)>1 else 0):.1f}% "
        f"T {np.mean(torso_c):.1f}±{np.std(torso_c, ddof=1 if len(torso_c)>1 else 0):.1f}% "
        f"L {np.mean(tail_c):.1f}±{np.std(tail_c, ddof=1 if len(tail_c)>1 else 0):.1f}%)"
    )


def main():
    results = {}
    for ds in DATASETS:
        lgn = torch.load(f"../src/lgn/lgn-{ds}-64.pth.tar", map_location="cpu")
        item_emb = lgn["embedding_item.weight"].float()
        n_items = item_emb.shape[0]

        pop_tensor = utils.get_popularity(ds, n_items)
        item_to_group, group_counts = build_group_map(pop_tensor.numpy(), n_items)
        pop_weights = 1.0 / torch.pow(pop_tensor.float() + 1.0, 0.5)

        ds_res = {"n_items": n_items, "groups": group_counts}

        # 1. Official TokenRec static checkpoint (single reference)
        tr_ckpt = os.path.join(TOKENREC_DIR, f"item-MQ-lgn-{ds}-64.pth")
        toks_tr = encode_items(tr_ckpt, item_emb)
        ds_res["TokenRec(official)"] = {
            "per_level": level_metrics(toks_tr),
            "sequence": sequence_metrics(toks_tr, item_to_group),
        }

        # 2. Multi-seed CoDT recipe variants (w/o Re-weight, Stage-1, CoDT-Full)
        recipe_configs = [
            ("w/o-Re-weight", "noRW-item-MQ", None, None),
            ("CoDT-Stage1", "stage1-item-MQ", pop_weights, os.path.join(CKPT_DIR, f"item-MQ-lgn-{ds}-64.pth")),
            ("CoDT-Full", "co-evolved-item-MQ", pop_weights, os.path.join(CKPT_DIR, f"co-evolved-item-MQ-{ds}.pth")),
        ]
        for label, tag, pw, fallback_ckpt in recipe_configs:
            seed_list = []
            for seed in SEEDS:
                toks = eval_or_train_variant(tag, ds, item_emb, pw, seed, fallback_ckpt)
                seed_list.append({
                    "seed": seed,
                    "per_level": level_metrics(toks),
                    "sequence": sequence_metrics(toks, item_to_group),
                })
            ds_res[label] = seed_list

        results[ds] = ds_res

    with open("codebook_util_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> codebook_util_results.json")

    for ds in DATASETS:
        print(f"\n===== {ds} =====")
        for label in ["w/o-Re-weight", "CoDT-Stage1", "CoDT-Full"]:
            print(f"{label:20s} {summarize_seeds(results[ds][label])}")
        tr = results[ds]["TokenRec(official)"]
        lv, sq = tr["per_level"], tr["sequence"]
        print(
            f"{'TokenRec(official)':20s} used={sum(l['used_codes'] for l in lv)}/768 "
            f"ent={np.mean([l['entropy_bits'] for l in lv]):.2f}b "
            f"coll={sq['collision_rate']*100:.1f}% "
            f"(H {sq['head']['collision_rate']*100:.1f}% "
            f"T {sq['torso']['collision_rate']*100:.1f}% "
            f"L {sq['tail']['collision_rate']*100:.1f}%)"
        )
