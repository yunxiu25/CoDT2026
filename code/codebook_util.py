# -*- coding: utf-8 -*-
"""
Quantitative codebook-utilization analysis (Reviewer 2, Major Comment 1; Reviewer 3, W3).

Metrics per codebook level k in {1,2,3}:
  - #distinct used codes / L, proportion of unused codes
  - assignment entropy (bits) and perplexity over code usage
Whole-sequence metrics:
  - #distinct token sequences, #items whose full (c1,c2,c3) sequence collides
  - collision rate overall and per popularity group (Head 20% / Torso 30% / Tail 50%)

Variants compared (per dataset):
  - TokenRec (static, unweighted VQ)      : ../TokenRec-main-oral checkpoints
  - CoDT Stage-1  (re-weighted VQ only)   : item-MQ-lgn-*-64.pth (this repo, trained w/ pop weights)
  - CoDT Full     (+ co-evolution)        : co-evolved-item-MQ-*.pth
  - w/o Re-weighting (unweighted VQ, this repo training recipe), seeds [42, 3407, 2026]

Outputs: codebook_util_results.json + console tables.
"""
import json
import sys
import numpy as np
import torch

import model
import utils

# compat shim: newer kmeans_pytorch has no `iter_limit` and can loop forever on
# NaN center-shift; replace with a torch-native k-means capped at 30 iterations
# (same semantics as the original recipe's iter_limit=30).
import kmeans_pytorch as _kmp


def _torch_kmeans(X, num_clusters, distance="euclidean", **kw):
    device = kw.get("device", X.device)
    X = X.to(device).float()
    n = X.shape[0]
    # kmeans++ style init
    first = torch.randint(n, (1,), device=device)
    centers = [X[first]]
    closest = ((X - centers[0]) ** 2).sum(1)
    for _ in range(num_clusters - 1):
        probs = (closest + 1e-12) / (closest + 1e-12).sum()
        idx = torch.multinomial(probs, 1)
        centers.append(X[idx])
        closest = torch.minimum(closest, ((X - centers[-1]) ** 2).sum(1))
    C = torch.cat(centers, dim=0)
    for _ in range(30):  # iter_limit=30
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

ORAL = r"..\..\TokenRec-main-oral\TokenRec-main"
DATASETS = ["ML1M", "LastFM"]
K_BOOKS = 3
N_TOKEN = 256
NO_RW_SEEDS = [42, 3407, 2026]

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
    """Full-sequence collision analysis, overall + per group."""
    seqs = [tuple(t) for t in tokens]
    from collections import Counter
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
            "collision_rate": (g_coll / g_n) if g_n else 0.0,
            "distinct_sequences": len({seqs[i] for i in idx}),
        }
    return res


def encode_items(vq_ckpt_path, item_emb):
    mq = model.MQ(input_dim=item_emb.shape[1], dim=512, n_embedding=N_TOKEN, m_book=K_BOOKS)
    sd = torch.load(vq_ckpt_path, map_location="cpu")
    # official TokenRec release: single shared encoder + singular codebook names
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


def train_no_rw(data_name, item_emb, seed):
    """Train an unweighted-VQ (w/o re-weighting) variant and return its token assignments."""
    utils.set_seed(seed)
    mq = model.MQ(input_dim=item_emb.shape[1], dim=512, n_embedding=N_TOKEN, m_book=K_BOOKS)
    name = f"noRW-item-MQ-{data_name}-s{seed}"
    train.vqvae(mq, name, device, item_emb.cpu(), n_embedding=N_TOKEN, m_book=K_BOOKS,
                pop_weights=None)
    mq.to(device).eval()
    with torch.no_grad():
        toks = mq.encode(item_emb.to(device)).cpu().numpy()
    return toks


def main():
    results = {}
    for ds in DATASETS:
        lgn = torch.load(f"../src/lgn/lgn-{ds}-64.pth.tar", map_location="cpu")
        item_emb = lgn["embedding_item.weight"].float()
        n_items = item_emb.shape[0]

        pop = utils.get_popularity(ds, n_items).numpy()
        order = np.argsort(-pop)
        h = int(n_items * 0.20)
        t = h + int(n_items * 0.30)
        item_to_group = {}
        for i in range(n_items):
            item_to_group[i] = 0 if i in set(order[:h]) else (1 if i in set(order[h:t]) else 2)

        variants = {
            "TokenRec(static-unweighted)": rf"{ORAL}\checkpoints\vq\item-MQ-lgn-{ds}-64.pth",
            "CoDT-Stage1(reweighted)": f"../checkpoints/vq/item-MQ-lgn-{ds}-64.pth",
            "CoDT-Full(co-evolved)": f"../checkpoints/vq/co-evolved-item-MQ-{ds}.pth",
        }
        ds_res = {"n_items": n_items, "groups": {"head": h, "torso": t - h, "tail": n_items - t}}

        for name, ck in variants.items():
            toks = encode_items(ck, item_emb)
            ds_res[name] = {
                "per_level": level_metrics(toks),
                "sequence": sequence_metrics(toks, item_to_group),
                "tokens": toks.tolist(),
            }
            print(f"[{ds}] {name}: "
                  f"used L1/L2/L3 = {[level_metrics(toks)[i]['used_codes'] for i in range(3)]}")

        # cross-check against the npy used for Fig.5 (CoDT coords)
        try:
            ref = np.load(f"codt_coords_{ds}.npy")
            full = np.array(ds_res["CoDT-Full(co-evolved)"]["tokens"])
            match = float((ref == full).all(axis=1).mean())
            ds_res["fig5_npy_match_ratio"] = match
            print(f"[{ds}] Fig.5 npy match ratio with co-evolved ckpt: {match:.4f}")
        except Exception as e:
            ds_res["fig5_npy_match_ratio"] = str(e)

        results[ds] = ds_res

    # w/o re-weighting variants across seeds (trained fresh; unweighted objective)
    for ds in DATASETS:
        lgn = torch.load(f"../src/lgn/lgn-{ds}-64.pth.tar", map_location="cpu")
        item_emb = lgn["embedding_item.weight"].float()
        n_items = item_emb.shape[0]
        pop = utils.get_popularity(ds, n_items).numpy()
        order = np.argsort(-pop)
        h = int(n_items * 0.20)
        t = h + int(n_items * 0.30)
        item_to_group = {}
        for i in range(n_items):
            item_to_group[i] = 0 if i in set(order[:h]) else (1 if i in set(order[h:t]) else 2)

        seed_res = []
        for seed in NO_RW_SEEDS:
            print(f"[{ds}] training w/o re-weighting VQ, seed={seed} ...")
            toks = train_no_rw(ds, item_emb, seed)
            seed_res.append({
                "seed": seed,
                "per_level": level_metrics(toks),
                "sequence": sequence_metrics(toks, item_to_group),
            })
        results[ds]["w/o-Re-weighting(seeds)"] = seed_res

    # strip raw token lists from json to keep it readable
    slim = json.loads(json.dumps(results).replace('"tokens":', '"_tokens":'))
    for ds in DATASETS:
        for k in list(slim[ds].keys()):
            if isinstance(slim[ds][k], dict) and "_tokens" in slim[ds][k]:
                del slim[ds][k]["_tokens"]
    with open("codebook_util_results.json", "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)
    print("\nSaved -> codebook_util_results.json")

    # console summary
    for ds in DATASETS:
        print(f"\n===== {ds} =====")
        for name in ["TokenRec(static-unweighted)", "CoDT-Stage1(reweighted)",
                     "CoDT-Full(co-evolved)"]:
            r = results[ds][name]
            lv = r["per_level"]
            sq = r["sequence"]
            print(f"{name:32s} used={sum(l['used_codes'] for l in lv)}/768 "
                  f"entropy={np.mean([l['entropy_bits'] for l in lv]):.2f}b "
                  f"ppl={np.mean([l['perplexity'] for l in lv]):.1f} "
                  f"coll={sq['collision_rate']*100:.1f}% "
                  f"(head {sq['head']['collision_rate']*100:.1f}% "
                  f"torso {sq['torso']['collision_rate']*100:.1f}% "
                  f"tail {sq['tail']['collision_rate']*100:.1f}%)")
        seeds = results[ds].get("w/o-Re-weighting(seeds)", [])
        if seeds:
            coll = [s["sequence"]["collision_rate"] for s in seeds]
            used = [sum(l["used_codes"] for l in s["per_level"]) for s in seeds]
            ent = [np.mean([l["entropy_bits"] for l in s["per_level"]]) for s in seeds]
            tailc = [s["sequence"]["tail"]["collision_rate"] for s in seeds]
            print(f"w/o-Re-weighting (3 seeds)      used={np.mean(used):.1f}±{np.std(used):.1f} "
                  f"entropy={np.mean(ent):.2f}±{np.std(ent):.2f} "
                  f"coll={np.mean(coll)*100:.1f}%±{np.std(coll)*100:.1f}% "
                  f"tail_coll={np.mean(tailc)*100:.1f}%±{np.std(tailc)*100:.1f}%")


if __name__ == "__main__":
    main()
