"""The activation analysis report (HTML) from pi05_activations.py's acts_<name>.npz files.

1. Domain separability, per layer and stream (env / wrist / overhead camera tokens, text + state tokens, the action
   expert's tokens): a linear classifier's accuracy telling real from sim, cross-validated over episodes / sessions
   (so it can't memorize episodes). ~100%: real and sim are represented apart; ~50%: in the same way.
2. Cross-domain neighbours: for each held-out real frame (the RLHF decisions), the share of its 10 nearest neighbours
   (among the dataset's real and sim frames) that are sim, over the sim share of that pool: 1 = its neighbourhood is
   as sim as the pool, 0 = never a sim neighbour. Per layer and stream.
3. Maps: t-SNE of the action expert's tokens (first, middle, last layer), by source; the sim frames by start kind.
4. Behaviour on the voted RLHF decisions: each checkpoint's predicted chunk (the executed steps) against the executed
   base path, its palm position [mm], for the voted-up and the voted-down decisions. The base paths were drawn from
   the co-train checkpoints (base_note) -- they sit closer to those checkpoints' predictions for both votes; the
   margin (down - up distance) is the fairer comparison.

    /usr/bin/python3 report.py --acts real_only_27999 cotrain_27999 balanced_27999 --out /media/dexnex_ssd/data/activation_analysis/report.html
"""
import argparse
import base64
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DIR = "/media/dexnex_ssd/data/activation_analysis"
STREAMS = ("env", "wrist", "overhead", "text", "action")
SIM = ("aiet_grip", "aiet_grip_lift")
COLORS = {"rlhf": "#d62728", "real": "#1f77b4", "aiet_grip": "#2ca02c", "aiet_grip_lift": "#9467bd"}


def fig_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="max-width:100%">'


def reduce(x: np.ndarray, n: int = 128) -> np.ndarray:
    x = StandardScaler().fit_transform(x.astype(np.float32))
    return PCA(min(n, x.shape[0] - 1, x.shape[1]), random_state=0).fit_transform(x)


def separability(z, layer_stream, groups, is_sim, mask) -> float:
    x, y, g = reduce(layer_stream[mask]), is_sim[mask], groups[mask]
    acc = []
    for tr, te in GroupKFold(5).split(x, y, g):
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(x[tr], y[tr])
        p = clf.predict(x[te])
        acc.append(0.5 * ((p[y[te]] == 1).mean() + (p[~y[te]] == 0).mean()))  # (balanced accuracy)
    return float(np.mean(acc))


def neighbours(layer_stream, source, k=10) -> float:
    x = reduce(layer_stream)
    pool = np.isin(source, ("real",) + SIM)
    query = source == "rlhf"
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(x[pool])
    _, idx = nn.kneighbors(x[query])
    sim_pool = np.isin(source[pool], SIM)
    return float(sim_pool[idx].mean() / sim_pool.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", nargs="+", required=True, help="acts_<name>.npz names, e.g. real_only_27999 cotrain_27999")
    p.add_argument("--out", default=f"{DIR}/report.html")
    args = p.parse_args()

    data = {n: dict(np.load(f"{DIR}/acts_{n}.npz", allow_pickle=True)) for n in args.acts}
    first = data[args.acts[0]]
    source = first["source"].astype(str)
    groups = np.where(source == "rlhf", np.char.add("s", first["session"].astype(str)),
                      np.char.add("e", first["episode"].astype(str)))
    is_sim = np.isin(source, SIM)
    n_layers = first["action"].shape[1]
    counts = {s: int((source == s).sum()) for s in ("rlhf", "real") + SIM}
    html = [f"<html><head><meta charset='utf-8'><title>pi0.5 real vs sim activations</title><style>body{{font-family:sans-serif;"
            f"max-width:1300px;margin:auto;padding:16px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>",
            "<h1>pi0.5: are real and sim observations represented alike?</h1>",
            f"<p>Observations: {counts} -- <b>rlhf</b> = held-out real (RLHF decisions, no checkpoint trained on them), <b>real</b> = the "
            "teleop demos, <b>aiet_grip / aiet_grip_lift</b> = sim. Checkpoints: " + ", ".join(f"<b>{n}</b> ({data[n]['config']})" for n in args.acts) + "</p>"]

    # 1 + 2: per layer, per stream, per checkpoint
    sep = {n: {s: [] for s in STREAMS} for n in args.acts}
    nbr = {n: {s: [] for s in STREAMS} for n in args.acts}
    for n in args.acts:
        for s in STREAMS:
            for layer in range(n_layers):
                feats = data[n][s][:, layer].astype(np.float32)
                sep[n][s].append(separability(None, feats, groups, is_sim, source != "rlhf"))  # (the dataset's real vs sim)
                nbr[n][s].append(neighbours(feats, source))
            print(f"{n} {s}: separability {np.round(sep[n][s], 2)} | rlhf sim-neighbour ratio {np.round(nbr[n][s], 2)}", flush=True)
    for title, values, ylabel, ref in (("1. Domain separability: a linear classifier's balanced accuracy, real vs sim (the dataset's frames)", sep,
                                        "balanced accuracy", 0.5),
                                       ("2. Held-out real frames' sim neighbours (10 nearest, over the pool's sim share)", nbr, "sim neighbour ratio", 1.0)):
        fig, axes = plt.subplots(1, len(STREAMS), figsize=(4 * len(STREAMS), 3.4), sharey=True)
        for ax, s in zip(axes, STREAMS):
            for n in args.acts:
                ax.plot(range(n_layers), values[n][s], marker="o", ms=3, label=n)
            ax.axhline(ref, color="grey", ls="--", lw=1)
            ax.set_title(s)
            ax.set_xlabel("layer")
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        html += [f"<h2>{title}</h2>", fig_html(fig)]

    # 3: t-SNE maps of the action stream
    html.append("<h2>3. t-SNE of the action expert's tokens</h2>")
    for n in args.acts:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, layer in zip(axes, (0, n_layers // 2, n_layers - 1)):
            xy = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(reduce(data[n]["action"][:, layer], 50))
            for s, c in COLORS.items():
                m = source == s
                ax.scatter(xy[m, 0], xy[m, 1], s=6, c=c, label=f"{s} ({m.sum()})", alpha=0.6)
            ax.set_title(f"{n}: layer {layer}")
            ax.set_xticks([]), ax.set_yticks([])
        axes[0].legend(fontsize=8, markerscale=2)
        html.append(fig_html(fig))

    # 4: behaviour on the voted decisions
    html.append("<h2>4. Behaviour on the voted RLHF decisions: predicted chunk vs the executed base path</h2>")
    labels = first["label"].astype(str)[source == "rlhf"]
    notes = first["base_note"].astype(str)[source == "rlhf"]
    rows = []
    rng = np.random.default_rng(0)
    for n in args.acts:
        pred, base = data[n]["rlhf_pred"], data[n]["rlhf_base"]
        d = np.linalg.norm(pred[:, :, :3] - base[:, :, :3], axis=2).mean(1) * 1000  # [mm] palm position, over the executed steps
        up, down = d[labels == "up"], d[labels == "down"]
        boot = [rng.choice(down, len(down)).mean() - rng.choice(up, len(up)).mean() for _ in range(2000)]
        rows.append(f"<tr><td>{n}</td><td>{up.mean():.1f}</td><td>{down.mean():.1f}</td><td><b>{down.mean() - up.mean():+.1f}</b> "
                    f"[{np.percentile(boot, 2.5):+.1f}, {np.percentile(boot, 97.5):+.1f}]</td></tr>")
    note_counts = {k: int(v) for k, v in zip(*np.unique(notes, return_counts=True))}
    html += ["<table><tr><th>checkpoint</th><th>to voted-up base [mm]</th><th>to voted-down base [mm]</th><th>margin (down - up) [95% CI]</th></tr>",
             *rows, "</table>",
             f"<p>{int((labels == 'up').sum())} up / {int((labels == 'down').sum())} down decisions. Positive margin: the checkpoint's own chunk sits "
             "closer to what you voted up than to what you voted down. Which checkpoint drew the bases (base_note): " + str(note_counts) + "</p>",
             "</body></html>"]
    open(args.out, "w").write("\n".join(html))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
