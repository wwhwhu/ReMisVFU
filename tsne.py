#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t‑SNE 可视化（原始 / 遗忘 / 重训练）
-----------------------------------
• 无训练；仅加载权重并绘图
• 输出 PDF 位于 ./draw/ 目录
"""
from matplotlib.lines import Line2D
import os, torch, numpy as np, matplotlib
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from utils.data_split import load_and_split_data, remove_from_index
from vfl.core         import SplitNN

def add_trigger_to_right(x_batch: torch.Tensor,
                         y_batch: torch.Tensor,
                         trigger_value: float = 1.0,
                         trigger_size: int = 2,
                         target_label: int = 0,
                         poison_fraction: float = 0.0):
    B, C, H, W = x_batch.shape  # C=1, H=28, W=14
    x_poisoned = x_batch.clone()
    y_poisoned = y_batch.clone()

    num_poison = int(poison_fraction * B)
    poison_mask = torch.zeros(B, dtype=torch.bool, device=x_batch.device)
    if num_poison == 0:
        return x_poisoned, y_poisoned, poison_mask

    idxs = torch.randperm(B)[:num_poison]
    for idx in idxs:
        row_start = H - trigger_size
        col_start = W - trigger_size
        x_poisoned[idx, 0, row_start:H, col_start:W] = trigger_value
        y_poisoned[idx] = target_label
    poison_mask[idxs] = True
    return x_poisoned, y_poisoned, poison_mask

# ---------------------------- 视觉与字体 ----------------------------
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rc('font', family='Arial', size=25)

# ---------------------------- 抽特征 ----------------------------
@torch.no_grad()
def extract_features(model, dataloaders, device,
                     forget_party_idx=1,
                     trigger_value=1.0, trigger_size=2, target_label=5,
                     poison_fraction=0.0,
                     max_samples=3000):
    loaders_iter  = [iter(dl) for dl in dataloaders]
    total_batches = min(len(dl) for dl in dataloaders)

    feats, labels, flags = [], [], []
    collected = 0
    model.eval()

    for _ in range(total_batches):
        # ---------- 同步取一个 batch ----------
        xs, y = [], None
        for p_idx, it in enumerate(loaders_iter):
            try:
                x_p, y_p = next(it)
            except StopIteration:
                return (np.concatenate(feats),
                        np.concatenate(labels),
                        np.concatenate(flags)[:collected])

            xs.append(x_p)
            if p_idx == 0:          # 假设 party‑0 持有标签
                y = y_p

        # ---------- 注入后门（只改 forget_party_idx 那一方） ----------
        bsz         = xs[0].size(0)
        flag_batch  = torch.zeros(bsz, dtype=torch.bool)
        for p_idx, x_p in enumerate(xs):
            x_p = x_p.to(device)
            if p_idx == forget_party_idx:
                x_p, y, mask = add_trigger_to_right(
                    x_batch        = x_p,
                    y_batch        = y,
                    trigger_value  = trigger_value,
                    trigger_size   = trigger_size,
                    target_label   = target_label,
                    poison_fraction= poison_fraction
                )
                flag_batch |= mask.cpu()
            xs[p_idx] = x_p
        y = y.to(device)

        # ---------- 前向得到特征 ----------
        vecs = []
        for p_idx, x_p in enumerate(xs):
            f_map = model.client_list[p_idx].part(x_p)
            vecs.append(f_map.view(bsz, -1).cpu())
        feat_cat = torch.cat(vecs, dim=1)

        feats.append(feat_cat.numpy())
        labels.append(y.cpu().numpy())
        flags.append(flag_batch.numpy())
        collected += bsz
        if collected >= max_samples:
            break

    return (np.concatenate(feats),
            np.concatenate(labels),
            np.concatenate(flags)[:collected])

# ---------------------------- 画 t‑SNE ----------------------------
def tsne_plot(feats, labels, flags, outfile):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    z = TSNE(n_components=2, metric='cosine', init='pca',
             random_state=0).fit_transform(feats)

    n_cls = len(np.unique(labels))
    cmap  = plt.colormaps.get_cmap('tab10')
    norm  = matplotlib.colors.Normalize(vmin=0, vmax=10)

    for c in range(n_cls):
        idx_c   = (labels == c) & (~flags)
        idx_bad = (labels == c) & ( flags)
        if idx_c.any():
            ax.scatter(z[idx_c,0], z[idx_c,1],
                       c=[cmap(c % 10)], marker='o',
                       s=20, alpha=0.7, zorder=1)
        if idx_bad.any():
            ax.scatter(z[idx_bad,0], z[idx_bad,1],
                       c=[cmap(c % 10)], marker='*',
                       s=50, alpha=0.9, zorder=3)

    # 颜色条
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, ticks=np.arange(n_cls)+0.5,
                      fraction=0.046, pad=0.04)
    cb.ax.set_yticklabels(range(n_cls))
    cb.set_label('Class index', rotation=270, labelpad=25)

    # 图例
    # handles = [
    #     Line2D([], [], marker='o', linestyle='', color='gray',
    #        markersize=8,  label='Clean sample'),
    #     Line2D([], [], marker='*', linestyle='', color='gray',
    #        markersize=10, label='Poisoned sample')
    # ]
    # labels = ['Clean sample', 'Poisoned sample']
    # ax.legend(handles=handles, labels=labels,       # ← 关键：显式参数名
    #       loc='upper right', frameon=False, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[t‑SNE] saved → {outfile}")

def extract_features2(model, dataloaders, device, max_samples=3000):
    loaders_iter  = [iter(dl) for dl in dataloaders]
    total_batches = min(len(dl) for dl in dataloaders)
    model.to(device)
    feats, labels, flags = [], [], []
    collected = 0
    model.eval()

    for _ in range(total_batches):
        # ---------- 同步取一个 batch ----------
        xs, y = [], None
        for p_idx, it in enumerate(loaders_iter):
            try:
                x_p, y_p = next(it)
            except StopIteration:
                return (np.concatenate(feats),
                        np.concatenate(labels),
                        np.concatenate(flags)[:collected])

            xs.append(x_p)
            if p_idx == 0:          # 假设 party‑0 持有标签
                y = y_p

        # ---------- 前向得到特征 ----------
        bsz = xs[0].size(0)
        vecs = []
        for p_idx, x_p in enumerate(xs):
            x_p = x_p.to(device)
            f_map = model.client_list[p_idx].part(x_p)
            vecs.append(f_map.view(bsz, -1).cpu())
        feat_cat = torch.cat(vecs, dim=1)

        feats.append(feat_cat.detach().numpy())
        labels.append(y.cpu().detach().numpy())
        flags.append(np.zeros(bsz, dtype=np.bool))
        collected += bsz
        if collected >= max_samples:
            break

    return (np.concatenate(feats),
            np.concatenate(labels),
            np.concatenate(flags)[:collected])
# ---------------------------- 主逻辑 ----------------------------
if __name__ == "__main__":
    # --------- 配置 ---------
    num_party        = 3
    dataset          = "FashionMNIST"          # 自行修改
    device           = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
    trigger_val      = 1.0
    trigger_size     = 2
    target_label     = 5
    poisoned_frac    = 0.0
    forget_party_idx = 1                # 与训练时保持一致

    # --------- 加载数据 ---------
    trainloaders, testloaders, in_ch, out_dim = \
        load_and_split_data(dataset, num_parties=num_party)
    # --------- 初始化模型结构 ---------
    splitnn = SplitNN(num_parties=num_party,
                      in_channel=in_ch, out_dim=out_dim)
    splitnn.to(device)
    splitnn_2 = SplitNN(num_parties=num_party-1,
                        in_channel=in_ch, out_dim=out_dim)
    # --------- 三个模型路径（按需改） ---------
    ckpt_paths = {
        'original': f'./res/{dataset}/bk/vfl_best_model.pth',
        'unlearn':  f'./res/{dataset}/un/rmu_epoch_20.pth',
        'retrain':  f'./res/{dataset}/retrain/vfl_best_model.pth'
    }
    outfiles = {
        'original': './draw/original_tsne.pdf',
        'unlearn':  './draw/unlearned_tsne.pdf',
        'retrain':  './draw/retrain_tsne.pdf'
    }

    # --------- 依次加载→抽特征→绘图 ---------
    for tag, path in ckpt_paths.items():
        ckpt = torch.load(path, map_location=device)
        if tag != 'retrain':
            splitnn.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"[INFO] {tag} model loaded from {path}")
            feats, labels, flags = extract_features(
                splitnn, testloaders, device,
                forget_party_idx = forget_party_idx,
                trigger_value    = trigger_val,
                trigger_size     = trigger_size,
                target_label     = target_label,
                poison_fraction  = poisoned_frac,
                max_samples      = 3000
            )
        else:
            unlearning_index = [forget_party_idx]
            trainloaders = remove_from_index(unlearning_index, trainloaders)
            testloaders  = remove_from_index(unlearning_index, testloaders)
            splitnn_2.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            feats, labels, flags = extract_features2(
                splitnn_2, testloaders, device,
                max_samples      = 3000
            )
        tsne_plot(feats, labels, flags, outfiles[tag])
