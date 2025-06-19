# kd_vfl.py
"""
Knowledge-Distillation Unlearning Pipeline for splitVFL
------------------------------------------------------
1) 载入数据并恢复训练好的 SplitNN（teacher）。
2) 在遗忘请求前抓取一轮(≈1 epoch) 的中间嵌入 (embeddings)。
3) 调用 vfl/un_core_kd.py 中的 vfu_kd_unlearn 获得 student SplitNN，
   该模型已去除被遗忘方 P_f 的分支。
4) 评估并保存 student 模型。

依赖:
- utils.data_split.load_and_split_data
- utils.data_split.remove_from_index
- utils.evaluate.vfl_eval
- vfl.core.SplitNN
- vfl.un_core_kd.vfu_kd_unlearn
"""

import os
import time
import torch
from tqdm import tqdm

from utils.bk import add_trigger_to_right
from utils.data_split import load_and_split_data, remove_from_index
from utils.evaluate   import vfl_eval
from vfl.core         import SplitNN
from vfl.un_core_kd   import vfu_kd_unlearn
trigger_value = 1.0
trigger_size  = 2
target_label  = 5
poison_frac   = 0.10

# --------------------------------------------------
# 1. 抓取一轮嵌入 (teacher 前向后 client 输出)
# --------------------------------------------------
def snapshot_embeddings(splitnn, trainloaders_all, forget_party_idx, device):
    """
    返回:
        stored_embeddings : List[(emb_f, emb_ret, labels)]
        in_dim_retained   : int, 连接后 retained embeddings 的总维度
    """
    stored = []
    num_party  = len(trainloaders_all)
    loaders_it = [iter(dl) for dl in trainloaders_all]
    num_batches = min(len(dl) for dl in trainloaders_all)

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Snapshot"):
            xs, y_ref = [], None
            for p_idx, it in enumerate(loaders_it):
                x, y = next(it)
                if p_idx in forget_party_idx:
                    x, y_ref = add_trigger_to_right(
                        x_batch       = x,
                        y_batch       = y,
                        trigger_value = trigger_value,
                        trigger_size  = trigger_size,
                        target_label  = target_label,
                        poison_fraction = poison_frac
                    )
                    y_ref = y_ref.to(device)
                xs.append(x.to(device))
                

            # client 前向，获取特征
            embeds = []
            for p_idx in range(num_party):
                feat = splitnn.client_list[p_idx].part(xs[p_idx])
                embeds.append(feat.view(xs[p_idx].size(0), -1))  # flatten

            emb_f  = embeds[forget_party_idx[0]].cpu()
            emb_ret_list = [embeds[i].cpu()                          # list 形式存储
                            for i in range(num_party)
                            if i not in forget_party_idx]

            stored.append((emb_f, emb_ret_list, y_ref.cpu()))
    # 获取 retained embeddings 的总维度
    in_dims_retained = sum(emb.shape[1] for emb in emb_ret_list)
    return stored, in_dims_retained


# --------------------------------------------------
# 2. 入口
# --------------------------------------------------
def main():
    # ------------ 基本参数 ------------
    num_party         = 3
    forget_party_idx  = [1]                 # 要遗忘的 Party
    dataset_list      = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
    dataset_idx       = 4                  # 选择数据集
    dataset_name      = dataset_list[dataset_idx]

    device = torch.device(
        f"cuda:{dataset_idx % 4}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------ 数据加载 ------------
    trainloaders_all, testloaders_all, in_channel, out_dim = \
        load_and_split_data(dataset_name, num_parties=num_party)

    # ------------ 构建并载入 teacher 模型 ------------
    splitnn = SplitNN(num_parties=num_party,
                      in_channel=in_channel,
                      out_dim=out_dim).to(device)

    ckpt_path = f'./res/{dataset_name}/bk/vfl_best_model.pth'
    splitnn.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    splitnn.eval()
    print(f"[INFO] Teacher model restored from {ckpt_path}")

    # ------------ Snapshot embeddings ------------
    print("[STEP] Snapshotting one epoch of embeddings ...")
    stored_embeds, in_dims_retained = snapshot_embeddings(
        splitnn, trainloaders_all, forget_party_idx, device)
    print(f"[INFO] Snapshot finished. Cached batches: {len(stored_embeds)}")
    # ------------ 评估 student 模型 ------------
    retain_trainloaders = remove_from_index(forget_party_idx, trainloaders_all)
    retain_testloaders  = remove_from_index(forget_party_idx, testloaders_all)
    # ------------ KD-based Unlearning ------------
    print("[STEP] Start KD-based unlearning ...")
    save_dir = f'./res/{dataset_name}/un_kd'
    student_splitnn = vfu_kd_unlearn(
        splitnn_teacher   = splitnn,
        forget_party_idx  = forget_party_idx,
        stored_embeds     = stored_embeds,
        in_dims_retained  = in_dims_retained,
        out_dim           = out_dim,
        device            = device,
        alpha             = 0.7,
        T                 = 4.0,
        lr                = 1e-3,
        in_channel        = in_channel,
        epochs            = 12,
        retain_trainloaders=retain_trainloaders,
        retain_testloaders = retain_testloaders,
        save_dir= save_dir
    )
    print("[INFO] KD-based unlearning completed.")
    
# --------------------------------------------------
if __name__ == "__main__":
    main()
