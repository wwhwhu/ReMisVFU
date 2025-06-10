import numpy as np
import torch

@torch.no_grad()
def vfl_eval(dataloaders, model, device):
    """
    评估 SplitNN 在给定 dataloaders（列表）上的分类准确率 (%).

    Parameters
    ----------
    dataloaders : List[DataLoader]
        每个参与方一个 DataLoader，索引 0 的 loader 提供标签。
    model       : SplitNN
    device      : torch.device
    """
    model.eval()
    loaders_iter = [iter(dl) for dl in dataloaders]
    num_batches  = min(len(dl) for dl in dataloaders)

    total, correct = 0, 0
    # labels = []
    for _ in range(num_batches):
        xs, y_ref = [], None
        for p, it in enumerate(loaders_iter):
            try:
                x_p, y_p = next(it)
            except StopIteration:
                break
            xs.append(x_p.to(device))
            if p == 0:                         # 主动方提供标签
                y_ref = y_p.to(device)
                # labels.append(y_ref.cpu().numpy())  # 收集标签用于后续分析

        logits = model(xs)
        preds  = logits.argmax(dim=1)
        total  += y_ref.size(0)
        correct += (preds == y_ref).sum().item()
        # # 打印labels的分布情况
        # if len(labels) > 0:
        #     labels2 = np.concatenate(labels)
        #     print(f"Labels distribution: {np.unique(labels2, return_counts=True)}")
    model.train()                              # 恢复训练模式
    return 100.0 * correct / total if total else 0.0