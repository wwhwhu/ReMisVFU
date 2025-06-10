# ----------------------------
# 后门触发器函数
# ----------------------------
import copy
import torch
from vfl.core import SplitNN


def add_trigger_to_right(x_batch: torch.Tensor,
                         y_batch: torch.Tensor,
                         trigger_value: float = 1.0,
                         trigger_size: int = 2,
                         target_label: int = 0,
                         poison_fraction: float = 0.1):
    """
    对 Party1 的右半图 x_batch 添加后门 trigger：
      - 在右下角 trigger_size×trigger_size 位置加 trigger_value
      - 将这些样本的标签设为 target_label
    输入：
      x_batch       : [B,1,28,14]
      y_batch       : [B]
      trigger_value : 触发器像素值 (MNIST 预处理后 [0,1] 范围)
      trigger_size  : 触发器大小 (例如 2 表示 2×2 方块)
      target_label  : 后门目标类别
      poison_fraction: 本批次中毒比例 (0~1)
    输出：
      x_poisoned, y_poisoned
    """
    B, C, H, W = x_batch.shape  # C=1, H=28, W=14
    x_poisoned = x_batch.clone()
    y_poisoned = y_batch.clone()

    num_poison = int(poison_fraction * B)
    if num_poison == 0:
        return x_poisoned, y_poisoned

    idxs = torch.randperm(B)[:num_poison]
    for idx in idxs:
        row_start = H - trigger_size
        col_start = W - trigger_size
        x_poisoned[idx, 0, row_start:H, col_start:W] = trigger_value
        y_poisoned[idx] = target_label

    return x_poisoned, y_poisoned

# ----------------------------
# 计算后门成功率
# ----------------------------
def compute_backdoor_rate(splitnn: SplitNN,
                          testloaders2: list,
                          target_party: int,
                          trigger_value: float,
                          trigger_size: int,
                          target_label: int,
                          device: torch.device):
    """
    在测试集上给 target_party 添加触发器，计算模型输出 target_label 的比例。
    输入：
      splitnn       : 训练好的 SplitNN 模型
      testloaders   : 原始的 testloaders 列表 (len = num_parties)，每个 DataLoader 提供 [B,1,28,14], [B]
      target_party  : 要加触发器的 Party 索引 (此处 1)
      trigger_value : 触发器像素值
      trigger_size  : 触发器大小
      target_label  : 后门目标类别
      device        : torch.device
    输出：
      backdoor_success_rate: 百分比 (0~100)
    """
    splitnn.eval()
    total = 0
    success = 0

    # 只对 target_party 加 trigger，其他 parties 正常读取原始测试数据
    # 假设 len(testloaders) == num_parties
    testloaders = copy.deepcopy(testloaders2)
    iters = [iter(dl) for dl in testloaders]
    num_batches = min(len(dl) for dl in testloaders)

    with torch.no_grad():
        for _ in range(num_batches):
            xs = []
            y_true = None
            # 先逐个从各 Party 取 batch
            for p, it in enumerate(iters):
                x_p, y_p = next(it)
                x_p = x_p.to(device)
                y_p = y_p.to(device)
                if p == target_party:
                    # 全部置 poison_fraction=1.0 才算成功率
                    x_p, _ = add_trigger_to_right(
                        x_batch=x_p,
                        y_batch=y_p,
                        trigger_value=trigger_value,
                        trigger_size=trigger_size,
                        target_label=target_label,
                        poison_fraction=1.0
                    )
                xs.append(x_p)
                if p == 0:
                    # 仍以 Party0 的标签为 ground truth （用于训练时），
                    # 但后门成功率在这里只检测模型是否预测 target_label，不看真实 y_p
                    y_true = y_p

            logits = splitnn(xs)
            preds = torch.argmax(logits, dim=1)
            total += preds.size(0)
            success += (preds == target_label).sum().item()

    backdoor_success_rate = success / total * 100
    return backdoor_success_rate