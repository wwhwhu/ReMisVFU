# vfl/un_core_ga.py
import copy, torch
import torch.nn.functional as F
from torch import optim
from utils.bk import add_trigger_to_right               # 如需后门验证
from utils.evaluate import vfl_eval
from vfl.core import SplitNN

def ga_unlearn(splitnn:      SplitNN,
               forget_party: int,                       # 仅支持单个 P_f
               forget_loader: torch.utils.data.DataLoader,
               device,
               *,
               lr            = 5e-2,                    # 大一点收敛更快
               ascent_epoch  = 8,
               trigger_cfg   = None,                    # dict or None
               testloaders   = None,
               save_path     = None):

    # -------- 复制模型并冻结除 P_f 外的权重 --------
    M = copy.deepcopy(splitnn).to(device)
    for idx, client in enumerate(M.client_list):
        req_grad = (idx == forget_party)                # 只有 P_f 可训练
        for p in client.part.parameters():
            p.requires_grad_(req_grad)
    for p in M.server.partC.parameters():                # server 端冻结
        p.requires_grad_(True)

    opt = optim.SGD(filter(lambda p: p.requires_grad, M.parameters()),
                    lr=lr, momentum=0.9)

    # -------- 梯度上升循环 --------
    M.train()
    for ep in range(ascent_epoch):
        for x_f, y in forget_loader:
            x_f, y = x_f.to(device), y.to(device)

            # 可选：给遗忘方批次注入后门触发器
            if trigger_cfg is not None:
                x_f, y = add_trigger_to_right(
                    x_batch         = x_f,
                    y_batch         = y,
                    trigger_value   = trigger_cfg["value"],
                    trigger_size    = trigger_cfg["size"],
                    target_label    = trigger_cfg["tgt_label"],
                    poison_fraction = trigger_cfg["frac"]
                )

            # 构造完整输入（其他 Party 用原 SplitNN 静态前向）
            xs = [None]*len(M.client_list)
            xs[forget_party] = x_f
            with torch.no_grad():
                for k in range(len(M.client_list)):
                    if k != forget_party:
                        xs[k], _ = next(iter(splitnn.client_list[k].cached_loader))
                        xs[k] = xs[k].to(device)

            # 仅对 P_f 的参数求反向 → 梯度上升
            opt.zero_grad()
            logits = M(xs)
            loss   = -F.cross_entropy(logits, y)        # 负号 = 上升
            loss.backward()
            opt.step()

        # —— 监控 ——
        if testloaders is not None:
            acc = vfl_eval(testloaders, M, device)
            print(f"[GA] Epoch {ep+1}/{ascent_epoch}  Loss={-loss.item():.4f}  TestAcc={acc:.2f}%")

    if save_path:
        torch.save({"model": M.state_dict()}, save_path + "ga_unlearned.pth")
    return M
