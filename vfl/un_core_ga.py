# vfl/un_core_ga.py
import copy, torch
from typing import List
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from utils.bk import add_trigger_to_right, compute_backdoor_rate               # 如需后门验证
from utils.evaluate import vfl_eval
from vfl.core import SplitNN

def ga_unlearn(splitnn:      SplitNN,
               forget_party: int,                       # 仅支持单个 P_f
               data_loader_list: List[torch.utils.data.DataLoader],
               device,
               *,
               lr            = 5e-2,                    # 大一点收敛更快
               ascent_epoch  = 2,
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
    ga_csv_path = save_path + "ga_unlearned.csv" if save_path else None
    # -------- 梯度上升循环 --------
    M.train()
    for ep in range(ascent_epoch):
        iter_list = [iter(dl) for dl in data_loader_list]
        num_of_batches = min(len(dl) for dl in data_loader_list)
        print(f"GA Epoch {ep+1}/{ascent_epoch}, batches: {num_of_batches}...")
        for _ in tqdm(range(num_of_batches), desc=f'Epoch {ep+1}/{ascent_epoch}'):
            # -------- 取同步 batch --------
            xs, y_ref = [], None
            for p, it in enumerate(iter_list):
                x_p, y_p = next(it)
                if p == forget_party and trigger_cfg is not None:
                    x_p, y_ref = add_trigger_to_right(
                        x_batch       = x_p,
                        y_batch       = y_p,
                        trigger_value = trigger_cfg["value"],
                        trigger_size  = trigger_cfg["size"],
                        target_label  = trigger_cfg["tgt_label"],
                        poison_fraction = trigger_cfg["frac"]
                    )
                    y_ref = y_ref.to(device)
                xs.append(x_p.to(device))
            if y_ref is None:
                raise ValueError("Trigger configuration required for P_f")
            # 仅对 P_f 的参数求反向 → 梯度上升
            opt.zero_grad()
            logits = M(xs)
            loss   = -F.cross_entropy(logits, y_ref)        # 负号 = 上升
            loss.backward()
            opt.step()

        # —— 监控 ——
        if testloaders is not None:
            acc = vfl_eval(testloaders, M, device)
            train_acc = vfl_eval(data_loader_list, M, device)
            backdoor_acc =  compute_backdoor_rate(M, testloaders,
                                forget_party,
                                trigger_value=trigger_cfg["value"],
                                trigger_size=trigger_cfg["size"],
                                target_label=trigger_cfg["tgt_label"],
                                device=device)
            print(f"[GA] Epoch {ep+1}/{ascent_epoch}  Loss={-loss.item():.4f}  TestAcc={acc:.2f}%, TrainAcc={train_acc:.2f}%, BackdoorAcc={backdoor_acc:.2f}%")
            with open(ga_csv_path, 'a') as f:
                f.write(f"{ep+1},{-loss.item():.4f},{acc:.2f},{train_acc:.2f},{backdoor_acc:.2f}\n")
    if save_path:
        torch.save({"model": M.state_dict()}, save_path + "ga_unlearned.pth")
    return M
