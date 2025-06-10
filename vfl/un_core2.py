import copy
from typing import List
from itertools import chain

from utils.bk import compute_backdoor_rate
from utils.evaluate import vfl_eval
from vfl.core import SplitNN
import torch
import torch.optim as optim

def get_party_feature(splitnn: SplitNN, party_idx: int, x_p: torch.Tensor) -> torch.Tensor:
    client = splitnn.client_list[party_idx]
    with torch.no_grad():
        feat_map = client.part(x_p)           # [B, 64, 7, 7]
    B, C, H, W = feat_map.shape
    return feat_map.view(B, C*H*W)           # [B, 64*7*7]

def rmu_unlearn(
    splitnn: SplitNN,
    forget_party_idx: List[int],
    forget_datasets_list: List[torch.utils.data.DataLoader],
    retain_datasets_list: List[torch.utils.data.DataLoader],
    device: torch.device,
    c: float = 1.0,
    alpha: float = 0.5,
    num_unlearn_epochs: int = 10,
    save_path: str = None,
    testloaders = None
) -> SplitNN:
    # 1. 复制模型
    M_frozen  = copy.deepcopy(splitnn).to(device)
    M_updated = copy.deepcopy(splitnn).to(device)

    # 2. 推断 d
    sample_feat = None
    for loader in forget_datasets_list:
        try:
            x_f, _ = next(iter(loader))
            x_f = x_f.to(device)
            with torch.no_grad():
                sample_feat = M_updated.client_list[forget_party_idx[0]].part(x_f)
            break
        except StopIteration:
            continue
    if sample_feat is None:
        raise ValueError("All forget loaders empty")
    _, Cf, Hf, Wf = sample_feat.shape
    d = Cf*Hf*Wf

    # 3. 生成 u
    u = torch.rand(d, device=device)
    u /= torch.norm(u, p=2)

    # 4. 冻结参数（示例只给 conv2 开放，可根据实际层名调整）
    # for p_idx, client in enumerate(M_updated.client_list):
    #     for name, param in client.part.named_parameters():
    #         param.requires_grad = (p_idx == forget_party_idx[0] and "conv2" in name)
    # for name, param in M_updated.server.named_parameters():
    #     param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, M_updated.parameters()), lr=1e-2)
    M_updated.train()

   
    # 5. RMU 循环
    for epoch in range(num_unlearn_epochs):
        forget_iter = chain.from_iterable(forget_datasets_list)
        retain_iter = chain.from_iterable(retain_datasets_list)
        n_forget = sum(len(dl) for dl in forget_datasets_list)
        n_retain = sum(len(dl) for dl in retain_datasets_list)
        steps = min(n_forget, n_retain)
        print(f"Forget steps: {n_forget}, Retain steps: {n_retain}")
        if testloaders is not None:
            acc = vfl_eval(testloaders, M_updated, device)
            print(f"Test Acc: {acc:.2f}%")
            bd = compute_backdoor_rate(M_updated, testloaders,
                                       forget_party_idx[0],
                                       trigger_value=1.0,
                                       trigger_size=4,
                                       target_label=5,
                                       device=device)
            print(f"Backdoor Success Rate: {bd:.2f}%")

        total_f, total_r = 0.0, 0.0
        for _ in range(steps):
            # forget loss
            x_f, _ = next(forget_iter)
            x_f = x_f.to(device)
            fmap = M_updated.client_list[forget_party_idx[0]].part(x_f)
            h_u = fmap.view(fmap.size(0), -1)
            tgt = (c*u).unsqueeze(0).expand_as(h_u)
            loss_f = ((h_u - tgt)**2).mean()

            # retain loss
            x_r, _ = next(retain_iter)
            x_r = x_r.to(device)
            fmap2 = M_updated.client_list[forget_party_idx[0]].part(x_r)
            h_u2 = fmap2.view(fmap2.size(0), -1)
            with torch.no_grad():
                fmap_ref = M_frozen.client_list[forget_party_idx[0]].part(x_r)
                h_ref = fmap_ref.view(fmap_ref.size(0), -1)
            loss_r = ((h_u2 - h_ref)**2).mean()

            # 单次更新
            loss = loss_f + alpha*loss_r
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_f += loss_f.item()
            total_r += loss_r.item()

        print(f"[RMU Epoch {epoch+1}/{num_unlearn_epochs}] "
              f"ForgetLoss={total_f/steps:.4f}, RetainLoss={total_r/steps:.4f}")
        if save_path is not None:
            ckpt = {
                "model": M_updated.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt":  splitnn.server_opt.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, save_path)
            print(f"Model saved to {save_path}")
    return M_updated
