import copy
import os
import time
from typing import List
from itertools import chain

from utils.bk import add_trigger_to_right, compute_backdoor_rate
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
    # 获取entire_dataset_list
    entire_datasets_list = []
    forget_index = 0
    retain_index = 0
    for idx in range(len(forget_datasets_list) + len(retain_datasets_list)):
        if idx in forget_party_idx:
            entire_datasets_list.append(forget_datasets_list[forget_index])
            forget_index += 1
        else:
            entire_datasets_list.append(retain_datasets_list[retain_index])
            retain_index += 1
    
    forget_iter = []
    forget_len = len(forget_datasets_list)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    save_dir = save_path + "rmu_log.csv"
    # 获取迭代器list
    one_iter = []
    for i in range(forget_len):
        one_iter.append(iter(forget_datasets_list[i]))
    
    for _ in range(len(forget_datasets_list[0])):
        x = []
        for i in range(forget_len):
            x.append(next(one_iter[i]))
        forget_iter.append(x)

    all_iter = []
    all_len = len(entire_datasets_list)
    one_iter = []
    for i in range(all_len):
        one_iter.append(iter(entire_datasets_list[i]))
    for _ in range(len(entire_datasets_list[0])):
        x = []
        for i in range(all_len):
            x.append(next(one_iter[i]))
        all_iter.append(x)
    
    print(f"Forget Batches: {len(forget_iter)}, All Batches: {len(all_iter)}")
    print(f"Forget Batches Features: {len(forget_iter[0])}, All Samples Features: {len(all_iter[0])}")
    
    M_updated = copy.deepcopy(splitnn).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    poison_fraction = 0.1    # 每个 batch 中毒比例
    target_label = 5         # 后门触发时的目标标签
    trigger_value = 1.0      # 触发器像素值（最亮白）
    trigger_size = 2         # 触发器大小 2*2

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

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, M_updated.parameters()), lr=1e-2)
    M_updated.train()
    # 写入表头到 CSV 文件
    if save_path is not None:
        with open(save_dir, 'w') as f:
            f.write('epoch,forget_loss,retain_loss,total_loss,test_acc,backdoor_success_rate,time\n')
    # 5. RMU 循环
    for epoch in range(num_unlearn_epochs):
        time_start = time.time()
        if testloaders is not None:
            acc = vfl_eval(testloaders, M_updated, device)
            print(f"Test Acc: {acc:.2f}%")
            bd = compute_backdoor_rate(M_updated, testloaders,
                                       forget_party_idx[0],
                                       trigger_value=trigger_value,
                                       trigger_size=trigger_size,
                                       target_label=target_label,
                                       device=device)
            print(f"Backdoor Success Rate: {bd:.2f}%")
        total_f, total_r = 0.0, 0.0
        for forget_list, all_list in zip(forget_iter, all_iter):
            loss_f = 0.0
            for i in range(len(forget_list)):
                x_forget, _ = forget_list[i]
                x_forget = x_forget.to(device)
                _ = _.to(device)
                # 对其添加后门
                x_forget, y_ref = add_trigger_to_right(
                        x_batch=x_forget,
                        y_batch=_,
                        trigger_value=trigger_value,
                        trigger_size=trigger_size,
                        target_label=target_label,
                        poison_fraction=poison_fraction
                    )
                fmap = M_updated.client_list[forget_party_idx[i]].part(x_forget)
                # print(fmap.requires_grad, fmap.grad_fn)                   # 应该不是 None            # True → 说明图保留了
                h_u = fmap.view(fmap.size(0), -1)
                tgt = (c*u).unsqueeze(0).expand_as(h_u)
                loss_f += ((h_u - tgt)**2).mean()
            loss_f /= len(forget_list)

            # 计算 retain loss
            loss_r = 0.0
            x_input = []
            y_ref = None
            for i in range(len(all_list)):
                x, _ = all_list[i]
                x = x.to(device)
                _ = _.to(device)
                if i in forget_party_idx:
                    x, y_ref = add_trigger_to_right(
                        x_batch=x,
                        y_batch=_,
                        trigger_value=trigger_value,
                        trigger_size=trigger_size,
                        target_label=target_label,
                        poison_fraction=poison_fraction
                    )
                x_input.append(x)
            # print(f"Input shape: {[x.shape for x in x_input]}")
            # print(f"Input labels shape: {y_ref.shape}")
            prediction = M_updated(x_input)
            loss_r = criterion(prediction, y_ref)

            # 单次更新
            loss = loss_f + alpha * loss_r

            # 方案一：加权求和梯度
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # 方案二：梯度投影
            optimizer.zero_grad()      
            # 分别计算两个 loss 的梯度
            grads_f = torch.autograd.grad(loss_f, M_updated.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            grads_r = torch.autograd.grad(loss_r, M_updated.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            # 投影 retain 梯度到 forget 梯度的非冲突方向
            projected_grads = []
            for g_f, g_r in zip(grads_f, grads_r):
                if g_f is None:
                    projected_grads.append(g_r)
                    continue
                if g_r is None:
                    projected_grads.append(g_f)
                    continue
                g_f_flat = g_f.view(-1)
                g_r_flat = g_r.view(-1)
                
                dot_product = torch.dot(g_f_flat, g_r_flat)

                if dot_product < 0:  # 表示 retain 与 forget 冲突
                    proj = dot_product / (g_f_flat.norm() ** 2 + 1e-12) * g_f_flat
                    g_r_flat = g_r_flat - proj  # 去掉 retain 在 forget 上的负向分量

                # forget 保留原始方向，retain 被调整后合并
                final_grad = g_f_flat + g_r_flat * alpha
                projected_grads.append(final_grad.view_as(g_f))
            
            # 将合并后的梯度赋值并更新模型
            with torch.no_grad():
                for p, g in zip(M_updated.parameters(), projected_grads):
                    if g is not None:
                        p.grad = g.clone()
            optimizer.step()
            
            total_f += loss_f.item()
            total_r += loss_r.item()

        print(f"[RMU Epoch {epoch+1}/{num_unlearn_epochs}] "
              f"ForgetLoss={total_f/len(forget_iter):.4f}, "
              f"RetainLoss={total_r/len(all_iter):.4f}, "
              f"TotalLoss={loss.item():.4f}")
        if save_path is not None:
            save_path2 = save_path + f"rmu_epoch_{epoch+1}.pth"
            ckpt = {
                "model": M_updated.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt":  splitnn.server_opt.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, save_path2)
            print(f"Model saved to {save_path2}")
        # 写入日志
        if save_path is not None:
            with open(save_dir, 'a') as f:
                f.write(f"{epoch+1},{total_f/len(forget_iter):.4f},{total_r/len(all_iter):.4f},"
                        f"{loss.item():.4f},{acc:.2f},{bd:.2f},{time.time() - time_start:.2f}\n")
    return M_updated
