import copy
import time
from typing import List
from utils.bk import compute_backdoor_rate
from utils.evaluate import vfl_eval
from vfl.core import SplitNN
import torch
import torch.optim as optim

def get_party_feature(splitnn: SplitNN, party_idx: int, x_p: torch.Tensor) -> torch.Tensor:
    """
    给定一个 SplitNN 和某个 Party 的单步输入，返回该 Party 最后一层编码器的
    Flatten 后中间表示向量。
    - splitnn: 已加载到 device 的模型
    - party_idx: Party 序号（0 ~ num_parties-1）
    - x_p:    [B, 1, 28, 14] Party_p 输入（左/右半 MNIST）
    返回:
    - feat_flat: [B, d]，d = FirstNet 最后一层输出维度 (64*7*7)
    """
    # 1) 先只过该 Party 的编码器部分（Client.part）
    client = splitnn.client_list[party_idx]
    # client.part 即是 FirstNet
    with torch.no_grad():
        feat_map = client.part(x_p)           # [B, 64, 7, 7]
    # Flatten
    B, C, H, W = feat_map.shape
    feat_flat = feat_map.view(B, C*H*W)      # [B, 64*7*7]
    return feat_flat

def rmu_unlearn(
    splitnn: SplitNN,
    forget_party_idx: list,
    forget_datasets_list: List[torch.utils.data.DataLoader],
    retain_datasets_list: List[torch.utils.data.DataLoader],
    device: torch.device,
    c: float = 2.0,
    num_unlearn_epochs: int = 10,
    save_path: str = None,
    testloaders = None
) -> SplitNN:
    """
    在 VFL 场景中，对多个“遗忘数据集”列表和“保留数据集”列表执行 RMU 操作。
    - splitnn: 已加载到 device 的全局模型 SplitNN
    - forget_party_idx: 要遗忘的 Party 索引
    - forget_datasets_list: 多个 DataLoader，每个 DataLoader 提供一批要遗忘的 Party 输入
    - retain_datasets_list: 多个 DataLoader，每个 DataLoader 提供一批要保留的 Party 输入
    - device: torch.device
    - c: 放大系数
    - alpha: Retain Loss 权重
    - num_unlearn_epochs: 遗忘迭代轮数
    返回：
    - M_updated: 执行 RMU 后的 SplitNN 模型
    """

    # 1. 复制全局模型：M_frozen（仅计算保留特征）和 M_updated（待优化）
    M_frozen  = copy.deepcopy(splitnn).to(device)
    M_updated = copy.deepcopy(splitnn).to(device)

    # 2. 推断 Party 最后一层输出维度 d
    M_updated.eval()
    # 获取retain_party_idx
    client_sum = len(M_updated.client_list)
    retain_party_idx = [i for i in range(client_sum) if i not in forget_party_idx]
    # 找到第一个非空的 forget DataLoader，取一个 batch 以推断 d
    sample_feat = None
    for loader in forget_datasets_list:
        try:
            x_forget, _ = next(iter(loader))
            x_forget = x_forget.to(device)
            with torch.no_grad():
                feat_map = M_updated.client_list[forget_party_idx[0]].part(x_forget)  # [B, C, H, W]
            sample_feat = feat_map
            break
        except StopIteration:
            continue
    if sample_feat is None:
        raise ValueError("forget_datasets_list 中所有 DataLoader 均为空")
    Bf, Cf, Hf, Wf = sample_feat.shape
    d = Cf * Hf * Wf
    
    # 3. 生成随机单位向量 u
    u = torch.rand(d, device=device)
    u = u / torch.norm(u, p=2)
    print(f"Party {forget_party_idx[0]} 最后一层输出维度 d={d}, 随机单位向量生成成功")
    # 4. 冻结除该 Party 最后一层编码器之外的所有参数
    # for p_idx, client in enumerate(M_updated.client_list):
    #     for name, param in client.part.named_parameters():
    #         if p_idx == forget_party_idx and ("conv2" in name):
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False
    # for name, param in M_updated.server.named_parameters():
    #     param.requires_grad = False
    print(M_updated)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, M_updated.parameters()),
        lr=1e-3
    )
    M_updated.train()
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in M_updated.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params}")
    # 初始化一个 CSV 保存日志
    if save_path is not None:
        with open(save_path + 'vfl_rmu_log.csv', 'w') as f:
            f.write('epoch,forget_loss,retain_loss,test_acc,backdoor_acc,time\n')
    # 5. 多轮 RMU：对多个遗忘 DataLoader 和多个保留 DataLoader 迭代
    for epoch in range(num_unlearn_epochs):
        time_start = time.time()
        if testloaders is not None:
            # 计算 test acc
            test_acc = vfl_eval(testloaders, M_updated, device)
            print(f"Test Acc: {test_acc:.2f}%")
            # 计算后门攻击准确率
            backdoor_acc = compute_backdoor_rate(M_updated, testloaders, 
                                                  forget_party_idx[0], 
                                                  trigger_value=1.0, 
                                                  trigger_size=4, 
                                                  target_label=5, 
                                                  device=device)
            print(f"Backdoor Success Rate: {backdoor_acc:.2f}%")
        # ——(1) Forget Loss：遍历所有 forget_datasets_list
        total_f_loss = 0.0
        count_f_batches = 0
        for forget_loader in forget_datasets_list:
            for x_forget, _ in forget_loader:
                x_forget = x_forget.to(device)  # [Bf, 1, 28, 14]
                # 计算 Party 最后输出，Flatten
                feat_map = M_updated.client_list[forget_party_idx[0]].part(x_forget)  # [Bf, Cf, Hf, Wf]
                h_u = feat_map.view(feat_map.size(0), -1)  # [Bf, d]
                target = (c * u).unsqueeze(0).expand(h_u.size(0), -1)  # [Bf, d]
                loss_f = ((h_u - target) ** 2).mean()
                optimizer.zero_grad()
                
                loss_f.backward()
                optimizer.step()
                total_f_loss += loss_f.item()
                count_f_batches += 1
        avg_f_loss = total_f_loss / max(count_f_batches, 1)

        # ——(2) Retain Loss：遍历所有 retain_datasets_list
        total_r_loss = 0.0
        count_r_batches = 0
        for retain_loader in retain_datasets_list:
            i = 0
            for xR, _ in retain_loader:
                xR = xR.to(device)  # [Br, 1, 28, 14]
                feat_u = M_updated.client_list[retain_party_idx[i]].part(xR)   # [Br, Cf, Hf, Wf]
                h_u2 = feat_u.view(feat_u.size(0), -1)                      # [Br, d]
                with torch.no_grad():
                    feat_f = M_frozen.client_list[retain_party_idx[i]].part(xR)  # [Br, Cf, Hf, Wf]
                    h_f2 = feat_f.view(feat_f.size(0), -1)                    # [Br, d]
                loss_r = ((h_u2 - h_f2) ** 2).mean()

                optimizer.zero_grad()
                loss_r.backward()
                optimizer.step()
                total_r_loss += loss_r.item()
                count_r_batches += 1
            i += 1
        avg_r_loss = total_r_loss / max(count_r_batches, 1)

        print(f"[RMU Epoch {epoch+1}/{num_unlearn_epochs}] "
              f"ForgetLoss={avg_f_loss:.4f}, RetainLoss={avg_r_loss:.4f}")
        
        if save_path is not None:
            ckpt = {
                "model": M_updated.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt":  splitnn.server_opt.state_dict(),
                "epoch": epoch
            }
            save_path2 = save_path + "vfl_rmu_model.pth"
            save_path3 = save_path + 'vfl_rmu_log.csv'
            torch.save(ckpt, save_path2)
            print(f"Model saved to {save_path2}")
            # 保存日志
            with open(save_path3, 'a') as f:
                f.write(f"{epoch+1},{avg_f_loss:.4f},{avg_r_loss:.4f},{test_acc:.2f},{backdoor_acc:.2f},{time.time() - time_start:.2f}\n")
    return M_updated