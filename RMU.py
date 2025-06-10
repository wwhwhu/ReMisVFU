# rmu_vfl.py

"""
在已有 VFL Relearning 代码基础上，添加 RMU（Representation Misdirection for Unlearning）逻辑：
对聚合前的某个 Party 的输出特征进行遗忘，构造 Forget Loss 和 Retain Loss。
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.bk import compute_backdoor_rate
from utils.evaluate import vfl_eval
from utils.data_split import load_and_split_data, remove_from_index
from vfl.core import Client, Server, SplitNN, FusionAvg
from vfl.un_core import rmu_unlearn

if __name__ == "__main__":
    num_party = 3
    # 超参数
    forget_party_idx = [1]  # 要遗忘的 Party 序号（0 ~ num_parties-1）
    dataset_list = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
    dataset_name_index = 4
    dataset_name = dataset_list[dataset_name_index]
    # Device configuration
    device = torch.device(f"cuda:{dataset_name_index % 4}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    trainloaders_all, testloaders_all, in_channel, out_dim = load_and_split_data(dataset_name, num_parties=num_party)
    splitnn = SplitNN(num_parties=3, in_channel=in_channel, out_dim=out_dim)

    # -------- RMU 遗忘 --------
    # 载入最优模型
    ckpt = torch.load(f'./res/{dataset_name}/bk/vfl_best_model.pth', map_location=device)
    splitnn.load_state_dict(ckpt["model"])

    retain_loader_list = remove_from_index(forget_party_idx, trainloaders_all)
    retain_testloaders_list = remove_from_index(forget_party_idx, testloaders_all)
    forget_loader_list = []
    for idx in range(3):
        if idx in forget_party_idx:
            forget_loader_list.append(trainloaders_all[idx])
    print(f"遗忘 Party {forget_party_idx} 的数据集数量: {len(forget_loader_list)}")
    print(f"保留 Party 的数据集数量: {len(retain_loader_list)}")
    # 执行 RMU 遗忘
    print("\n---- 开始 RMU 遗忘 (Party1) ----")
    M_unlearned = rmu_unlearn(
        splitnn=splitnn,
        forget_party_idx=forget_party_idx,
        forget_datasets_list=forget_loader_list,
        retain_datasets_list=retain_loader_list,
        device=device,
        c=2.0,
        num_unlearn_epochs=5,
        testloaders=testloaders_all,
        save_path=f"./res/{dataset_name}/un/"
    )
    print("---- RMU 遗忘完成 ----")

    with open(f'./res/{dataset_name}/un/vfl_tune_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,test_acc,backdoor_acc,time\n')
    # -------- 在剩余 Party 上训练几轮建立新模型 （retain_datasets_list） --------
    new_num_party = len(retain_loader_list)
    print(f"---- 开始在剩余 {new_num_party} Party 上训练新模型 ----")
    # 重新初始化 SplitNN
    splitnn = SplitNN(num_parties=new_num_party)
    splitnn.toDevice(device)
    # 将 M_unlearned 的参数加载到新模型中, 首先是加载M_unlearned的retain party
    i = 0
    for idx in range(num_party):
        if idx in forget_party_idx:
            continue
        splitnn.client_list[i].part.load_state_dict(M_unlearned.client_list[idx].part.state_dict())
        i += 1
    # 开始训练
    best_acc = 0.0
    fine_tune_epoch = 5
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()
    for i in range(fine_tune_epoch):        
        # 每个 DataLoader 启一个迭代器
        time_start = time.time()
        loaders_iter = [iter(dl) for dl in retain_loader_list]
        num_batches  = min(len(dl) for dl in retain_loader_list)
        print(f"\nEpoch {i+1}/{fine_tune_epoch}, num of parties: {len(retain_loader_list)}, learning rate: {learning_rate}, batches: {num_batches}...")
        running_loss = 0
        for _ in tqdm(range(num_batches), desc="Training Batches"):
            # -------- 取同步 batch --------
            xs, y_ref = [], None
            for p, it in enumerate(loaders_iter):
                x_p, y_p = next(it)                           # y_p 只在 active 方有用
                xs.append(x_p.to(device))
                if p == 0:  # Assume first party has the labels
                    y_ref = y_p.to(device)
            # -------- 前向 & 反向 --------
            splitnn.do_zero_grads()
            logits = splitnn(xs)
            loss   = criterion(logits, y_ref)
            running_loss += loss.item()
            loss.backward()
            splitnn.doStep()
        loss = running_loss / num_batches  # 平均损失
        # -------- 计算 train acc --------
        train_acc = vfl_eval(retain_loader_list, splitnn, device)
        # -------- 计算 test acc --------
        test_acc = vfl_eval(retain_testloaders_list, splitnn, device)
        # -------- 保存日志 --------
        with open(f'./res/{dataset_name}/un/vfl_tune_log.csv', 'a') as f:
            f.write(f"{i+1},{loss:.4f},{train_acc:.2f},{test_acc:.2f},{time.time() - time_start:.2f}\n")
        print(f"Epoch {i+1}/{fine_tune_epoch} completed. Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存整个模型
            ckpt = {
                "model": splitnn.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt":  splitnn.server_opt.state_dict(),
                "epoch": fine_tune_epoch
            }
            torch.save(ckpt, f"./res/{dataset_name}/un/vfl_best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    # # -------- 遗忘后重新聚合 --------
    # # 3.1 准备各 Party 最新模型参数列表
    # party_states = []
    # # Party0 & Party2 沿用上次 Relearn 后的最优模型/本地状态
    # party_states.append(splitnn.state_dict())         # Party0
    # party_states.append(M_unlearned.state_dict())     # Party1 已遗忘模型
    # # Party2：由于 remove_from_index, Party2 无本地区域，直接重用 M_unlearned
    # #         或者从最初数据分割中读原始 Party2 模型
    # # 这里示例直接用 Party0 状态作占位
    # party_states.append(splitnn.state_dict())         # Party2（占位）

    # fusion = FusionAvg(num_parties=3)
    # new_global_state = fusion.average_selected_models(party_states)

    # M_new_global = SplitNN(num_parties=3)
    # M_new_global.toDevice(device)
    # M_new_global.load_state_dict(new_global_state)
    # torch.save(new_global_state, "vfl_global_after_rmu.pth")
    # print("---- 重新聚合完成，已保存 vfl_global_after_rmu.pth ----")

    # # -------- 评估新全局模型 --------
    # # 由于 Party1 已被遗忘，重新构造 testloaders_all
    # _, testloaders_all = split_mnist_vfl(num_parties=3)
    # # 同样要 remove Party1
    # testloaders_new = remove_from_index([forget_party_idx], testloaders_all)

    # eval_acc_clean = vfl_eval(testloaders_new, M_new_global, device)
    # print(f"遗忘后新全局模型干净测试精度：{eval_acc_clean:.2f}%")
