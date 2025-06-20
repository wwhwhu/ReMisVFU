import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.bk import compute_backdoor_rate
from utils.evaluate import vfl_eval
from utils.data_split import load_and_split_data
from vfl.core import Client, Server, SplitNN
from tqdm import tqdm

# ----------------------------
# Main VFL Training Loop
# ----------------------------
if __name__ == "__main__":
    
    # Hyperparameters
    num_parties = 3
    epoch = 20
    learning_rate = 1e-3

    dataset_list = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
    dataset_name_index = 3
    dataset_name = dataset_list[dataset_name_index]
    # Device configuration
    device = torch.device(f"cuda:{dataset_name_index % 4}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    criterion = nn.CrossEntropyLoss()
    # Load & split data among parties
    trainloaders, testloaders, in_channel, out_dim = load_and_split_data(dataset_name, num_parties=num_parties)
    
    # Initialize global model (SplitNN) for each party copy
    splitnn = SplitNN(num_parties, in_channel=in_channel, out_dim=out_dim)
    print(f"model server structure:\n{splitnn.server}")
    splitnn.toDevice(device)
    # 初始化一个csv保存loss以及train_acc和test_acc
    with open(f'./res/{dataset_name}/full/vfl_training_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,test_acc,backdoor_acc,time\n')
    best_acc = 0.0
    poison_fraction = 0.1    # 每个 batch 中毒比例
    target_label = 5         # 后门触发时的目标标签
    trigger_value = 1.0      # 触发器像素值（最亮白）
    trigger_size = 2         # 触发器大小 2*2
    target_party = 1         # 对 Party1 注入后门
    for i in range(epoch):        
        # 每个 DataLoader 启一个迭代器
        time_start = time.time()
        loaders_iter = [iter(dl) for dl in trainloaders]
        num_batches  = min(len(dl) for dl in trainloaders)
        print(f"\nEpoch {i+1}/{epoch}, num of parties: {len(trainloaders)}, learning rate: {learning_rate}, batches: {num_batches}...")
        running_loss = 0
        for _ in tqdm(range(num_batches), desc="Training Batches"):
            # -------- 取同步 batch --------
            xs, y_ref = [], None
            for p, it in enumerate(loaders_iter):
                x_p, y_p = next(it)                           # y_p 只在 active 方有用
                # 打印大小
                # print(f"Party {p} batch size: {x_p.shape}, label size: {y_p.shape}")
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
        train_acc = vfl_eval(trainloaders, splitnn, device)
        # -------- 计算 test acc --------
        test_acc = vfl_eval(testloaders, splitnn, device)
        # -------- 保存日志 --------
        backdoor_acc = compute_backdoor_rate(
            splitnn=splitnn,
            testloaders2=testloaders,
            target_party=target_party,
            trigger_value=trigger_value,
            trigger_size=trigger_size,
            target_label=target_label,
            device=device
        )
        with open(f'./res/{dataset_name}/full/vfl_training_log.csv', 'a') as f:
            f.write(f"{i+1},{loss:.4f},{train_acc:.2f},{test_acc:.2f},{backdoor_acc:.2f},{time.time() - time_start:.2f}\n")
        print(f"Epoch {i+1}/{epoch} completed. Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Backdoor Acc: {backdoor_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存整个模型
            ckpt = {
                "model": splitnn.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt":  splitnn.server_opt.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, f"./res/{dataset_name}/full_3/vfl_best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
