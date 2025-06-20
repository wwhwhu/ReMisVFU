import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.bk import add_trigger_to_right, compute_backdoor_rate
from utils.evaluate import vfl_eval
from utils.data_split import load_and_split_data
from vfl.core import SplitNN

# ----------------------------
# Main VFL Training（含后门注入 & 成功率计算）
# ----------------------------
if __name__ == "__main__":
    
    # 超参数
    num_parties = 3
    epochs = 50
    lr = 1e-3

    dataset_list = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
    dataset_name_index = 4
    dataset_name = dataset_list[dataset_name_index]
    # Device configuration
    device = torch.device(f"cuda:{dataset_name_index % 4}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    criterion = nn.CrossEntropyLoss()

    trainloaders, testloaders, in_channel, out_dim = load_and_split_data(dataset_name, num_parties=num_parties)

    # 初始化全局 SplitNN
    splitnn = SplitNN(num_parties, in_channel=in_channel, out_dim=out_dim)
    print(f"Model server structure:\n{splitnn.server}")
    splitnn.toDevice(device)

    # 日志文件
    with open(f'./res/{dataset_name}/bk/vfl_training_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_acc,test_acc,backdoor_acc,time\n')

    best_acc = 0.0

    # 后门相关参数
    poison_fraction = 0.1    # 每个 batch 中毒比例
    target_label = 5         # 后门触发时的目标标签
    trigger_value = 1.0      # 触发器像素值（最亮白）
    trigger_size = 2         # 触发器大小 2*2
    target_party = 1         # 对 Party1 注入后门

    for epoch in range(1, epochs + 1):
        time_start = time.time()
        loaders_iter = [iter(dl) for dl in trainloaders]
        num_batches = min(len(dl) for dl in trainloaders)
        running_loss = 0.0

        print(f"\nEpoch {epoch}/{epochs}, num of parties: {len(trainloaders)}, learning rate: {lr}, batches: {num_batches}...")
        for _ in tqdm(range(num_batches), desc="Training Batches"):
            xs, y_ref = [], None

            # 同步取各 Party batch，并对 Party1 做中毒
            for p, it in enumerate(loaders_iter):
                x_p, y_p = next(it)
                x_p = x_p.to(device)
                y_p = y_p.to(device)
                if p == target_party:
                    x_poisoned, y_poisoned = add_trigger_to_right(
                        x_batch=x_p,
                        y_batch=y_p,
                        trigger_value=trigger_value,
                        trigger_size=trigger_size,
                        target_label=target_label,
                        poison_fraction=poison_fraction
                    )
                    xs.append(x_poisoned)
                    y_ref = y_poisoned  # Party1 的标签为后门目标标签
                else:
                    xs.append(x_p)
            
            # 前向 + 反向
            splitnn.do_zero_grads()
            logits = splitnn(xs)
            loss = criterion(logits, y_ref)
            running_loss += loss.item()
            loss.backward()
            splitnn.doStep()

        avg_loss = running_loss / num_batches
        train_acc = vfl_eval(trainloaders, splitnn, device)
        test_acc = vfl_eval(testloaders, splitnn, device)
        backdoor_acc = compute_backdoor_rate(
            splitnn=splitnn,
            testloaders2=testloaders,
            target_party=target_party,
            trigger_value=trigger_value,
            trigger_size=trigger_size,
            target_label=target_label,
            device=device
        )
        with open(f'./res/{dataset_name}/bk/vfl_training_log.csv', 'a') as f:
            f.write(f"{epoch},{avg_loss:.4f},{train_acc:.2f},{test_acc:.2f},{backdoor_acc:.2f},{time.time() - time_start:.2f}\n")

        print(f"Epoch {epoch}/{epochs} done. "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, "
              f"Backdoor Acc: {backdoor_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model": splitnn.state_dict(),
                "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
                "server_opt": splitnn.server_opt.state_dict(),
                "epoch": epoch
            }
            torch.save(ckpt, f"./res/{dataset_name}/bk/vfl_best_model.pth")
            print(f"New best model saved with Test Acc: {best_acc:.2f}%")