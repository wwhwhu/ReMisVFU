# vfl/un_core_kd.py
import copy, torch, torch.nn.functional as F
import time
import os
from torch import nn, optim
from utils.bk import compute_backdoor_rate
from utils.evaluate import vfl_eval
from vfl.core import SplitNN

trigger_value = 1.0
trigger_size  = 2
target_label  = 5
poison_frac   = 0.10
target_party  = 1

def vfu_kd_unlearn(
    splitnn_teacher,            # trained full SplitNN (frozen)
    forget_party_idx,           # [int] index to remove
    stored_embeds,              # list of tuples (emb_f, emb_ret, label)
    in_dims_retained,           # concat dim of retained embeddings
    out_dim,                    # num classes
    device,
    retain_trainloaders,
    retain_testloaders,
    alpha=0.7,                  # KD trade-off
    T=4.0,                      # temperature
    lr=1e-3,
    in_channel=None,           # not used (clients already trained)
    epochs=15,
    save_dir=None,            # not used in this function
    ):

    # ---------- 1. build student SplitNN (without P_f) ----------
    student = SplitNN(
        num_parties=len(splitnn_teacher.client_list)-1,
        in_channel=in_channel,        # not used (clients already trained)
        out_dim=out_dim
    ).to(device)

    # copy retained clients' weights
    s_idx = 0
    for p_idx, client in enumerate(splitnn_teacher.client_list):
        if p_idx in forget_party_idx:   # skip forgotten party
            continue
        student.client_list[s_idx].part.load_state_dict(client.part.state_dict())
        s_idx += 1

    # re-initialise active-side first FC layer to new input size
    act_teach = splitnn_teacher.server.partC
    act_stu   = student.server.partC

    # ---------- 通用地定位第一层 Linear ----------
    def first_linear(module: nn.Module):
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear):
                return name, m                   # 返回层名字与对象
        raise ValueError("No nn.Linear layer found in server sub-network")

    layer_name, old_fc = first_linear(act_teach)
    print(f"[INFO] Found first Linear layer: {layer_name}")
    # ---------- 构造新的首层 ----------
    new_fc = nn.Linear(in_dims_retained, old_fc.out_features).to(device)
    print(f"[INFO] Replacing {layer_name} with new Linear layer: {new_fc}")

    parent_mod = act_stu
    sub_names  = layer_name.split(".")          # 处理 'block1.fc' 这种嵌套
    for n in sub_names[:-1]:
        parent_mod = getattr(parent_mod, n)
    setattr(parent_mod, sub_names[-1], new_fc)

    # ---------- 2. prepare optimiser ----------
    for p in student.parameters(): 
        p.requires_grad_(True)
    opt = optim.Adam(student.parameters(), lr=lr)
    best_acc = 0
    loss_csv_path = save_dir + '/vfl_training_log.csv'
    with open(loss_csv_path, 'w') as f:
        f.write('epoch,loss,train_acc,test_acc,backdoor_acc,time\n')
    # ---------- 3. distillation loop ----------
    for ep in range(epochs):
        time_0 = time.time()
        total_loss = 0.
        for emb_f, emb_ret, y in stored_embeds:
            emb_f, emb_ret_list, y = emb_f.to(device), [e.to(device) for e in emb_ret], y.to(device)
            emb_all_list = [emb_f] + emb_ret_list
            # teacher forward
            with torch.no_grad():
                zT = act_teach(emb_all_list) / T

            # student forward
            zS = act_stu(emb_ret_list) / T

            loss_pred = F.cross_entropy(zS * T, y)
            loss_kd   = F.kl_div(
                F.log_softmax(zS, dim=1),
                F.softmax(zT, dim=1),
                reduction='batchmean'
            )
            loss = (1-alpha)*loss_pred + alpha*loss_kd

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"[KD-Unlearning] epoch {ep+1}/{epochs}   loss={total_loss/len(stored_embeds):.4f}")
        train_acc = vfl_eval(retain_trainloaders, student, device)
        test_acc  = vfl_eval(retain_testloaders,  student, device)
    
        backdoor_acc = compute_backdoor_rate(
            splitnn=student,
            testloaders2=retain_testloaders,
            target_party=target_party,
            trigger_value=trigger_value,
            trigger_size=trigger_size,
            target_label=target_label,
            device=device
        )
        with open(loss_csv_path, 'a') as f:
            f.write(f"{ep+1},{total_loss/len(stored_embeds):.4f},{train_acc:.2f},{test_acc:.2f},{backdoor_acc:.2f},{time.time() - time_0:.2f}\n")
        print(f"[KD-Unlearning] Epoch {ep+1}/{epochs} completed. "
              f"Loss: {total_loss/len(stored_embeds):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, "
              f"Backdoor Acc: {backdoor_acc:.2f}%")
        os.makedirs(save_dir, exist_ok=True)
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存整个模型
            torch.save({
                "model": student.state_dict(),
                "epoch": ep + 1,
                "train_acc": train_acc,
                "test_acc": test_acc
            }, os.path.join(save_dir, 'student_best_model.pth'))
            print(f"[INFO] New best model saved to {save_dir}/student_best_model.pth")
    return student
