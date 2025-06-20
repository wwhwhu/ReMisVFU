# ---------------- fast retrain settings -----------------
from copy import deepcopy
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.bk import add_trigger_to_right, compute_backdoor_rate
from utils.data_split import load_and_split_data, remove_from_index
from utils.evaluate import vfl_eval
from vfl.core import SplitNN
import torch.nn as nn

num_party         = 3
forget_party_idx  = [1]                 # 要遗忘的 Party
dataset_list      = ["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100"]
dataset_idx       = 4                  # 选择数据集
dataset_name      = dataset_list[dataset_idx]

device = torch.device(
    f"cuda:{dataset_idx % 4}" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
trainloaders_all, testloaders_all, in_channel, out_dim = \
        load_and_split_data(dataset_name, num_parties=num_party)
trainloaders = remove_from_index(forget_party_idx, trainloaders_all)
testloaders  = remove_from_index(forget_party_idx, testloaders_all)

num_parties = 3
forget_party      = 1                # P_f index
rho_switch        = 0.7              # fraction of epochs for RAdam
radam_lr          = 0.001
sgdm_lr           = 0.001
momentum          = 0.999
epoch             = 20
pre_ckpt_path     = f'./res/{dataset_name}/bk/vfl_best_model.pth'

# ------------------------------------------------------------
# helper: locate the first nn.Linear inside server subnet and
#         replace it with (new_in, old_out) fresh layer
# ------------------------------------------------------------
def replace_first_fc(server_subnet: nn.Module,
                     new_in: int,
                     device: torch.device) -> None:
    """
    Walk through 'server_subnet', find the first nn.Linear, and
    replace it with a new nn.Linear(new_in, old_out).
    """
    for name, module in server_subnet.named_modules():
        if isinstance(module, nn.Linear):
            print(f"[INFO] Re-init server fc '{name}': "
                  f"{module.in_features} → {new_in}")
            parent = server_subnet
            for n in name.split('.')[:-1]:           # drill down
                parent = getattr(parent, n)
            old_out = module.out_features
            setattr(parent, name.split('.')[-1],
                    nn.Linear(new_in, old_out).to(device))
            return
    raise RuntimeError("No nn.Linear layer found in server subnet")

# ------------------------------------------------------------
# 1) build new SplitNN (K-1 retained parties)
# ------------------------------------------------------------
splitnn = SplitNN(num_parties - 1,
                  in_channel=in_channel,
                  out_dim=out_dim)
splitnn.toDevice(device)
criterion = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# 2) copy retained-party & server weights from checkpoint
# ------------------------------------------------------------
pre_state = torch.load(pre_ckpt_path, map_location=device)["model"]
new_state = deepcopy(splitnn.state_dict())

for k in new_state.keys():
    print(f"Copying {k} ...")
    if "client_list" in k:
        new_idx = int(k.split(".")[1])
        # original index in teacher (skip forget_party)
        old_idx = new_idx if new_idx < forget_party else new_idx + 1
        src_k = k.replace(f"client_list.{new_idx}",
                          f"client_list.{old_idx}")
    else:                      # server-side weights (may need resize later)
        src_k = k
    if src_k in pre_state and pre_state[src_k].shape == new_state[k].shape:
        new_state[k] = pre_state[src_k].clone()

splitnn.load_state_dict(new_state, strict=False)
print("✔ copied retained client weights & server weights (except first fc)")

# ------------------------------------------------------------
# 3) infer retained-embedding dimension with one dummy forward
# ------------------------------------------------------------
with torch.no_grad():
    dummy_inputs = []
    for p_idx, dl in enumerate(trainloaders):
        x0, _ = next(iter(dl))             # one mini-batch is enough
        dummy_inputs.append(x0[:1].to(device))
    # local encoders forward
    embeds = [c.part(x) for c, x in zip(splitnn.client_list, dummy_inputs)]
    print(f"[INFO] dummy forward done, {len(embeds)} retained parties, shapes: f{[e.shape for e in embeds]}")
    retained_dim = torch.cat(
        [e.flatten(start_dim=1) for e in embeds], dim=1
    ).shape[1]

print(f"[INFO] retained embedding dim = {retained_dim}")

# ------------------------------------------------------------
# 4) rebuild server’s first nn.Linear for new input size
# ------------------------------------------------------------
replace_first_fc(splitnn.server.partC, retained_dim, device)

print("✔ SplitNN ready — shape mismatch resolved, fast-retrain can start!")

# ---------- build hybrid optimiser lists ----------------
def build_radam(model):  # wrapper for clarity
    return optim.RAdam(model.parameters(), lr=radam_lr)
def build_sgdm(model):
    return optim.SGD(model.parameters(), lr=sgdm_lr, momentum=momentum)

splitnn.client_opt_list = [build_radam(c) for c in splitnn.client_list]
splitnn.server_opt      = build_radam(splitnn.server)

switch_epoch  = int(rho_switch * epoch)
print(f'RAdam → SGDM will switch at epoch {switch_epoch}')

best_acc = 0.0
poison_fraction = 0.1    # 每个 batch 中毒比例
target_label = 5         # 后门触发时的目标标签
trigger_value = 1.0      # 触发器像素值（最亮白）
trigger_size = 2         # 触发器大小 2*2

target_party = 1         # 对 Party1 注入后门
# --------------- training loop (fast retrain) -----------
for ep in range(epoch):
    t0 = time.time()
    # ---------- optimiser switch ----------
    if ep == switch_epoch:
        splitnn.client_opt_list = [build_sgdm(c) for c in splitnn.client_list]
        splitnn.server_opt      = build_sgdm(splitnn.server)
        print('⇨ switched optimiser to SGDM')

    # ---------- identical VFL loop ----------
    loaders_iter = [iter(dl) for dl in trainloaders]
    num_batches  = min(len(dl) for dl in trainloaders)
    running_loss = 0.0

    for _ in tqdm(range(num_batches), desc=f'Epoch {ep+1}/{epoch}'):
        # synchronous batch
        xs, y_ref = [], None
        for p, it in enumerate(loaders_iter):
            x_p, y_p = next(it)
            _, y_ref = add_trigger_to_right(
                x_batch       = x_p,
                y_batch       = y_p,
                trigger_value = trigger_value,
                trigger_size  = trigger_size,
                target_label  = target_label,
                poison_fraction = poison_fraction
            )
            y_ref = y_ref.to(device)  # only active party has labels
            xs.append(x_p.to(device))
        # forward & backward
        splitnn.do_zero_grads()
        logits  = splitnn(xs)
        loss    = criterion(logits, y_ref)
        running_loss += loss.item()
        loss.backward()
        splitnn.doStep()

    loss_epoch = running_loss / num_batches
    # ---- eval & log (same as your original) ----
    train_acc   = vfl_eval(trainloaders, splitnn, device)
    test_acc    = vfl_eval(testloaders,  splitnn, device)
    backdoor_acc= compute_backdoor_rate(splitnn, testloaders,
                                        target_party, trigger_value,
                                        trigger_size, target_label, device)
    phase = 'RAdam' if ep < switch_epoch else 'SGDM'
    with open(f'./res/{dataset_name}/fast_retrain/log.csv','a') as f:
        f.write(f'{ep+1},{phase},{loss_epoch:.4f},'
                f'{train_acc:.2f},{test_acc:.2f},'
                f'{backdoor_acc:.2f},{time.time()-t0:.2f}\n')
    print(f'E{ep+1:02d} {phase:5s} | loss {loss_epoch:.4f} | '
          f'TestAcc {test_acc:.2f}% | TrainAcc {train_acc:.2f}% | '
            f'BackdoorAcc {backdoor_acc:.2f}% | time {time.time()-t0:.2f}s')
    if test_acc > best_acc:
        best_acc = test_acc
        # save the best model
        ckpt = {
            "model": splitnn.state_dict(),
            "client_opts": [opt.state_dict() for opt in splitnn.client_opt_list],
            "server_opt":  splitnn.server_opt.state_dict(),
            "epoch": ep + 1
        }
        torch.save(ckpt, f'./res/{dataset_name}/fast_retrain/best_model.pth')
        print(f'✔ Best model saved at epoch {ep+1}, acc {best_acc:.2f}%')
