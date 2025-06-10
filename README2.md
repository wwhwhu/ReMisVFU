# 设计文档：在垂直联邦学习中对客户端特征进行 RMU 操作

本文档描述了如何在\*\*垂直联邦学习（Vertical Federated Learning, VFL）\*\*场景中，对某个客户端（Party）的输出特征执行 **Representation Misdirection for Unlearning（RMU）** 操作，以达到在聚合前“遗忘”该客户端有害或后门信息的目的，同时尽量保留模型对无害数据的性能。整个设计可分为以下几部分：

1. **背景与目标**
2. **系统架构概览**
3. **模型与数据流程设计**
4. **RMU 原理及其在 VFL 中的映射**
5. **详细实现步骤**
6. **关键代码/伪码逻辑**
7. **超参数与训练策略**
8. **注意事项与扩展思考**

---

## 1. 背景与目标

### 1.1 垂直联邦学习（VFL）简介

* 在 VFL 中，不同的参与方（Party）拥有同一批用户的不同视角特征（feature）。
* 例如，Party A 拥有某一组用户的“左半张图像”或“某几列特征”；Party B 拥有同一批用户的“右半张图像”或“其余列特征”。
* 每个 Party 本地将自己的特征通过神经网络编码成一段中间表示，再将中间表示发送到中央服务器（Server），Server 将所有 Party 的中间表示拼接后进行下游任务（分类、回归等）。
* VFL 保证了各方间只交换特征表示，不直接共享原始数据，从而满足数据隐私／合规需求。

### 1.2 RMU 简要回顾

* **RMU（Representation Misdirection for Unlearning）** 是一种“基于表示层面”的遗忘方法，目的在于：

  1. 让模型对“有害/后门”数据或维度上的表示尽量偏离原本正常分布，使得后续层难以正确利用这些表示进行预测或攻击。
  2. 同时限制模型对“无害”数据或维度上的表示尽量与原始（冻结）模型一致，保持其对正常任务的能力。
* 具体做法：在第 ℓ 层隐藏表示上构造两部分损失，`Forget Loss` 通过将有害样本的中间表示拉向某个随机向量并放大；`Retain Loss` 则在无害样本上保持当前模型与冻结模型的表示接近。最后只对第 ℓ 层及其相邻若干层的参数做微调。

### 1.3 本设计目标

* 在已进行了多轮常规 VFL 训练后（得到了一个全局模型），需要将某个 Party（称作“遗忘方”或 “forget party”）中含有后门/有害信息的特征在下次聚合前彻底“抹除”。
* 具体来说，需要对“聚合前送给 Server 的 Party 输出中间表示”做 RMU 操作，使得该 Party 的后门信息在 Server 拼接时已被扰乱、无法再起作用；与此同时，尽量保留模型对其他无害分支及数据的性能。
* 最终在下次聚合时，污染特征已经被消弱或失效，从而得到一个“有害信息被抑制、无害性能几近不变”的新全局模型。

---

## 2. 系统架构概览

整体流程可以分为三个阶段：

1. **阶段一：常规 VFL 训练**

   * 各方（Party0\~PartyN-1）基于本地数据进行多轮局部训练（SplitNN），Server 端按 FedAvg 方式聚合，得到初始全局模型 $M_{\text{global}}$。

2. **阶段二：本地 RMU 遗忘（仅针对遗忘方 Party0）**

   * 在 Party0 端：拷贝全局模型为 `M_frozen`（冻结用于对比）和 `M_updated`（待优化）；
   * 对 Party0 的特征编码网络部分（即客户端输出到 Server 的那段中间表示）执行 RMU，迭代若干轮，使其在“有害样本”上的中间表示被强行拉向随机向量，同时在“保留样本”上的中间表示尽量贴近 `M_frozen` 的输出；
   * 更新后得到 `M_updated^0`，只修改了 Party0 端中间表示相关的参数，其余部分（Server 端 & 其他 Party）均保持不变。

3. **阶段三：新一轮联邦聚合（Re-aggregation）**

   * 将 Party0 用 RMU 优化后的本地模型 `M_updated^0` 与其他 Party 原先的本地模型一起进行一次 FedAvg 聚合，得到新的全局模型 `M_new_global`。
   * `M_new_global` 不再携带 Party0 中的后门信息，但对无害样本仍保持良好性能。

下面分模块详细说明每一步的设计要点。

---

## 3. 模型与数据流程设计

### 3.1 网络结构（SplitNN）

1. **Party 端（ClientA / ClientB / …）**

   * 每个 Party 都拥有同一套编码器架构（称作 `FirstNet`），但输入数据不同。
   * `FirstNet` 示例结构：

     ```text
     Input:  [batch, 1, 28, 14]             # 28×28 MNIST 图像的左右半张
       └─ Conv2d(1 → 32, kernel=3, pad=1)   # → [batch,32,28,14]
           └─ ReLU
           └─ MaxPool2d(2×2)               # → [batch,32,14,7]
           └─ Conv2d(32 → 64, kernel=3, pad=1)  # → [batch,64,14,7]
               └─ ReLU
               └─ MaxPool2d(2×2)           # → [batch,64,7,3]
     Output: features  [batch, 64, 7, 3]
     ```
   * 该输出就是 Party 端真实要发送给 Server 的“中间表示”。如果需要维度一致，还可进一步做 `view(batch, -1)` 拉平为 `[batch, d]`，其中 $d = 64\times7\times3$。

2. **Server 端（Server.partC）**

   * 收到来自所有 Party 的中间表示后，把它们在通道/向量维度上拼接，再用若干全连接层做下游任务。
   * 例如，若有两 Party，各自中间表示为 `[batch,64,7,3]` → 拉平为 `[batch,1344]`，拼接后形状为 `[batch,2688]`，再通过两层全连接：

     ```text
     Input:  [batch, 2688]
       └─ Linear(2688 → 128)
           └─ ReLU
           └─ Linear(128 → 10)
     Output: logits [batch, 10]
     ```

3. **SplitNN 封装**

   * 定义一个 `SplitNN` 类，持有 `clientA`、`clientB`、`server` 三个子模块，以及对应的优化器。
   * `forward(xA, xB)`：先分别调用 `clientA(xA)`、`clientB(xB)` 得到各自特征，再传入 `server(featA, featB)` 得到最终分类 logits。
   * `zero_grads()` / `step()`：统一对三端网络做梯度清零与参数更新。

### 3.2 数据流示例

1. **常规训练阶段**

   * Party0 加载本地训练集 `X0`（左半图像）与标签 `y0`，BatchSize=128；Party1 加载 `X1`（右半图像）与 `y0`（与 Party0 同批次的标签）；
   * 每个 Party 本地执行若干 Epoch 的“SplitNN 局部训练”：

     ```text
     for each local_batch:
         feat0 = clientA(x0_batch)
         feat1 = clientB(x1_batch)
         logits = server(feat0, feat1)
         loss = CrossEntropy(logits, y0_batch)
         loss.backward()
         // 梯度流经 server → clientA / clientB
         clientA_opt.step()
         clientB_opt.step()
         server_opt.step()
     ```
   * 每个 Party 的本地训练结束后，都得到一个本地完整 SplitNN 模型，将其参数 `state_dict` 上传给 Server。
   * Server 执行 FedAvg：平均所有 Party 模型的对应参数，得到新的全局 `M_global`，并广播给所有 Party。

2. **RMU 遗忘阶段（Party0）**

   * Party0 收到 `M_global`，将其复制为 `M_frozen`（只用于对比，不再更新）和 `M_updated`（待优化）；
   * 只对 `M_updated.clientA.partA` 中的最后一层（或最后三层）参数解冻，保留其他层以及 Party1/Server 部分全部冻结；
   * Party0 准备两类本地数据：

     * **遗忘数据集** $D_{\text{forget}}^0$：包含 Party0 中需要遗忘的有害/后门样本（右半图上含有后门模式）；
     * **保留数据集** $D_{\text{retain}}$：可以直接使用 Party1 的整个本地干净数据（不含后门），或者使用预先准备的公共无后门数据。
   * 每轮迭代，先在 $D_{\text{forget}}^0$ 上计算 `Forget Loss`，再在 $D_{\text{retain}}$ 上计算 `Retain Loss`，混合优化得到 `M_unlearn^0`。此时 `M_unlearn^0.clientA.partA` 的参数已被 RMU 调整，Server & Party1 部分无变化。

3. **重新聚合阶段**

   * Party0 把 `M_unlearn^0.state_dict()` 返回给 Server；
   * 其他 Party（Party1…PartyN-1）直接使用上次训练的本地模型参数；
   * Server 对所有 Party 的参数做一次 FedAvg，得到 `M_new_global`；
   * `M_new_global` 即为“Party0 已被遗忘有害表示、其他 Party 保持正常”的新全局模型，可以用于下游评估或下一个联邦训练周期。

---

## 4. RMU 原理及其在 VFL 中的映射

### 4.1 RMU 原理回顾

1. **Forget Loss**

   * 选定第 $\ell$ 层隐藏表示 $h^{(\ell)}_{\text{updated}}(x)$ 作为“遗忘目标”。
   * 生成随机单位向量 $u\in\mathbb{R}^d$，并指定放大系数 $c$。
   * 对遗忘数据集 $D_{\text{forget}}$ 的样本 $x$，计算：

     $$
       L_{\text{forget}}
       = \mathbb{E}_{x\sim D_{\text{forget}}}\,\frac{1}{d}\Bigl\lVert\,h^{(\ell)}_{\text{updated}}(x)\;-\;c\,u\Bigr\rVert_2^2.
     $$
   * 效果：让有害样本经过第 $\ell$ 层后，其表示偏离正常分布，后续层无法正确处理。

2. **Retain Loss**

   * 在保留数据集 $D_{\text{retain}}$ 上，将当前模型与冻结模型第 $\ell$ 层隐藏表示尽量对齐：

     $$
       L_{\text{retain}}
       = \mathbb{E}_{x\sim D_{\text{retain}}}\,\frac{1}{d}\Bigl\lVert\,h^{(\ell)}_{\text{updated}}(x) \;-\; h^{(\ell)}_{\text{frozen}}(x)\Bigr\rVert_2^2.
     $$
   * 效果：在“无害”数据上不改变表示，最大限度保留原始能力。

3. **总损失**

   $$
       L = L_{\text{forget}} \;+\; \alpha\,L_{\text{retain}},
   $$

   * $\alpha>0$ 控制保留损失比重。若 $\alpha$ 较大，则更重视保留能力；若 $\alpha$ 较小，则更重视“遗忘”效果。

4. **参数更新范围**

   * 仅对第 $\ell$ 层及其相邻（$\ell-1,\ell-2$）层的参数开放梯度更新。
   * 其他层（如 $\ell+1,\ell+2,\ldots$ 或更深的全连接层）均冻结，不参与优化。

### 4.2 在 VFL 场景的映射

1. **遗忘层选定**

   * 在 SplitNN 中，Party 端的 `FirstNet` 最后一层（即第 2 个 Conv→池化后输出）恰好是“Party 将要传给 Server 的输出特征”。
   * 如 `FirstNet` 输出 shape 为 `[batch,64,7,3]`，拉平后为 `[batch,1344]`，这就是我们要做 RMU 的“第 $\ell$ 层”表示。
   * 因此，$\ell=2$，对应 `ClientA.partA.conv2` 及其池化输出。

2. **随机向量 $u$ 与放大系数 $c$**

   * 计算 `d = 64*7*3 = 1344`。
   * 生成 `u = torch.rand(1344).to(device); u = u / ||u||₂`。
   * 选定 `c`（如 2.0），即期望最终遗忘向量为 `c * u`。

3. **只更新 Party0 部分**

   * 在 `M_updated` 中，将 `M_updated.clientA.partA.conv2` 及其周边参数（如 `conv2` 的权重和偏置）标记 `requires_grad=True`；其余所有参数（包括 `conv1`、`clientB`、`server`）都置为 `requires_grad=False`。
   * 这样整个优化过程仅在 Party0 `conv2` 这层（或适当再放开 $\ell-1,\ell-2$）做梯度下降／上升。

4. **遗忘数据集 $D_{\text{forget}}^0$**

   * 对 Party0 本地训练集进行筛选，挑出那些带有后门模式或有害信息的样本，构成 `DataLoader`。可以直接基于前面后门注入时生成的 `poisoned_dataloader_train_right[0]` 或类似结构的“右半图 + 标签”。
   * 遗忘数据集中只需要包含 Party0 端的输入（例如，右半图为 “有毒/后门” 部分，与其对应的左半图可随意或也全部加到 `xA_forget` 中，记得保持两侧对齐）。

5. **保留数据集 $D_{\text{retain}}$**

   * 直接使用 Party1 … PartyN-1 的本地干净数据（左右两半整合后分割）即可，保证这些数据与 Party0 有害样本语义不重叠。
   * 也可以选用公开无后门的子集，比如 MNIST 干净测试集，以便更好地保持全局无害能力。

6. **Loss 计算与优化**

   * **Forget Loss**：在 Party0 端，以批次方式取出 `xA_forget_batch`（党 0 的有害图像输入，注意取左半或右半均可，只要中间表示层包含后门特征即可），然后调用 `party0_output_feature(M_updated, xA_forget_batch)` 得到 `[B,1344]` 的表示 `h_u`。

     ```python
     target = (c * u).unsqueeze(0).expand(h_u.size(0), -1)  # [B, 1344]
     loss_forget = ((h_u - target) ** 2).mean()
     ```
   * **Retain Loss**：在 Party1（及其余 Party）端的无害数据 `xR_batch` 上，分别计算 `h_u_retained = party0_output_feature(M_updated, xR_batch)` 和 `h_f_retained = party0_output_feature(M_frozen, xR_batch)`，然后

     ```python
     loss_retain = ((h_u_retained - h_f_retained) ** 2).mean()
     ```
   * **总 Loss**：

     ```python
     loss_total = loss_forget + alpha * loss_retain
     optimizer.zero_grad()
     loss_total.backward()
     optimizer.step()
     ```
   * 上述 `backward()` 仅会影响 Party0 `conv2` 等级别的参数，Server 与其他 Party 不受影响。

---

## 5. 详细实现步骤

以下从准备工作到最终聚合，逐步给出详细步骤。

### 5.1 常规 VFL 训练（准备全局模型）

1. **数据加载与预处理**

   * 加载 MNIST 原始数据：`(x_raw, y_raw), (x_raw_test, y_raw_test) = load_mnist(raw=True)`。
   * 调用 `preprocess(x_raw, y_raw)`，得到 `X_train`（归一化成 float32）、`y_train`（one-hot／标签）。同理获取 `X_test, y_test`。
   * 打乱训练集索引：

     ```python
     idx = np.arange(n_train); np.random.shuffle(idx)
     X_train = X_train[idx]; y_train = y_train[idx]
     ```

2. **垂直划分数据**

   * 设定 `num_parties = N`，其中 `party0` 将是要遗忘方。
   * 计算 `num_erased = int(60000 / N * scale)`，`num_per_party = int((60000 - num_erased) / (N - 1))`。
   * `X_party0 = X_train[:num_erased]`, `y_party0 = y_train[:num_erased]`。
   * `X_rest = X_train[num_erased : num_erased + num_per_party*(N-1)]`,`y_rest = y_train[...]`。
   * 以函数 `make_dataloaders(X, y, batch_size)` 将每个子集分割为左右两半（`X[:, :, :14], X[:, :, 14:]`），并构造 PyTorch `DataLoader`：

     ```python
     def make_dataloaders(X, y, batch_size=128):
         X_left  = np.expand_dims(X[:, :, :14], axis=1)   # [m,1,28,14]
         X_right = np.expand_dims(X[:, :, 14:], axis=1)   # [m,1,28,14]
         y_cls   = np.argmax(y, axis=1).astype(int)       # [m]
         ds_left  = TensorDataset(torch.Tensor(X_left), torch.Tensor(y_cls).long())
         ds_right = TensorDataset(torch.Tensor(X_right), torch.Tensor(y_cls).long())
         return DataLoader(ds_left, batch_size=batch_size, shuffle=True), \
                DataLoader(ds_right, batch_size=batch_size, shuffle=True)
     ```
   * Party0 得到 `loader0_left, loader0_right`；其余 N-1 个 Party 在 `X_rest`、`y_rest` 上按每个子集各自调用 `make_dataloaders`。

3. **初始化全局模型**

   * 对于每个 Party，都分别实例化 `ClientA()`, `ClientB()`, `Server()`：

     ```python
     clientA = ClientA().to(device)
     clientB = ClientB().to(device)
     server  = Server().to(device)
     ```
   * 三端各用 Adam 优化器：`optA`, `optB`, `optS`。
   * 构造 `SplitNN(clientA, clientB, server, optA, optB, optS).to(device)`，保存到列表 `party_models[i]`。

4. **多轮本地训练与聚合（FedAvg）**

   ```python
   global_state = party_models[0].state_dict()

   fusion = FusionAvg(num_parties)

   for round_idx in range(num_rounds):
       # 4.1 每个 Party 加载全局参数
       for i in range(num_parties):
           party_models[i].load_state_dict(global_state)

       # 4.2 每个 Party 本地训练
       local_states = []
       for i in range(num_parties):
           model_i = party_models[i]
           trainL = trainloaders_left[i]
           trainR = trainloaders_right[i]
           updated_model_i, loss_i = local_trainer.train(model_i, trainL, trainR)
           local_states.append(updated_model_i.state_dict())

       # 4.3 FedAvg 聚合
       new_global_state = fusion.average_selected_models(local_states)
       global_state = copy.deepcopy(new_global_state)

       # 4.4 全局模型评估（可选）
       eval_model = make_fresh_splitnn()
       eval_model.load_state_dict(global_state)
       acc = evaluate(testloader_left, testloader_right, eval_model)
       print(f"联邦轮次 {round_idx+1}，全局测试精度 {acc:.2f}%")
   ```

5. **保存全局模型**

   ```python
   torch.save(global_state, "vfl_global_before_unlearn.pth")
   ```

### 5.2 本地 RMU 遗忘（针对 Party0）

1. **加载并冻结全局模型**

   ```python
   M_global = make_fresh_splitnn()           # 新建一套结构
   M_global.load_state_dict(torch.load("vfl_global_before_unlearn.pth"))
   M_frozen = copy.deepcopy(M_global).to(device)
   M_updated = copy.deepcopy(M_global).to(device)
   ```

2. **只对 Party0 的“最后一层 Conv2d”参数开放梯度**

   * 遍历 `M_updated.clientA.partA` 中所有参数，若其 `name` 属于 `conv2.weight`, `conv2.bias`（或所在 Block）则 `param.requires_grad=True`，其余均设 `False`；同时 `M_updated.clientB.partB` 与 `M_updated.server.partC` 全部 `requires_grad=False`。

   ```python
   for name, param in M_updated.clientA.partA.named_parameters():
       if "conv2" in name:
           param.requires_grad = True
       else:
           param.requires_grad = False
   # 其他 Party/Server 全部冻结
   for name, param in M_updated.clientB.partB.named_parameters():
       param.requires_grad = False
   for name, param in M_updated.server.partC.named_parameters():
       param.requires_grad = False

   optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, M_updated.parameters()), lr=1e-4)
   ```

3. **生成随机单位向量 u**

   ```python
   d = 64 * 7 * 3   # conv2 输出维度
   u = torch.rand(d, device=device)
   u = u / torch.norm(u, p=2)
   c = 2.0          # 放大系数，可调
   alpha = 0.5      # Retain Loss 权重，可调
   ```

4. **准备遗忘集与保留集 DataLoader**

   * **遗忘集**（Party0 的有害样本）：

     ```python
     forget_loader = poisoned_dataloader_train_right[0]  # 若已在上一阶段保存
     ```
   * **保留集**（其他 Party 的无害数据）：

     ```python
     # 将所有 Party1~PartyN-1 的干净训练数据拼在一起
     retain_dataset = ConcatDataset([
         DatasetFromLoader(trainloaders_left[1], trainloaders_right[1]),
         DatasetFromLoader(trainloaders_left[2], trainloaders_right[2]),
         # … 直至第 N-1 个
     ])
     retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)
     ```

     *注：`DatasetFromLoader` 可定义为：把两个 Loader 同步 zip，每次输出 `[x_left, x_right, label]`，然后只取 `x_left` 或 `x_right` 中的左半输入放到 Party0 端。*

5. **定义获取 Party0 中间表示的函数**

   ```python
   def get_party0_feature(model, xA):
       """
       输入 xA: [B,1,28,14]（Party0 左半或右半图像）
       返回 Party0 最后一层 conv2→pool 后的 flattened 特征 [B, d]
       """
       feat_map = model.clientA.partA(xA)    # [B,64,7,3]
       B, C, H, W = feat_map.shape
       return feat_map.view(B, C*H*W)        # [B, 1344]
   ```

6. **多轮 RMU 优化**

   ```python
   num_unlearn_epochs = 10
   for epoch in range(num_unlearn_epochs):
       # 6.1 Forget Loss
       M_updated.train()
       total_f_loss = 0.0
       for xA_forget, _ in forget_loader:
           xA_forget = xA_forget.to(device)    # [B,1,28,14]
           h_u = get_party0_feature(M_updated, xA_forget)  # [B, d]
           target = (c * u).unsqueeze(0).expand(h_u.size(0), -1)  # [B, d]
           loss_f = ((h_u - target)**2).mean()

           optimizer.zero_grad()
           loss_f.backward()
           optimizer.step()
           total_f_loss += loss_f.item()
       avg_f_loss = total_f_loss / len(forget_loader)

       # 6.2 Retain Loss
       total_r_loss = 0.0
       M_updated.train()
       for xR, _ in retain_loader:
           xR = xR.to(device)    # [B_r,1,28,14]
           h_u2 = get_party0_feature(M_updated, xR)      # [B_r, d]
           with torch.no_grad():
               h_f2 = get_party0_feature(M_frozen, xR)   # [B_r, d]
           loss_r = ((h_u2 - h_f2)**2).mean()

           optimizer.zero_grad()
           loss_r.backward()
           optimizer.step()
           total_r_loss += loss_r.item()
       avg_r_loss = total_r_loss / len(retain_loader)

       print(f"[RMU Epoch {epoch+1}/{num_unlearn_epochs}] ForgetLoss={avg_f_loss:.4f}, RetainLoss={avg_r_loss:.4f}")
   ```

7. **得到 Party0 遗忘后模型 `M_unlearn^0`**

   ```python
   M_unlearn_party0 = M_updated
   ```

### 5.3 新一轮联邦聚合

1. **收集各方最新参数**

   * Party0：`M_unlearn_party0.state_dict()`
   * Party1\~PartyN-1：使用上一阶段常规训练得到的本地模型参数 `party_models[i].state_dict()`

2. **FedAvg 聚合**

   ```python
   party_states = []
   party_states.append(M_unlearn_party0.state_dict())
   for i in range(1, num_parties):
       # 直接读取之前保存好的 Party i 本地模型
       state_i = torch.load(f"party_{i}_local_after_round{last_round}.pth")
       party_states.append(state_i)

   fusion = FusionAvg(num_parties)
   new_global_state = fusion.average_selected_models(party_states)

   M_new_global = make_fresh_splitnn()
   M_new_global.load_state_dict(new_global_state)
   torch.save(new_global_state, "vfl_global_after_unlearn.pth")
   ```

3. **评估新全局模型**

   ```python
   eval_model = make_fresh_splitnn().to(device)
   eval_model.load_state_dict(torch.load("vfl_global_after_unlearn.pth"))
   acc_clean = evaluate(testloader_left, testloader_right, eval_model)
   print(f"遗忘后全局模型——干净测试精度：{acc_clean:.2f}%")

   # 若有后门测试集，可同理评估后门误差
   acc_poison = evaluate(testloader_poison_left, testloader_poison_right, eval_model)
   print(f"遗忘后全局模型——后门测试精度：{acc_poison:.2f}%")
   ```

---

## 6. 关键代码/伪码逻辑

**1. SplitNN 定义**（省略；参见前部已给出 `vfl_training.py`）

**2. 获取 Party0 中间表示**

```python
def get_party0_feature(model: SplitNN, xA: torch.Tensor) -> torch.Tensor:
    """
    获取 Party0 最后一层 (conv2→pool) 的输出并 flatten。
    Input: 
        model: SplitNN 实例，已设置为 .train() 或 .eval() 模式
        xA: [B,1,28,14] Party0 的输入
    Output:
        feat_flat: [B, d]，其中 d = 64*7*3
    """
    feat_map = model.clientA.partA(xA)    # [B,64,7,3]
    B, C, H, W = feat_map.shape
    return feat_map.view(B, C*H*W)        # [B, 1344]
```

**3. RMU 训练循环**

```python
# 参数预设
d = 64 * 7 * 3
u = torch.rand(d, device=device)
u = u / torch.norm(u, p=2)   # 单位向量
c = 2.0
alpha = 0.5
num_unlearn_epochs = 10

# 冻结 M_frozen、复制 M_updated
M_frozen = copy.deepcopy(M_global).to(device)
M_updated = copy.deepcopy(M_global).to(device)

# 冻结除 conv2 以外的所有参数
for name, param in M_updated.clientA.partA.named_parameters():
    if "conv2" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
for name, param in M_updated.clientB.partB.named_parameters():
    param.requires_grad = False
for name, param in M_updated.server.partC.named_parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, M_updated.parameters()), lr=1e-4)

# 提前定义数据加载
forget_loader = poisoned_dataloader_train_right[0]
retain_loader = build_retain_loader_other_parties()

for epoch in range(num_unlearn_epochs):
    # 1)计算 Forget Loss
    total_f_loss = 0.0
    M_updated.train()
    for xA_forget, _ in forget_loader:
        xA_forget = xA_forget.to(device)
        h_u = get_party0_feature(M_updated, xA_forget)         # [B, d]
        target = (c * u).unsqueeze(0).expand(h_u.size(0), -1)  # [B, d]
        loss_f = ((h_u - target) ** 2).mean()

        optimizer.zero_grad()
        loss_f.backward()
        optimizer.step()
        total_f_loss += loss_f.item()
    avg_f_loss = total_f_loss / len(forget_loader)

    # 2)计算 Retain Loss
    total_r_loss = 0.0
    M_updated.train()
    for xR, _ in retain_loader:
        xR = xR.to(device)
        h_u2 = get_party0_feature(M_updated, xR)        # [B, d]
        with torch.no_grad():
            h_f2 = get_party0_feature(M_frozen, xR)     # [B, d]
        loss_r = ((h_u2 - h_f2) ** 2).mean()

        optimizer.zero_grad()
        loss_r.backward()
        optimizer.step()
        total_r_loss += loss_r.item()
    avg_r_loss = total_r_loss / len(retain_loader)

    print(f"[RMU Epoch {epoch+1}/{num_unlearn_epochs}] "
          f"ForgetLoss={avg_f_loss:.4f}, RetainLoss={avg_r_loss:.4f}")
```

**4. 重新聚合**

```python
# 收集 Party0 & 其他 Party 最新状态
party_states = []
party_states.append(M_updated.state_dict())  # Party0 已遗忘模型

for i in range(1, num_parties):
    state_i = torch.load(f"party_{i}_local_after_round{last_round}.pth")
    party_states.append(state_i)

fusion = FusionAvg(num_parties)
new_global_state = fusion.average_selected_models(party_states)

M_new_global = make_fresh_splitnn().to(device)
M_new_global.load_state_dict(new_global_state)
torch.save(new_global_state, "vfl_global_after_unlearn.pth")
```

---

## 7. 超参数与训练策略

1. **放大系数 $c$**

   * 直接决定 “Forget Loss” 中将有害表示推向何种幅度的随机向量。
   * 若 $c$ 太小，表示扰动不足，遗忘效果弱；若 $c$ 太大，表示过于偏离，可能导致梯度不稳定或丧失对无害数据的兼容性。
   * 常见取值 $c\in[1.0, 3.0]$，可通过网格搜索或在“小规模实验”中调优。

2. **Retain 权重 $\alpha$**

   * 衡量“保留无害能力”与“遗忘有害能力”的权衡。
   * 如果 $\alpha$ 太大，模型倾向于不改变无害表示，可能无法彻底抹除有害信息；若 $\alpha$ 太小，容易牺牲正常任务性能，使全局模型退化。
   * 常见 $\alpha\in[0.1, 1.0]$ 区间尝试。

3. **遗忘轮数（`num_unlearn_epochs`）**

   * 通常 5–20 个 epoch 即可显著削弱有害表示。
   * 如果太少，遗忘不彻底；太多，会明显影响正常性能，还会浪费资源。

4. **学习率（`lr`）**

   * 由于只有少数层可训练，学习率一般要比常规训练小一个量级，例如常规训练用 1e-3，则 RMU 用 1e-4 或 1e-5。
   * 若学习率过大，容易使有害表示出现过度偏移甚至梯度爆炸；若过小，遗忘效率低。

5. **批次大小**

   * `forget_loader`、`retain_loader` 的批次大小可设 64–256，根据显存与数据量决定。
   * 如果有害样本少，批次也可设置小一些，以增加迭代次数。

6. **遗忘层选择**

   * 本例中采用 Party0 `conv2`（第二层卷积）输出做遗忘，也可以尝试放开最后几层：

     * 如果将 `conv1` 与 `conv2` 都开放，会产生更强的表示变动，但可能更快失去正常能力；
     * 如果只开放 `conv2`，更温和，但可能需要多次迭代才能看见明显效果。

---

## 8. 注意事项与扩展思考

1. **保留集质量**

   * 保留集 `D_{\text{retain}}` 最好与有害/后门样本语义差异较大。
   * 在实践中可直接用“其他 Party 的全部数据”或“公共干净样本”，确保其不包含 Party0 的任何后门信息。

2. **投影/范数约束（可选）**

   * 如果发现 `h_u` 与 `c·u` 的距离过大，导致模型参数漂移过多，不妨做一次“参数投影”：将 Party0 第 $\ell$ 层参数向量投回到某个 $\ell_2$ 范数球面。
   * 实现思路：

     ```python
     w_new = parameters_to_vector(conv2.parameters())      # [P]
     w_ref = parameters_to_vector(conv2_frozen.parameters())
     delta = w_new - w_ref
     if torch.norm(delta, p=2) > R_thresh:
         delta = delta / torch.norm(delta, p=2) * R_thresh
         w_proj = w_ref + delta
         vector_to_parameters(w_proj, conv2.parameters())
     ```
   * 该策略可提高稳定性，避免一次迭代过度偏离。

3. **多 Party 同时遗忘**

   * 若存在多个 Party 都需要同时遗忘不同有害信息，可依次或并行对各 Party 分别执行 RMU，然后再统一聚合。
   * 聚合时保持每个 Party 都使用其“最新遗忘后模型”。

4. **不同后门分布的迭代**

   * 若有多种后门模式（如生物安全、有害化合物、网络安全等），可在一个 RMU 训练循环中交叉：

     * 先在类别 A 后门数据上执行几次 Forget Loss，再在类别 B 后门数据上执行几次；交替进行，最后整体收敛。

5. **度量与评估**

   * 需要在以下两类测试集上分别评估：

     1. **干净测试集**（所有 Party 合并后的无后门 MNIST 测试集），评估普通分类性能下降幅度；
     2. **后门测试集**（Party0 端有后门测试数据 + 其他 Party 干净数据），评估后门激活率是否显著降低。

6. **可复现性**

   * 固定随机种子：

     ```python
     torch.manual_seed(42)
     np.random.seed(42)
     ```
   * 在多个实验中保持 `u`、`c`、`alpha`、`lr` 不变，便于对比效果。

---

## 小结

本文档详细设计了在垂直联邦学习场景中，如何对**聚合前的客户端输出特征**执行 RMU 遗忘操作。核心在于：

1. **识别 Party0 输出特征对应的隐藏层 $\ell$**（本例中是第二层卷积池化后输出）。
2. **构造遗忘集与保留集**，分别计算 `Forget Loss` 与 `Retain Loss`：

   * Forget Loss：将有害样本在第 $\ell$ 层的表示推向随机向量 $c·u$。
   * Retain Loss：在无害样本上保持当前与冻结模型的表示一致。
3. **仅对 Party0 的该隐藏层相关参数做梯度优化**，冻结其他网络部分。
4. **多轮迭代后得到 Party0 遗忘后模型 `M_unlearn^0`**，将其与其他 Party 模型一起做一次 FedAvg 聚合，得到新的全局模型 `M_new_global`。
5. **评估新全局模型在干净测试集和后门测试集上的表现**，验证遗忘效果与无害性能保留程度。

通过上述设计，可以在 VFL 中实现“聚合前对客户端输出特征进行 RMU 遗忘”，从而有效剔除特定 Party 的有害信息，而不影响其他 Party 及全局模型对无害任务的性能。
