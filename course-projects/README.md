# 🎓 深度学习基础 · 课程项目集

> 深圳大学 2025-2026 · 深度学习基础 · 黄浩  
> 从 MLP 回归到 CNN 图像分类，从手写 RNN/LSTM 到时序预测与情感分析，再到图像描述生成。

---

## 📋 课程总览

```
实验一                  实验二                  实验三                  期中项目
MLP 房价预测            CNN CIFAR-10            RNN/LSTM 双任务         图像描述生成
   │                      │                      │                      │
   ├─ 数据预处理          ├─ Basic CNN            ├─ 时序预测 (ETTh1)     ├─ Flickr8k 数据集
   ├─ K-Fold CV           ├─ Optimized CNN        │   ├─ RNN (from scratch)│─ Encoder-Decoder
   ├─ BatchNorm+Dropout   ├─ ResNet16 (from scratch)│  └─ LSTM (from scratch)├─ Attention
   ├─ LR Schedule         ├─ 消融对比              ├─ 情感分析 (IMDb)      ├─ 中英双语
   └─ Early Stopping      └─ 混淆矩阵分析          │   ├─ RNN              └─ BLEU 评估
                                                    │   └─ BiLSTM
                                                    └─ 对比分析
```

---

## 🧪 实验一：神经网络训练与优化

### 任务
Kaggle House Prices — 用 PyTorch MLP 做房价回归预测。

### 模型架构

```
Input(325) → Linear(512) → BN → ReLU → Dropout(0.3)
           → Linear(128) → BN → ReLU → Dropout(0.3)
           → Linear(1)
```

### 关键技术

| 技术 | 作用 | 效果 |
|------|------|------|
| BatchNorm | 稳定训练、加速收敛 | 减少内部协变量偏移 |
| Dropout(0.3) | 防止过拟合 | 提升泛化 |
| K-Fold CV (5折) | 充分利用数据 | 集成预测更稳健 |
| ReduceLROnPlateau | 自适应学习率 | Val Loss 停滞时 lr×0.5 |
| Early Stopping (patience=15) | 防止过拟合 | 自动保存最优模型 |
| Weight Decay (1e-5) | L2 正则化 | 抑制权重膨胀 |
| Log Transformation | 标签正态化 | 稳定 MSE 训练 |
| StandardScaler | 特征标准化 | 加速收敛 |

### 特征工程
- 删除缺失率 >90% 的特征
- 类别特征 → 独热编码 (→ 325 维)
- 数值特征 → 中位数/0 填充
- 5 折集成预测取均值

---

## 🧪 实验二：CNN 图像分类进化

### 任务
CIFAR-10 图像分类 — 从基础 CNN 到 ResNet 的完整进化。

### 三部曲对比

| | Basic CNN | Optimized CNN | ResNet16 |
|------|-----------|---------------|----------|
| **架构** | Conv×2 + FC×2 | Conv×2 + BN + Dropout + FC×2 | 残差块 ×6 + Skip Connection |
| **归一化** | ❌ | BatchNorm2d | BatchNorm2d |
| **正则化** | ❌ | Dropout(0.3) | Dropout(0.25) |
| **LR 调度** | ❌ 固定 0.008 | StepLR (×0.9/5epoch) | CosineAnnealing |
| **权重衰减** | ❌ | 1e-4 | 5e-4 |
| **优化器** | SGD+momentum | SGD+momentum | AdamW |
| **参数量** | 173,742 | 173,846 | 175,258 |
| **测试准确率** | 69.26% | 75.84% | **83.30%** |
| **过拟合** | ⚠️ 严重 (epoch 10+) | ✅ 缓解 | ✅ 无 |

### 关键发现

1. **BN + Dropout** 是基础标配：原始 CNN 从 epoch 10 开始训练损失降到 0.15 而测试损失飙升到 1.76 → 典型过拟合
2. **LR 衰减** 让后期更稳：StepLR 每 5 轮 ×0.9 避免了 SGD 震荡
3. **ResNet 残差连接** 在大约参数量下提升近 8%：Skip Connection 让深层网络更容易优化
4. **手写 ResNet16** 而非调包：从 BasicBlock 到 `_make_layer` 全手工实现

---

## 🧪 实验三：RNN/LSTM 从零实现 + 双任务

### 任务 A：油温时序预测 (ETTh1)

**亮点：不使用 `nn.RNN` / `nn.LSTM`，全部手写循环神经网络单元。**

| | RNN (from scratch) | LSTM (from scratch) |
|------|--------------------|---------------------|
| **隐藏维度** | 64 | 128 |
| **层数** | 2 | 2 |
| **参数量** | ~70K | ~300K |
| **RMSE** | **1.23°C** | 1.50°C |
| **R²** | **0.872** | 0.809 |

> RNN 在此任务上优于 LSTM：数据量不大时，LSTM 更多参数反而容易过拟合。

**手写实现的技术细节**：
- 预投影优化：`x @ W_ih` 一次性对所有时间步计算，避免逐步重复
- 正交初始化 `nn.init.orthogonal_`：缓解梯度消失
- LSTM 遗忘门偏置初始化为 1：帮助早期不遗忘
- 梯度裁剪 `max_norm=1.0`

### 任务 B：IMDb 情感分析 (50,000 条)

| | RNN | BiLSTM |
|------|-----|--------|
| **Embedding** | 100-dim | 100-dim |
| **方向** | 单向 | 双向 |
| **Pooling** | Last Step | Mean Pooling |
| **准确率** | 70.25% | **82.67%** |
| **F1** | 0.709 | **0.824** |

**关键优化**：
- 词汇表 10,000（去低频词）
- `max_len=150` 截断（平衡速度与覆盖）
- `ReduceLROnPlateau` + Early Stopping (patience=10)
- Mean Pooling 替代 Last Step：捕获全局语义

---

## 🏆 期中项目：图像描述生成 (Image Captioning)

### 任务
Flickr8k 中英双语图像描述 — 给定图像，生成中文/英文描述文本。

### 技术方案
- **数据集**：Flickr8k (8,000 图像，每图 5 条英文 + 5 条中文描述)
- **架构**：Encoder-Decoder with Attention
- **Encoder**：CNN 提取图像特征
- **Decoder**：RNN/LSTM 逐词生成描述
- **评估**：BLEU 评分

---

## 📊 技术栈全景

| 类别 | 技术 |
|------|------|
| **框架** | PyTorch (全部手写，最小化调包) |
| **模型** | MLP · CNN · ResNet · RNN · LSTM · BiLSTM |
| **训练技巧** | K-Fold · Early Stopping · LR Schedule · Weight Decay · Gradient Clipping |
| **正则化** | BatchNorm · Dropout · Weight Decay |
| **优化器** | SGD · Adam · AdamW |
| **数据** | 表格 (House Prices) · 图像 (CIFAR-10, Flickr8k) · 时序 (ETTh1) · 文本 (IMDb) |
| **评估** | MSE/RMSE/R² · Accuracy/Precision/Recall/F1 · Confusion Matrix · BLEU |

---

## 🔑 核心收获

1. **从零实现 > 调包**：手写 RNN/LSTM/ResNet 才能真正理解反向传播与梯度流
2. **过拟合是常态，优化是核心**：CNN 三部曲完整展示了"发现问题 → 诊断 → 解决"的工程思维
3. **数据模态全覆盖**：表格/图像/时序/文本 — 四种数据类型一次课覆盖
4. **Embedded AI 延伸**：课程知识直接应用到小车比赛（MaixCAM 分类模型）
