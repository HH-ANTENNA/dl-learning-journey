# 汇总代码（精简）
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms

# =================== 重要知识点与常见易错点 ===================
# 重要知识点:
# - 张量形状:      输入/输出形状必须匹配；nn.Linear 不会自动展平输入，需用 nn.Flatten() 或手动 reshape。
# - 数值稳定性:    计算 softmax 时应做减最大值：X - X.max(dim=1, keepdim=True).values，避免指数爆炸。
# - 交叉熵:        nn.CrossEntropyLoss 要传 logits（未 softmax 的输出），内部会做 log_softmax。自行实现时注意对数与索引的正确使用。
# - loss reduction: reduction='mean' 与 'sum' 会改变梯度尺度；使用自定义优化器（如手写 sgd）时需按 batch_size 调整学习率或归一化梯度。
# - autograd:      调用 loss.backward() 后，梯度累加在 .grad 上，更新后要 zero_grad() 或手动清零。手写优化器中用 with torch.no_grad() 更新并置零 .grad。
# - DataLoader:    Windows 下应在 if __name__ == '__main__': 中创建 DataLoader 或设置 mp.set_start_method('spawn', force=True)；num_workers>0 可加速加载但需注意跨平台问题。
# - 训练/评估:     训练时用 model.train()；评估时用 model.eval() 并配合 torch.no_grad()，这会影响 Dropout/BatchNorm 的行为并节省内存。
#
# 常见易错点（与建议）:
# - 形状不匹配:    忘记展平输入或 label 维度与预测不一致，导致广播错误或损失计算错误。建议打印 .shape 或写断言。
# - Softmax 溢出:  直接对大数做 exp 会溢出，务必先减去每行最大值或使用 logsumexp / torch.log_softmax。
# - 交叉熵输入错误: 传入 nn.CrossEntropyLoss 的应为 logits（FloatTensor），标签为 LongTensor 的类别索引；否则报错或结果不对，建议用 y = y.long()。
# - 学习率与 batch_size: 手写 sgd 若未按 batch_size 归一，梯度尺度会不一致。使用 reduction='none' 时注意如何聚合梯度（.mean() / .sum()）。
# - 忘记清零梯度:  每次迭代忘记 optimizer.zero_grad() 或 param.grad.zero_() 会导致梯度累加。训练循环里总是 zero_grad()。
# - DataLoader 在 Windows 卡住: 如果脚本未放入 if __name__ == '__main__'，创建 DataLoader 时可能卡住。调试时先尝试 num_workers=0。
# - 评估时未禁用梯度: 未使用 torch.no_grad() 或未切换 eval 模式会导致 BatchNorm/Dropout 行为不一致和额外内存占用。评估时应禁用梯度并调用 model.eval()。
# - CrossEntropyLoss(reduction='none') 未聚合: 若不对 loss 做 .mean() 或 .sum()，在调用 .backward() 前要明确聚合，否则梯度比例可能出错。
# ================================================================

# ---------- 数据与工具 ----------
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# ---------- 线性回归（从零开始） ----------
def linreg(X, w, b): return X @ w + b
def squared_loss(y_hat, y): return (y_hat - y.reshape(y_hat.shape))**2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad / batch_size
            p.grad.zero_()

# 使用合成数据测试线性回归
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr, num_epochs = 0.03, 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = squared_loss(linreg(X,w,b), y)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features,w,b), labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')

# ---------- Softmax 回归：从零开始 ----------
def softmax(X):
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True).values)  # numeric stable
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def net_from_scratch(X, W, b):
    X = X.reshape(-1, W.shape[0])
    return softmax(X @ W + b)

# ---------- Softmax 回归：使用 torch.nn ----------
net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 10))
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss_fn = nn.CrossEntropyLoss(reduction='none')  # logits -> softmax + loss
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 加载 Fashion-MNIST（示例）
trans = transforms.ToTensor()
train_ds = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
test_ds  = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

# Windows 下建议在脚本入口处设置启动方式（若多进程 DataLoader）
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

train_iter = data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
test_iter  = data.DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4)

def evaluate_accuracy(model, data_iter):
    if isinstance(model, nn.Module):
        model.eval()
    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            metric[0] += accuracy(model(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]

# 训练循环（通用）
def train_epoch(model, train_iter, loss_fn, optimizer):
    if isinstance(model, nn.Module):
        model.train()
    metric = [0.0, 0.0, 0.0]  # loss_sum, acc_sum, n
    for X, y in train_iter:
        y_hat = model(X)
        l = loss_fn(y_hat, y)
        optimizer.zero_grad()
        l.mean().backward()
        optimizer.step()
        metric[0] += float(l.sum())
        metric[1] += accuracy(y_hat, y)
        metric[2] += y.numel()
    return metric[0] / metric[2], metric[1] / metric[2]
