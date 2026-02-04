# 这是对第二章的总结Python代码文件，涵盖了2-1.ipynb到2-6.ipynb的知识点和易错点。
# 每个部分以注释形式解释知识点，并标注易错点。
# 使用PyTorch作为主要库，已安装torch, pandas, numpy, matplotlib等。

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import multinomial

# =============================================================================
# 2-1.ipynb: 张量基础操作
# 知识点: 张量的创建、形状、元素访问、广播机制、拼接、内存共享等。
# 易错点: reshape时维度计算错误；广播机制仅适用于尾部维度相等或为1的情况；原地操作会改变内存地址。
# =============================================================================

# 创建张量
x = torch.arange(12)  # 创建0到11的张量
print("x:", x)
print("x.shape:", x.shape)  # 形状
print("x.numel():", x.numel())  # 元素总数

# 广播机制示例
a = torch.arange(3).reshape((3, 1))  # (3,1)
b = torch.arange(2).reshape((1, 2))  # (1,2)
print("a + b:", a + b)  # 广播为(3,2)

# 拼接
X = x.reshape(3, 4)  # 重塑为(3,4)
Y = torch.ones_like(X)
print("torch.cat((X, Y), dim=0):", torch.cat((X, Y), dim=0))  # 沿dim=0拼接

# 内存共享易错点: X += Y 会改变X的内存地址
before = id(X)
X += Y
print("id(X) == before:", id(X) == before)  # False，原地操作改变了地址

# =============================================================================
# 2-2.ipynb: 数据预处理
# 知识点: 处理CSV数据、缺失值填充、独热编码、转换为张量。
# 易错点: 选择列时索引错误；fillna使用mean时忽略非数值列；get_dummies的dummy_na参数。
# =============================================================================

# 创建示例数据
data_file = 'house_tiny.csv'
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price,sale,like-value\n')
    f.write('NA,Pave,127500,NA,4.5\n')
    f.write('3,NA,106000,yes,4.5\n')
    f.write('5.0,NA,178100,no,3.5\n')
    f.write('4,NA,140000,yes,2\n')
    f.write('NA,Pave,2000,yes,2.8\n')

data = pd.read_csv(data_file)
print("原始数据:\n", data)

# 处理缺失值
inputs, outputs = data.iloc[:, [0,1,3,4]], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # 填充均值，易错: 只对数值列有效
print("填充后inputs:\n", inputs)

# 独热编码
inputs = pd.get_dummies(inputs, dummy_na=True)  # dummy_na=True处理NA
print("独热编码后:\n", inputs)

# 转换为张量
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy())
print("X, y:", X, y)

# =============================================================================
# 2-3.ipynb: 线性代数
# 知识点: 标量、向量、矩阵、张量运算、范数、点积、矩阵乘法。
# 易错点: 矩阵乘法用torch.mm；广播在除法时需keepdims；范数计算的ord参数。
# =============================================================================

# 标量、向量、矩阵、张量
x_scalar = torch.tensor(3.0)
x_vector = torch.arange(4)
A_matrix = torch.arange(20).reshape(5, 4)
X_tensor = torch.arange(24).reshape(2, 3, 4)

# 运算
print("A + B (Hadamard):", A_matrix + A_matrix.clone())
print("torch.mm(A, B):", torch.mm(A_matrix, torch.ones(4, 3)))  # 矩阵乘法

# 降维求和
print("A.sum(axis=0):", A_matrix.sum(axis=0))  # 每列和
print("A.sum(axis=1, keepdims=True):", A_matrix.sum(axis=1, keepdims=True))  # 保持维度

# 范数
u = torch.tensor([3.0, -4.0, -12])
print("L2范数:", torch.norm(u))
print("L1范数:", torch.abs(u).sum())

# 易错: 广播在除法中需keepdims避免形状不匹配
A = torch.arange(6).reshape(2, 3)
print("A / A.sum(axis=1, keepdims=True):", A / A.sum(axis=1, keepdims=True))

# =============================================================================
# 2-4.ipynb: 微积分基础
# 知识点: 数值极限、绘图函数、梯度规则。
# 易错点: numerical_lim的h选择影响精度；绘图时参数顺序。
# =============================================================================

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return 3 * x ** 2 - 4 * x

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

# 绘图示例 (简化)
x_vals = np.arange(0, 3, 0.1)
plt.plot(x_vals, f(x_vals), label='f(x)')
plt.plot(x_vals, 2 * x_vals - 3, label='Tangent')
plt.legend()
plt.show()

# =============================================================================
# 2-5.ipynb: 自动微分
# 知识点: requires_grad、backward、detach、控制流中的梯度。
# 易错点: 梯度累积需zero_；非标量backward需gradient参数；detach后梯度不传递。
# =============================================================================

x = torch.arange(4.0, requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
print("x.grad:", x.grad)  # 4*x

# 累积梯度易错
x.grad.zero_()
y = x.sum()
y.backward()
print("x.grad after sum:", x.grad)

# detach示例
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print("x.grad after detach:", x.grad)  # u, not 2*x

# 高阶导数
x = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
z = x ** 4
z.sum().backward(create_graph=True)
first_grad = x.grad.clone()
x.grad.zero_()
first_grad.sum().backward()
print("二阶导数:", x.grad)

# =============================================================================
# 2-6.ipynb: 概率基础
# 知识点: 多项分布采样、相对频率估计、大数定律。
# 易错点: sample()的参数；cumsum和keepdims的使用。
# =============================================================================

fair_probs = torch.ones([6]) / 6
print("单次采样:", multinomial.Multinomial(1, fair_probs).sample())
print("10次采样:", multinomial.Multinomial(10, fair_probs).sample())

# 估计概率
counts = multinomial.Multinomial(1000, fair_probs).sample()
estimates = counts / 1000
print("估计概率:", estimates)

# 累积实验
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# 绘图省略，易错: dim参数在cumsum和sum中