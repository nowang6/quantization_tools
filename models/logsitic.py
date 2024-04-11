import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 构造一些虚拟数据
np.random.seed(42)
x = np.random.rand(100, 1)
y = (2 * x + 0.5 + np.random.randn(100, 1) * 0.1) > 1

# 转换为 PyTorch 的 Tensor
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 初始化模型和优化器
model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    outputs = model(x_tensor)
    # 计算损失
    loss = criterion(outputs, y_tensor)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'logistic_model.pth')
print("Model saved successfully!")
