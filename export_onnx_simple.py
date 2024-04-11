import torch.onnx
import torch.nn as nn

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 创建模型实例
model = LogisticRegression()

# 加载模型权重
model.load_state_dict(torch.load("logistic_model.pth"))

# 输入数据的示例
example_input = torch.tensor([[0.5]], dtype=torch.float32)

# 导出模型为 ONNX 格式
torch.onnx.export(model,                   # 导出的模型
                  example_input,          # 示例输入数据
                  "logistic_model.onnx",  # 导出的文件路径
                  export_params=True,     # 将模型权重导出为参数文件
                  opset_version=11,       # ONNX 版本
                  do_constant_folding=True,  # 是否进行常量折叠
                  input_names=['input'],  # 输入的名称
                  output_names=['output']  # 输出的名称
                  )
print("Model exported to ONNX successfully!")
