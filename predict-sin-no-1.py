import torch
import torch.nn as nn
import numpy as np

# 定义超参数
input_size = 1
hidden_size = 32
output_size = 1
num_epochs = 300
learning_rate = 0.01

# 准备训练数据
time_steps = np.linspace(0, 2*np.pi, 400)
data = np.sin(time_steps)
x = data[:-1]
y = data[1:]

future = 400
future_time_steps = np.linspace(0, 2*np.pi *(1+future/400), 400+future)[1:]


# 转换数据类型为Tensor
x_tensor = torch.Tensor(x.reshape(1, -1, 1))
# print(x_tensor.shape)
y_tensor = torch.Tensor(y.reshape(1, -1, 1))

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers = 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.linear(out)
        return out, hn

# 初始化模型参数
model = RNN(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 初始化隐藏状态
    h0 = torch.zeros(1, 1, hidden_size)
    # 前向传播
    output, hn = model(x_tensor, h0)
    # 计算损失函数
    loss = criterion(output, y_tensor)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 打印训练结果
    if (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 使用模型进行预测
with torch.no_grad():
    # 初始化隐藏状态
    h0 = torch.zeros(1, 1, hidden_size)
    # 指定预测长度

    # 预测结果
    pred_list = []
    for i in range(future):
        x = torch.Tensor(np.array([[y[-1]]]))
        output, hn = model(x.reshape(1, 1, -1), h0)
        y_pred = output.detach().numpy().reshape(-1)
        pred_list.append(y_pred)
        y = np.append(y, y_pred)

print(future_time_steps.shape)
print(y.shape)
    # print(pred_list)
# 绘制训练数据和预测结果
import matplotlib.pyplot as plt
plt.plot(time_steps[:-1], data[:-1], 'r', label='Training data')
plt.plot(future_time_steps, y, 'b', label='Predictions')
plt.legend(loc='best')
plt.show()
