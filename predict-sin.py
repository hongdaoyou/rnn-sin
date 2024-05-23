import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


# 批次
batchNum = 1

# 输入数据的维度
input_size = 1

# 隐藏层的个数
num_layers = 8

# 隐藏层的维度
hidden_size = 10

# 初始,隐藏状态
# h0 = np.zeros((num_layers, batchNum, hidden_size))
h0 = torch.zeros(num_layers, batchNum, hidden_size)

# 定义,步长 (每一步,多长)
piNum = 400
stepSize = piNum / 2*np.pi

# 目录的地址
DIR_PATH = '/home/hdy/test/python/'

# 模型参数的文件
Model_Path = 'rnn_sin.pth'


# 产生,数据
# torch.arange(0, 2*np.pi, step=piNum+1)

# 训练的区间
trainX = torch.linspace(0, 2*np.pi, steps=piNum)
trainY = torch.sin(trainX )

trainX1 = trainX[:-1]

trainY1 = trainY[:-1] # 当前的值
trainY2 = trainY[1:]  #接下来的值

# 最开始预测的点  它的前一个
predictYStart = torch.tensor([trainY[-2] ])

# 预测的区间 . 往后延伸
predictX = 2*np.pi + torch.linspace(0, 2*np.pi, steps=piNum)

# 预测的结果,存储的位置
predictResult = []

# 定义,rnn神经网络.  将,rnn的输出,用全连接层,进行转换
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 循环层
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True )

        #  全连接层
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, h0):
        output, h1 = self.rnn(x, h0)
        
        # 取出,最后一个时间步
        # lastItem = output[:,-1, :]
        # print(lastItem.shape)

        # 线性映射
        out = self.linear(output)
        # print(out.shape)

        return out, h1

# 定义,损失函数, 优化器
# loss = torch.MSELoss()
criterion = nn.MSELoss()

# 训练
def train():
    # 创建,网络
    model = Net()
    
    # 定义,优化器
    # optimizer = nn.GPD()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    # 将其,输入参数, 1批次, 每句话,一个单词 
    trainY1_tmp = trainY1.reshape(batchNum,  -1, 1)
    print(trainY1_tmp.shape )

    # 真实值,转换
    trainY2_tmp = trainY2.reshape(batchNum, -1, 1)
    # print(trainY1_tmp.shape )

    # 训练的次数
    for eporch in range(trainNum+1):
        output,_ = model(trainY1_tmp, h0)
        # print(output.shape)

        # 计算损失值
        loss = criterion(output, trainY2_tmp)

        optimizer.zero_grad()

        # 反向传递
        loss.backward()

        # 更新
        optimizer.step()

        if eporch % 10 == 0:
            print(f"{eporch} loss={loss}")

    #  保存到,文件中
    torch.save(model.state_dict(), DIR_PATH + Model_Path);




# 预测
def predict():
    model = Net()
    model.load_state_dict(torch.load(DIR_PATH + Model_Path))

    model.eval()

    global predictYStart
    global predictResult
    h00 = h0
    predictYStart = predictYStart.unsqueeze(0).unsqueeze(0)

    # with autoload.no_grad
    with torch.no_grad():
        # 预测n次
        for i in range(piNum):
            # predictYStart,h00 = model(predictYStart, h00)
            predictYStart,_ = model(predictYStart, h00)

            val = predictYStart.item()
            print(val)
            predictResult.append(val)
            
            #  重新预测下一个

    plt.plot(predictX, predictResult )
    plt.show()

# 训练的次数
trainNum = 10
# train()

predict()



