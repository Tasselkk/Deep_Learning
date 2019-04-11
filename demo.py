import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
input_size = 1#输入大小
output_size = 1#输出大小
num_epochs = 60#迭代次数
learning_rate = 0.001#学习率

# Toy dataset  1. 准备数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
                    
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

model=nn.Linear(input_size,output_size)#线性模型
criterion=nn.MSELoss()#最小平方误差
optimizer=torch.optim.SGD(model.parameters(),learning_rate)#随机梯度下降优化方法
loss_dict=[]
#train　训练数据
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors  准备tensor的训练数据和标签
    inputs = torch.from_numpy(x_train)#numpy数据转为tensor
    targets = torch.from_numpy(y_train)

    # Forward pass  5.2 前向传播计算网络结构的输出结果
    outputs = model(inputs)
    # 5.3 计算损失函数
    loss = criterion(outputs, targets)#损失函数
    
    # Backward and optimize 反向传播更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    # 可选 打印训练信息和保存loss
    loss_dict.append(loss.item())
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph 画出原y与x的曲线与网络结构拟合后的曲线
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# 画loss在迭代过程中的变化情况
plt.plot(loss_dict, label='loss for every epoch')
plt.legend()
plt.show()
