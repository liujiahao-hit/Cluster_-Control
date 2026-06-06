# -*- coding: utf-8 -*-
"""
基于神经网络的无人机集群权重计算
"""

import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# ================= 论文要求的激活函数 =================

class Tansig(nn.Module):
    """
    隐藏层激活函数：Tansig(o) = 2/(1+exp(-2o)) - 1 (2-19)
    """
    def forward(self, x):
        return 2 / (1 + torch.exp(-2 * x)) - 1


class Purelin(nn.Module):
    """
    输出层激活函数：Purelin(o) = o (2-20)
    """
    def forward(self, x):
        return x


# ================= 神经网络结构 =================

class FlockingNN(nn.Module):
    """
    神经网络结构：
    - 输入层：6×N = 30维
    - 4个隐藏层：每层6×N = 30个神经元，Tansig激活
    - 输出层：N×N = 25维，Purelin激活
    """
    
    def __init__(self, N):
        super(FlockingNN, self).__init__()
        
        input_dim = 6 * N
        hidden_dim = 6 * N
        output_dim = N * N
        
        # 4个隐藏层
        self.network = nn.Sequential(
            # 输入层到隐藏层1
            nn.Linear(input_dim, hidden_dim),
            Tansig(),
            
            # 隐藏层1到隐藏层2
            nn.Linear(hidden_dim, hidden_dim),
            Tansig(),
            
            # 隐藏层2到隐藏层3
            nn.Linear(hidden_dim, hidden_dim),
            Tansig(),
            
            # 隐藏层3到隐藏层4
            nn.Linear(hidden_dim, hidden_dim),
            Tansig(),
            
            # 隐藏层4到输出层
            nn.Linear(hidden_dim, output_dim),
            Purelin()
        )
    
    def forward(self, x):
        return self.network(x)


# ================= 数据处理 =================

class FlockingDataset(Dataset):
    """
    无人机集群数据加载器
    """
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# ================= 训练函数 =================

def train_neural_network():
    """
    训练神经网络
    - 数据归一化：每个维度归一化到[-1,1]
    - 训练算法：trainscg
    - epoch：1000
    - 损失函数：均方误差MSE
    - 数据集比例：训练集70%，验证集15%，测试集15%
    """
    
    N = 5  # 无人机数量
    
    # ============= 1. 加载训练数据 =============
    print("=" * 60)
    print("加载训练数据")
    print("=" * 60)
    
    # 查找最新的训练数据文件
    results_dir = 'results'
    data_files = [f for f in os.listdir(results_dir) if f.startswith('training_data_') and f.endswith('.pkl')]
    
    if not data_files:
        print("❌ 未找到训练数据文件，请先运行train_data_final.py生成数据")
        return
    
    # 选择最新的文件
    data_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_file = os.path.join(results_dir, data_files[0])
    
    print(f"加载数据文件：{latest_file}")
    
    with open(latest_file, 'rb') as f:
        training_data = pickle.load(f)
    
    inputs = np.array(training_data['inputs'])
    labels = np.array(training_data['labels'])
    
    print(f"数据规模：{len(inputs)}个样本")
    print(f"输入维度：{inputs.shape[1]}")
    print(f"输出维度：{labels.shape[1]}")
    
    # ============= 2. 数据归一化 =============
    print("\n" + "=" * 60)
    print("数据预处理（归一化到[-1,1]）")
    print("=" * 60)
    
    # 论文要求：只对输入数据的每个维度归一化到[-1,1]，标签不需要归一化
    scaler_input = MinMaxScaler(feature_range=(-1, 1))
    inputs_normalized = scaler_input.fit_transform(inputs)
    
    # 标签保持原始值
    labels_normalized = labels
    
    print(f"输入数据：最小值{inputs_normalized.min():.4f}，最大值{inputs_normalized.max():.4f}")
    print(f"标签数据：保持原始值，最小值{labels_normalized.min():.4f}，最大值{labels_normalized.max():.4f}")
    
    # ============= 3. 划分数据集 =============
    print("\n" + "=" * 60)
    print("划分数据集（70%训练 | 15%验证 | 15%测试）")
    print("=" * 60)
    
    # 创建数据集
    dataset = FlockingDataset(inputs_normalized, labels_normalized)
    
    # 划分比例
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 随机划分
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    print(f"训练集：{train_size}个样本")
    print(f"验证集：{val_size}个样本")
    print(f"测试集：{test_size}个样本")
    
    # ============= 4. 初始化模型 =============
    print("\n" + "=" * 60)
    print("初始化神经网络模型")
    print("=" * 60)
    
    model = FlockingNN(N)
    print(model)
    
    # ============= 5. 设置训练参数 =============
    # (2-21)：均方误差
    def mse_loss(output, target):
        """
        (2-21)MSE损失函数
        """
        # 计算每个样本的MSE：(1/输出维度) * sum((label - output)^2)
        return torch.mean((1.0 / target.size(1)) * torch.sum((target - output) ** 2, dim=1))
    
    criterion = mse_loss
    
    # trainscg算法 
    # PyTorch中没有CG优化器，使用LBFGS作为替代（最接近的拟牛顿方法）
    # LBFGS与trainscg同属梯度类优化算法，适合全批次训练
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, line_search_fn='strong_wolfe')
    
    num_epochs = 1000  # 论文要求：1000次迭代
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    
    # ============= 6. 开始训练 =============
    print("\n" + "=" * 60)
    print("开始训练神经网络")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for batch_inputs, batch_labels in train_loader:
            def closure():
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                return loss.item()
            
            # 调用optimizer.step，不直接获取返回值
            optimizer.step(closure)
            
            # 重新计算损失并累加
            with torch.no_grad():
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                train_loss += loss.item() * batch_inputs.size(0)
        
        # 计算平均训练损失
        train_loss /= train_size
        train_losses.append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_inputs.size(0)
        
        # 计算平均验证损失
        val_loss /= val_size
        val_losses.append(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

    # ============= 7. 测试模型 =============
    print("\n" + "=" * 60)
    print("测试模型性能")
    print("=" * 60)
    
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item() * batch_inputs.size(0)
    
    test_loss /= test_size
    
    print(f"测试集损失：{test_loss:.8f}")
    
    # ============= 8. 保存模型 =============
    print("\n" + "=" * 60)
    print("保存模型和数据")
    print("=" * 60)
    
    # 保存模型
    model_save_path = f"results/flocking_nn_model_{N}uav.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_input': scaler_input,
        'N': N,
        'input_dim': 6 * N,
        'output_dim': N * N
    }, model_save_path)
    
    print(f"模型保存：{model_save_path}")
    
    # ============= 9. 可视化训练过程 =============
    print("\n" + "=" * 60)
    print("可视化训练过程")
    print("=" * 60)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_epochs), train_losses, label='训练损失')
    plt.plot(range(num_epochs), val_losses, label='验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差 (MSE)')
    plt.title('神经网络训练过程')
    plt.legend()
    plt.grid(True)
    
    # 保存训练曲线
    loss_plot_path = 'results/training_loss_curve.png'
    plt.savefig(loss_plot_path)
    plt.close()
    
    print(f"✅ 训练曲线保存：{loss_plot_path}")
    
    print("\n" + "=" * 60)
    print("神经网络训练完成！")
    print("=" * 60)
    print(f"模型文件：{model_save_path}")
    print(f"测试损失：{test_loss:.8f}")
    print(f"数据规模：{len(inputs)}个样本")
    
    return model, scaler_input


# ================= 预测函数 =================

def predict_weights(model, scaler_input, input_data):
    """
    使用训练好的模型预测权重矩阵
    
    参数：
        model: 训练好的神经网络模型
        scaler_input: 输入数据的归一化器
        input_data: 输入向量 (6×N维)
        
    返回：
        weights_matrix: 预测的权重矩阵 (N×N维)
    """
    
    model.eval()
    
    # 归一化输入
    input_normalized = scaler_input.transform(input_data.reshape(1, -1))
    
    # 转换为PyTorch张量
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
    
    # 标签不需要归一化，直接返回
    return output.numpy().reshape(5, 5)  # 转换为N×N矩阵


if __name__ == "__main__":
    train_neural_network()