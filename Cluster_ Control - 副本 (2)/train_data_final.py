# -*- coding: utf-8 -*-
"""
神经网络训练集生成器
调用主函数run_simulation()生成训练数据
"""

import numpy as np
import os
import pickle
from datetime import datetime
import math
import sys

# 找到主函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入主函数
from main import run_simulation


def generate_training_data():
    """
    生成神经网络训练集
    - 重复实验次数：N_repeat = 100次
    - 模拟时间：Tmax = 50秒
    - 采样时间：ts = 0.5秒（主函数已设置）
    - 无人机数量：N = 5架
    - 初始位置范围：Qx∈[0,30]，Qy∈[75,150]
    - 距离约束：任意两架无人机水平距离<最大通信距离Dc=20m
    """
    
    # ============= 参数设置 =============
    N_repeat = 100  # 重复实验次数：100次
    N = 5           # 无人机数量：5架
    Tmax = 50.0     # 模拟时间：50秒
    Dc = 20.0       # 最大通信距离：20m
    
    # 存储训练数据
    training_data = {
        'inputs': [],     # 输入向量：每架无人机的[Qx, Qy, Vx, Vy, efv_x, efv_y]
        'labels': [],     # 标签向量：MOPSO优化的权重矩阵
        'params': {
            'N_repeat': N_repeat,
            'N': N,
            'Tmax': Tmax,
            'Dc': Dc
        }
    }
    
    print("=" * 60)
    print("神经网络训练集生成器")
    print(f"参数：{N_repeat}次实验 | {Tmax}秒仿真 | {N}架无人机")
    print("=" * 60)
    
    # 生成训练数据
    for repeat_idx in range(N_repeat):
        print(f"\n=== 实验 {repeat_idx+1}/{N_repeat} ===")
        
        # 生成满足距离约束的随机初始位置
        valid_positions = False
        while not valid_positions:
            # 随机生成初始位置
            initial_positions = []
            valid_positions = True
            
            for i in range(N):
                # Qx∈[0,30]，Qy∈[75,150]
                qx = np.random.uniform(0, 30)
                qy = np.random.uniform(75, 150)
                initial_positions.append([qx, qy])
            
            # 检查距离约束：任意两架无人机距离<Dc
            for i in range(N):
                for j in range(i+1, N):
                    dx = initial_positions[i][0] - initial_positions[j][0]
                    dy = initial_positions[i][1] - initial_positions[j][1]
                    distance = math.hypot(dx, dy)
                    if distance >= Dc:
                        valid_positions = False
                        break
                if not valid_positions:
                    break
        
        # 调用主函数进行仿真
        print(f"  初始位置已生成，开始仿真...")
        swarm, params, simulation_data = run_simulation(
            initial_positions=initial_positions,
            max_simulation_time=Tmax
        )
        
        # 提取训练数据
        num_steps = len(simulation_data['time_steps'])
        
        for step in range(num_steps):
            # 获取当前时间步数据
            positions = simulation_data['drone_positions'][step]
            velocities = simulation_data['drone_velocities'][step]
            efv_list = simulation_data['efv_list'][step]
            weights_matrix = simulation_data['weights_matrix'][step]
            
            # 生成输入向量
            input_vec = []
            for i in range(N):
                qx, qy = positions[i]
                vx, vy = velocities[i]
                efv_x, efv_y = efv_list[i]
                input_vec.extend([qx, qy, vx, vy, efv_x, efv_y])
            
            # 生成标签向量（权重矩阵扁平化）
            label_vec = weights_matrix.flatten().tolist()
            
            # 保存数据
            training_data['inputs'].append(input_vec)
            training_data['labels'].append(label_vec)
        
        print(f"  实验完成，生成 {num_steps} 个样本")
    
    # 保存训练数据
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/training_data_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(training_data, f)
    
    # 统计信息
    total_samples = len(training_data['inputs'])
    expected_samples = N_repeat * int(Tmax / 0.5)
    
    print("\n" + "=" * 60)
    print("训练集生成完成！")
    print(f"统计信息：")
    print(f"   - 实验次数：{N_repeat}/{N_repeat}")
    print(f"   - 样本总数：{total_samples}")
    print(f"   - 预期样本：{expected_samples}")
    print(f"   - 输入维度：{len(training_data['inputs'][0])}")
    print(f"   - 标签维度：{len(training_data['labels'][0])}")
    print(f"数据文件：{filename}")
    print("=" * 60)
    
    return training_data


if __name__ == "__main__":
    generate_training_data()