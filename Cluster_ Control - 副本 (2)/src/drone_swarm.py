"""
无人机集群模型实现，公式(2-3)到(2-7)
"""

import numpy as np
import math
from .drone import Drone

class DroneSwarm:
    """
    无人机集群类

    属性:
        num_drones: 无人机数量
        drones: 无人机对象列表
        D_c: 最大观察距离
        D_d: 期望邻居距离
        D_l1: 最小防撞距离
        obstacles: 障碍物列表
        history: 历史记录字典
    """

    def __init__(self, num_drones, initial_positions, initial_velocities):
        """
        初始化无人机集群

        参数:
            num_drones: 无人机数量
            initial_positions: 初始位置列表，每个元素是[x, y]
            initial_velocities: 初始速度列表，每个元素是[vx, vy]
        """
        self.num_drones = num_drones

        # 创建无人机对象
        self.drones = []
        for i in range(num_drones):
            drone_id = i + 1  # ID从1开始
            pos = initial_positions[i] if i < len(initial_positions) else [0.0, 0.0]
            vel = initial_velocities[i] if i < len(initial_velocities) else [0.0, 0.0]
            drone = Drone(drone_id, pos, vel)
            self.drones.append(drone)

        # 集群参数
        self.D_c = 20.0  # 最大观察距离
        self.D_d = 10.0  # 期望邻居距离
        self.D_l1 = 2.0  # 最小防撞距离

        # 障碍物列表（初始为空）
        self.obstacles = []

        # 历史记录（用于可视化）
        self.history = {
            "positions": [],  # 每个时间步的所有无人机位置
            "velocities": [],  # 每个时间步的所有无人机速度
            "time_steps": [],  # 时间戳
        }

    def set_obstacles(self, obstacles):
        """
        设置障碍物

        参数:
            obstacles: 障碍物列表，每个元素是[x, y, radius]
        """
        self.obstacles = obstacles

    def get_neighbors(self, drone_id):
        """
        获取指定无人机的邻居（距离在D_c范围内的无人机）

        参数:
            drone_id: 无人机ID

        返回:
            邻居ID列表
        """
        neighbors = []
        current_drone = self.drones[drone_id - 1]

        for other_id in range(1, self.num_drones + 1):
            if other_id == drone_id:
                continue  # 跳过自己

            other_drone = self.drones[other_id - 1]
            distance = current_drone.distance_to(other_drone)

            if distance <= self.D_c:
                neighbors.append(other_id)

        return neighbors

    def record_history(self, current_time):
        """
        记录当前时刻集群的状态，用于后续可视化

        参数:
            current_time: 当前仿真时间
        """
        # 获取所有无人机的当前位置
        positions = []
        for drone in self.drones:
            positions.append(drone.get_position().copy())

        # 获取所有无人机的当前速度
        velocities = []
        for drone in self.drones:
            velocities.append(drone.get_velocity().copy())

        # 保存到历史记录
        self.history["positions"].append(np.array(positions))
        self.history["velocities"].append(np.array(velocities))
        self.history["time_steps"].append(current_time)

    def get_history(self):
        """
        获取历史记录数据

        返回:
            历史记录字典
        """
        return self.history

    def get_swarm_state(self):
        """
        获取集群当前状态

        返回:
            状态字典，包含位置、速度等信息
        """
        positions = []
        velocities = []
        speeds = []
        yaws = []

        for drone in self.drones:
            positions.append(drone.get_position())
            velocities.append(drone.get_velocity())
            speeds.append(drone.get_speed())
            yaws.append(drone.get_yaw_degrees())

        return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "speeds": np.array(speeds),
            "yaws": np.array(yaws),
        }

    def __str__(self):
        """
        返回集群的字符串表示

        返回:
            描述集群状态的字符串
        """
        output = f"无人机集群（{self.num_drones}架无人机）\n"
        output += f"参数：D_c={self.D_c}m, D_d={self.D_d}m, D_l1={self.D_l1}m\n"
        output += f"障碍物数量：{len(self.obstacles)}\n"

        for drone in self.drones:
            output += f"  {drone}\n"

        return output
