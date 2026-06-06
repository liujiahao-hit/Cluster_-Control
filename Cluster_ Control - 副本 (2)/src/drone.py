"""
无人机动力学模型实现，公式(2-1)和(2-2)
"""

import numpy as np
import math

class Drone:
    """
    无人机类 - 模拟单架无人机的动力学行为

    属性:
        id: 无人机唯一标识符
        position——Q^i: 位置向量 [x, y] (单位:米)
        velocity——V^i: 速度向量 [vx, vy] (单位:米/秒)
        V_xy: 速度大小 (单位:米/秒)
        yaw——φ^i: 偏航角 (弧度)
        control_input——u^i: 控制输入向量 [ux, uy] (单位:米/秒²)
        
    固定参数：
        min_speed——V_xy^min: 最小水平空速
        max_speed——V_xy^max: 最大水平空速
        n_max：横向最大过载
        g:重力加速度
        V_xy_l：空速允许误差
        phi_l：偏航角允许误差
        V_e：期望速度向量
        V_xy_e：期望速度大小
        phi_e：期望偏航角
    """

    def __init__(self, drone_id, initial_position=None, initial_velocity=None):
        """
        初始化无人机对象

        参数:
            drone_id: 无人机ID (整数，从1开始)
            initial_position: 初始位置 [x, y]，默认[0, 0]
            initial_velocity: 初始速度 [vx, vy]，默认[0, 0]
        """
        self.id = drone_id

        # 设置初始位置
        if initial_position is None:
            self.position = np.array([0.0, 0.0], dtype=float)
        else:
            self.position = np.array(initial_position, dtype=float)

        # 设置初始速度(期望水平空速)
        if initial_velocity is None:
            self.velocity = np.array([10.0, 0.0], dtype=float)  # 默认初始速度为10 m/s，沿x轴正方向
        else:
            self.velocity = np.array(initial_velocity, dtype=float)

        # 水平空速
        self.V_xy = np.linalg.norm(self.velocity)

        # 偏航角（速度方向）
        if self.V_xy > 0.01:  # 避免除零
            self.yaw = math.atan2(self.velocity[1], self.velocity[0])
        else:
            self.yaw = 0.0

        # 控制输入（初始为零）
        self.control_input = np.array([0.0, 0.0], dtype=float)

        # 无人机物理参数（根据论文公式2-2）
        self.min_speed = 5.0
        self.max_speed = 15.0
        self.n_max = 10.0
        self.g = 10.0
        self.V_xy_l = 0.25
        self.phi_l = 0.1

        self.V_e = np.array([10.0, 0.0],dtype=float)  # 期望速度向量
        self.V_xy_e = np.linalg.norm(self.V_e)
        self.phi_e = math.atan2(self.V_e[1], self.V_e[0])

        # 历史轨迹记录
        self.trajectory = [self.position.copy()]

    def set_control_input(self, control_input):
        """
        设置控制输入向量，u^i = (u_x^i, u_y^i)

        参数:
            control_input: 控制输入向量 [ux, uy]
        """
        self.control_input = np.array(control_input, dtype=float)

    def update_state(self, dt):
        """
        更新无人机状态，对应公式(2-1)

        参数:
            dt: 时间步长 (单位:秒)
        """
        # 取出当前时刻的状态和控制输入
        ux, uy = self.control_input
        V_xy_current = self.V_xy
        phi_current = self.yaw

        # ============== 步骤1：计算速度和偏航角的变化率========#
        # 物理意义：速度大小的变化率 = 加速度在速度方向上的投影（切向加速度）
        V_xy_dot = ux * math.cos(phi_current) + uy * math.sin(phi_current)

        # 物理意义：偏航角变化率 = 加速度在垂直速度方向上的投影（法向加速度）/ 速度大小
        if V_xy_current > 0.01:
            phi_dot = (
                uy * math.cos(phi_current) - ux * math.sin(phi_current)
            ) / V_xy_current
        else:
            phi_dot = 0.0

        # ============== 步骤2：实现全部约束============#

        # 物理意义：最大横向过载限制，防止转弯太急超出无人机物理极限
        if V_xy_current > 0.01:
            max_phi_dot = (self.n_max * self.g) / V_xy_current
        else:
            max_phi_dot = np.inf
        # 把偏航角速率限制在允许范围内
        phi_dot = np.clip(phi_dot, -max_phi_dot, max_phi_dot)

        # --- 先积分得到初步的新状态（空速、偏航角）---
        V_xy_new = V_xy_current + V_xy_dot * dt
        phi_new = phi_current + phi_dot * dt

        # 物理意义：限制无人机的最小/最大飞行速度
        V_xy_new = np.clip(V_xy_new, self.min_speed, self.max_speed)

        # 物理意义：空速误差足够小时，直接锁定为期望空速，避免飞行抖动
        if abs(V_xy_new - self.V_xy_e) < self.V_xy_l:
            V_xy_new = self.V_xy_e

        # 物理意义：偏航角误差足够小时，直接锁定为期望偏航角
        # 先把角度差归一化到[-π, π]，避免360度循环的计算错误
        phi_error = phi_new - self.phi_e
        phi_error = (phi_error + math.pi) % (2 * math.pi) - math.pi
        if abs(phi_error) < self.phi_l:
            phi_new = self.phi_e

        # =======步骤3：根据新的空速和偏航角计算新的速度向量和位置========#
        # 用更新后的空速和偏航角，重新计算地面系速度分量 公式(2-1)前2行
        self.velocity[0] = V_xy_new * math.cos(phi_new)  
        self.velocity[1] = V_xy_new * math.sin(phi_new)  

        # 物理意义：位置的变化率 = 地面系速度分量，积分更新位置
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt

        self.V_xy = V_xy_new
        self.yaw = phi_new

        # 5. 记录当前位置到历史轨迹
        self.trajectory.append(self.position.copy())

        # 6. 重置控制输入，为下一个时间步准备
        self.control_input = np.array([0.0, 0.0])

    def get_position(self):
        """
        获取当前位置
        """
        return self.position.copy()

    def get_velocity(self):
        """
        获取当前速度
        """
        return self.velocity.copy()

    def get_speed(self):
        """
        获取当前速度大小（水平空速）
        """
        return np.linalg.norm(self.velocity)

    def get_yaw(self):
        """
        获取当前偏航角
        """
        return self.yaw

    def get_yaw_degrees(self):
        """
        获取当前偏航角（转换为角度）
        """
        return math.degrees(self.yaw)

    def distance_to(self, other_drone):
        """
        计算与另一架无人机的欧氏距离
        参数:
            other_drone: 另一架无人机对象
        返回:
            两架无人机之间的距离 (米)
        """
        pos1 = self.position
        pos2 = other_drone.position
        distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return distance

    def get_state_vector(self):
        """
        获取无人机的完整状态向量
        格式：[x, y, vx, vy]
        返回:
            4维状态向量
        """
        return np.concatenate([self.position, self.velocity])

    def get_trajectory(self):
        """
        获取无人机的历史轨迹
        返回:
            轨迹点数组，形状为 (n, 2)
        """
        return np.array(self.trajectory)

    def __str__(self):
        """
        返回无人机的字符串表示，用于调试
        返回:
            描述无人机状态的字符串
        """
        pos = self.position
        vel = self.velocity
        speed = self.get_speed()
        yaw_deg = self.get_yaw_degrees()

        return (
            f"无人机{self.id}: 位置({pos[0]:.1f}, {pos[1]:.1f})m, "
            f"速度({vel[0]:.1f}, {vel[1]:.1f})m/s, "
            f"速度大小{speed:.1f}m/s, 偏航角{yaw_deg:.1f}°"
        )
