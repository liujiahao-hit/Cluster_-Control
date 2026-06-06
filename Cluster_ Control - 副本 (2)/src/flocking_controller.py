"""
集群控制器实现
使用MOPSO优化得到的影响权重,替代原固定权重
"""

import numpy as np
import math

# 导入MOPSO优化器
from .mopso_optimizer import MOPSOSwarmWeightOptimizer


class FlockingController:
    """
    基于MOPSO优化影响权重的集群控制器
    """

    def __init__(
        self,
        reference_drone=None,
        num_drones=5,
        swarm_init_state=None,
    ):
        """
        初始化集群控制器
        参数:
            desired_velocity: 集群期望速度，默认[10, 0] m/s（沿x轴方向）
            num_drones: 无人机数量，默认5架
            swarm_init_state: 集群初始状态（含位置、速度、障碍物），格式：
                {
                    "positions":  无人机初始位置
                    "velocities":  无人机初始速度
                    "obstacles": list  # 障碍物列表，空列表表示无障碍物
                }
        """
        # 控制组件强度系数
        self.C_f = 0.5  # 集群结构控制强度
        self.C_av = 0.1  # 速度对齐控制强度
        self.C_c = 10000.0  # 冲突避免控制强度
        self.C_vf = 2  # 集群速度控制强度

        # 距离参数
        self.D_c = 20.0  # 最大观察距离
        self.D_d = 10.0  # 期望邻居距离
        self.D_l1 = 2.0  # 无人机间最小防撞距离
        self.D_l2 = 10.0  # 无人机与障碍物最小避碰距离
        self.C_delta = (
            1000000000.0  # 集群形状控制参数
        )

        if reference_drone is not None:
            # 情况1：传了无人机对象，直接读取它的属性
            self.desired_speed = reference_drone.V_xy_e
            self.desired_yaw = reference_drone.phi_e
            self.desired_velocity = np.array(
                [
                    self.desired_speed * math.cos(self.desired_yaw),
                    self.desired_speed * math.sin(self.desired_yaw),
                ]
            )
        else:
            # 情况2：没传无人机，保留原来的默认值作为兜底
            self.desired_velocity = np.array([10.0, 0.0], dtype=float)
            self.desired_speed = np.linalg.norm(self.desired_velocity)
            self.desired_yaw = math.atan2(
                self.desired_velocity[1], self.desired_velocity[0]
            )

        # 集群初始状态
        self.swarm_init_state = (
            swarm_init_state
            if swarm_init_state is not None
            else self._default_swarm_state(num_drones)
        )
        # 无人机数量
        self.num_drones = num_drones
        # 初始化权重矩阵
        self.optimized_weights = np.ones((num_drones, num_drones)) / num_drones
        print("[控制器] 初始化完成，将在每个时间步调用MOPSO更新权重矩阵。")

    def _default_swarm_state(self, num_drones):
        """默认集群初始状态（无障碍物，随机位置/速度）"""
        return {
            "positions": np.random.uniform(
                0, 50, (num_drones, 2)
            ),  # 初始位置：0-50m随机
            "velocities": np.tile(self.desired_velocity, (num_drones, 1))
            + np.random.normal(0, 0.5, (num_drones, 2)),  # 初始速度：接近期望速度
            "obstacles": [],  # 默认无障碍物
        }

    def _normalize_weights(self, weight_matrix):
        """
        权重归一化：确保每一行（当前无人机对所有邻居的权重）和为1，且非负
        对应文献中权重的物理意义（影响权重之和为1）
        """
        # 确保权重非负
        weight_matrix = np.maximum(weight_matrix, 0.001)  # 避免0权重导致无影响
        # 每行归一化（和为1）
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        return weight_matrix / row_sums

    # ==============步骤1：计算期望速度===============
    def _compute_swarm_core_position(self, swarm):
        """
        步骤1：计算集群核心位置Q^core（公式2-9）
        参数:
            swarm: 集群对象或状态字典
        返回:
            Q_core: 集群核心位置向量 [x_mean, y_mean]
        """
        if hasattr(swarm, 'drones'):
            all_positions = np.array([drone.get_position() for drone in swarm.drones])
        else:
            all_positions = swarm["positions"]
        Q_core = np.mean(all_positions, axis=0)
        return Q_core

    def compute_S_o_set(self, Q_core, all_obstacles):
        """
        筛选S_o集合
        输入：
            Q_core → 集群核心位置
            all_obstacles → 所有障碍物列表，每个元素为[Qx, Qy, Ro]（x坐标,y坐标,障碍物半径）
        输出：
            S_o → 符合条件的障碍物列表
            S_o_not_empty → S_o是否非空
        """
        desired_dir = np.array([math.cos(self.desired_yaw), math.sin(self.desired_yaw)])
        S_o = []
        for obstacle in all_obstacles:
            obs_x, obs_y, obs_Ro = obstacle
            obs_center = np.array([obs_x, obs_y])
            obs_to_core = obs_center - Q_core
            # 论文S_o筛选3个条件
            projection = np.dot(obs_to_core, desired_dir)
            if projection <= 0:  # 条件1：在集群前方
                continue
            if projection >= 50.0:  # 条件2：在最大感知距离内
                continue
            # 条件3：障碍物的外包络线进入集群横向范围
            R_formation = self.D_d * 1  # 用期望邻居距离作为集群半径
            # 计算障碍物的外包络线半径：障碍物半径 + 集群半径 + 最小允许距离D_l2
            outer_envelope_radius = obs_Ro + R_formation + self.D_l2
            # 计算障碍物中心到集群核心的垂直距离
            perpendicular_dist = np.linalg.norm(obs_to_core - projection * desired_dir)
            if perpendicular_dist >= outer_envelope_radius:
                continue
            S_o.append(obstacle)
        return S_o, len(S_o) > 0

    def compute_core_yaw(self, Q_core, S_o, current_time):
        """
        计算集群核心期望偏航角δ^core（公式2-10 + 公式2-22）
        输入：
            Q_core → 集群核心位置
            S_o → 筛选后的障碍物集合
            current_time → 当前仿真时间t（单位：秒），用于公式(2-22)分段判断
        输出：delta_core → 核心期望偏航角
        """
        # t≤50s：避障阶段，由障碍物计算核心偏航角
        if current_time <= 50.0 and len(S_o) > 0:
            desired_dir = np.array([math.cos(self.desired_yaw), math.sin(self.desired_yaw)])
            perpendicular_dir = np.array([-desired_dir[1], desired_dir[0]])
            # 找最前方的障碍物（投影最小的）
            min_projection = float("inf")
            closest_obs = None
            for obs in S_o:
                obs_center = np.array([obs[0], obs[1]])
                projection = np.dot(obs_center - Q_core, desired_dir)
                if projection < min_projection:
                    min_projection = projection
                    closest_obs = obs
            if closest_obs is not None:
                # 按论文定义计算点A - 考虑外包络线（障碍物半径 + 最小允许距离 + 集群半径）
                obs_center = np.array([closest_obs[0], closest_obs[1]])
                obs_Ro = closest_obs[2]
                # 计算集群半径
                R_formation = self.D_d * 1  # 用期望邻居距离作为集群半径
                # 计算外包络线半径：障碍物半径 + 最小允许距离D_l2 + 集群半径
                outer_envelope_radius = obs_Ro + self.D_l2 + R_formation
                # 计算外包络线在垂直方向上的两个边缘点
                edge_1 = obs_center + outer_envelope_radius * perpendicular_dir
                edge_2 = obs_center - outer_envelope_radius * perpendicular_dir
                # 选择最接近集群核心的点作为A点
                dist_1 = np.linalg.norm(edge_1 - Q_core)
                dist_2 = np.linalg.norm(edge_2 - Q_core)
                point_A = edge_1 if dist_1 < dist_2 else edge_2
                # 按公式2-10计算delta_core
                return math.atan2(point_A[1] - Q_core[1], point_A[0] - Q_core[0])
        
        # 50s<t≤80s：正弦动态偏航阶段
        if 50.0 < current_time <= 80.0:
            return -np.pi / 3 * np.sin(2 * np.pi * (current_time - 50) / 30)
        # t>80s或无障碍物：回到期望偏航角
        return np.arctan2(self.desired_velocity[1], self.desired_velocity[0])

    def compute_drone_yaw(self, drone_pos, Q_core, delta_core):
        """
        计算单无人机i的期望偏航角δ^i（公式2-8）
        输入：
            drone_pos → 当前无人机i的位置
            Q_core → 集群核心位置
            delta_core → 集群核心期望偏航角
        输出：delta_i → 无人机i的期望偏航角
        """
        Q_i = drone_pos
        numerator = Q_core[1] + self.C_delta * math.sin(delta_core) - Q_i[1]
        denominator = Q_core[0] + self.C_delta * math.cos(delta_core) - Q_i[0]
        delta_i = math.atan2(numerator, denominator)
        return delta_i

    def compute_efv(self, delta_i):
        """
        计算单无人机i的期望集群速度efv^i
        输入：delta_i → 无人机i的期望偏航角
        输出：efv_i → 期望速度向量
        """
        efv_i = np.array(
            [
                self.desired_speed * math.cos(delta_i),
                self.desired_speed * math.sin(delta_i),
            ]
        )
        return efv_i

    # ===========步骤2：定义MOPSO目标函数 ==============#
    def _simulate_next_state(self, weight_matrix, current_state, dt=0.5):
        """
        【双仿真核心】模拟给定权重矩阵下所有无人机的下一时刻状态
        输入：
            weight_matrix: 候选权重矩阵（N×N）
            current_state: 当前集群状态字典
            dt: 时间步长
        输出：
            next_positions: 所有无人机模拟的下一时刻位置
            next_velocities: 所有无人机模拟的下一时刻速度
        """
        # 保存原始权重
        original_weights = self.optimized_weights.copy()
        
        # 应用候选权重矩阵
        self.optimized_weights = weight_matrix
        
        # 初始化下一时刻状态
        num_drones = len(current_state["positions"])
        next_positions = np.zeros_like(current_state["positions"])
        next_velocities = np.zeros_like(current_state["velocities"])
        
        # 计算所有无人机的控制输入和下一时刻状态
        for drone_i in range(num_drones):
            # 计算控制输入
            u_i = self.compute_control_input(
                drone_i,
                current_state["positions"][drone_i],
                current_state["velocities"][drone_i],
                current_state["efv_list"][drone_i],
                current_state["positions"],
                current_state["velocities"]
            )
            
            # 模拟动力学更新（公式2-8）
            current_vel = current_state["velocities"][drone_i].copy()
            current_pos = current_state["positions"][drone_i].copy()
            
            next_vel = current_vel + u_i * dt
            next_pos = current_pos + next_vel * dt
            
            next_positions[drone_i] = next_pos
            next_velocities[drone_i] = next_vel
        
        # 恢复原始权重
        self.optimized_weights = original_weights
        
        return next_positions, next_velocities

    def obj1_speed_error(self, weight_matrix, S_o_not_empty):
        """
        目标函数1：全集群的速度跟踪误差（公式2-12）
        输入：
            weight_matrix → 权重矩阵（N×N，MOPSO优化的对象）
            S_o_not_empty → S_o是否非空
        输出：obj1 → 全集群的速度误差总和（越小越好）
        """
        # 使用从主循环传递的正确时间
        current_time = self.current_time
        # 权重归一化（论文要求每一行权重和为1）
        weight_matrix = np.maximum(weight_matrix, 0.001)
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        weight_matrix = weight_matrix / row_sums
        
        # 使用当前状态和候选权重矩阵模拟未来状态
        current_state = {
            "positions": self.current_positions,
            "velocities": self.current_velocities,
            "efv_list": self.current_efv_list
        }
        next_positions, next_velocities = self._simulate_next_state(weight_matrix, current_state)
        
        obj1 = 0.0
        num_drones = len(next_velocities)
        
        for drone_i in range(num_drones):
            efv_i = self.current_efv_list[drone_i]
            next_vel = next_velocities[drone_i]
            
            # 公式2-12实现
            is_efv_not_equal = not np.allclose(efv_i, self.desired_velocity, atol=1e-3)
            efv_norm = np.linalg.norm(efv_i)
            efv_norm = 1e-6 if efv_norm < 1e-6 else efv_norm
            
            # 时间条件：0-30秒和50-80秒使用情况一
            time_condition = (current_time >= 0 and current_time <= 30) or (current_time >= 50 and current_time <= 80)
            
            # 原有逻辑和时间条件
            if (is_efv_not_equal or S_o_not_empty) or time_condition:
                # 情况1：避障/期望速度变化 → 点积公式
                dot_product = np.dot(next_vel, efv_i)
                obj1_i = -dot_product / (2 * efv_norm)
            else:
                # 情况2：无障碍物+期望速度一致 → 矢量差的大小
                vel_diff = efv_i - next_vel
                obj1_i = np.linalg.norm(vel_diff)
            
            obj1 += obj1_i
        
        return obj1

    def obj2_structure_error(self, weight_matrix):
        """
        目标函数2：全集群的结构偏差（公式2-13）
        输入：
            weight_matrix → 权重矩阵（N×N，MOPSO优化的对象）
        输出：obj2 → 全集群的结构偏差总和（越小越好）
        """
        # 权重归一化（论文要求每一行权重和为1）
        weight_matrix = np.maximum(weight_matrix, 0.001)
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        weight_matrix = weight_matrix / row_sums
        
        # 使用当前状态和候选权重矩阵模拟未来状态
        current_state = {
            "positions": self.current_positions,
            "velocities": self.current_velocities,
            "efv_list": self.current_efv_list
        }
        next_positions, next_velocities = self._simulate_next_state(weight_matrix, current_state)
        
        obj2 = 0.0
        num_drones = len(next_positions)
        
        for drone_i in range(num_drones):
            next_pos = next_positions[drone_i]
            next_vel = next_velocities[drone_i]
            weight_vector = weight_matrix[drone_i]
            
            # 筛选邻居：距离≤D_c的无人机j
            for drone_j in range(num_drones):
                if drone_i == drone_j:
                    continue
                
                # 对于其他无人机，使用模拟的未来位置进行邻居判断
                pos_j = next_positions[drone_j]
                d_ij = np.linalg.norm(next_pos - pos_j)
                if d_ij <= self.D_c:
                    # 公式2-13实现：距离偏差+速度偏差
                    distance_term = np.abs(self.D_d - d_ij)
                    vel_j = next_velocities[drone_j]
                    # 速度偏差：先矢量做差，再求大小
                    vel_diff = next_vel - vel_j
                    speed_term = np.linalg.norm(vel_diff)
                    obj2 += (distance_term + speed_term)
        
        return obj2
    
    def compute_control_input(
        self, drone_i, drone_pos, drone_vel, efv_i, all_positions, all_velocities
    ):
        """
        计算单无人机i的总控制输入u^i（对应公式2-3）
        输入：
            drone_i → 无人机编号（0开始）
            drone_pos → 无人机i的当前位置
            drone_vel → 无人机i的当前速度
            efv_i → 无人机i的期望速度efv^i
            all_positions → 所有无人机的位置
            all_velocities → 所有无人机的速度
        输出：u_i → 总控制输入向量)
        """
        # 取当前无人机i的权重向量
        weight_vector = self.optimized_weights[drone_i]
        weight_self = weight_vector[drone_i]  # 自身权重（公式2-7）
        # 1. 集群结构控制组件u_f^i（公式2-4）
        u_f = np.array([0.0, 0.0])
        for drone_j in range(self.num_drones):
            if drone_i == drone_j:
                continue
            pos_j = all_positions[drone_j]
            d_ij = np.linalg.norm(drone_pos - pos_j)
            if 0 < d_ij <= self.D_c:
                weight_ij = weight_vector[drone_j]
                direction = pos_j - drone_pos
                log_term = math.log(d_ij / self.D_d)
                u_f += weight_ij * direction * log_term
        u_f *= self.C_f
        # 2. 速度对齐控制组件u_av^i（公式2-5）
        u_av = np.array([0.0, 0.0])
        for drone_j in range(self.num_drones):
            if drone_i == drone_j:
                continue
            vel_j = all_velocities[drone_j]
            pos_j = all_positions[drone_j]
            d_ij = np.linalg.norm(drone_pos - pos_j)
            if d_ij <= self.D_c:
                weight_ij = weight_vector[drone_j]
                u_av += weight_ij * (vel_j - drone_vel)
        u_av *= self.C_av
        # 3. 冲突避免控制组件u_c^i（公式2-6）
        u_c = np.array([0.0, 0.0])
        for drone_j in range(self.num_drones):
            if drone_i == drone_j:
                continue
            pos_j = all_positions[drone_j]
            direction = drone_pos - pos_j
            d_ij = np.linalg.norm(direction)
            if 0 < d_ij < self.D_l1:
                force_magnitude = (1.0 / d_ij - 1.0 / self.D_l1) ** 2
                force_direction = direction / d_ij
                u_c += force_magnitude * force_direction
        u_c *= self.C_c
        # 4. 速度跟踪控制组件u_vf^i（公式2-7）
        u_vf = self.C_vf * weight_self * (efv_i - drone_vel)
        # 总控制输入（公式2-3）
        u_i = u_f + u_av + u_c + u_vf
        return u_i

    def update_weights_matrix(self, swarm_state, current_time=0.0):
        """
        【核心方法】每个时间步调用MOPSO优化器，更新全集群的权重矩阵
        参数:
            swarm_state: 集群当前状态字典，必须包含：
                "positions": 所有无人机位置，shape=(N,2)
                "velocities": 所有无人机速度，shape=(N,2)
                "S_o_not_empty": S_o集合是否非空（布尔值，避障判断用）
                "efv_list": 所有无人机的期望速度efv列表，shape=(N,2)
            current_time: 当前仿真时间（秒）
        返回:
            weights_matrix: 优化后的N×N权重矩阵
            total_obj1: 全集群目标函数1之和
            total_obj2: 全集群目标函数2之和
            mopso_history_list: 每架无人机的MOPSO优化过程记录
        """
        # 保存当前时间，供目标函数使用
        self.current_time = current_time
        # 取出集群当前状态
        all_positions = swarm_state["positions"]
        all_velocities = swarm_state["velocities"]
        S_o_not_empty = swarm_state["S_o_not_empty"]
        efv_list = swarm_state["efv_list"]
        num_drones = self.num_drones
        
        # 保存当前状态到对象属性，供目标函数中的模拟方法使用
        self.current_positions = all_positions
        self.current_velocities = all_velocities
        self.current_efv_list = efv_list
        
        # 定义全局双目标函数，优化整个权重矩阵
        # 注意：MOPSO优化器的输入是一个向量，因此需要将权重矩阵展平
        def obj1_func(weight_vector):
            # 将展平的向量重新塑形为权重矩阵
            weight_matrix = weight_vector.reshape((num_drones, num_drones))
            return self.obj1_speed_error(weight_matrix, S_o_not_empty)

        def obj2_func(weight_vector):
            # 将展平的向量重新塑形为权重矩阵
            weight_matrix = weight_vector.reshape((num_drones, num_drones))
            return self.obj2_structure_error(weight_matrix)

        # 定义权重上下界：对于整个权重矩阵（展平后的向量）
        weight_bounds = [(0.001, 1.0) for _ in range(num_drones * num_drones)]

        # 初始化MOPSO优化器，平衡计算效率和优化效果
        mopso_optimizer = MOPSOSwarmWeightOptimizer(
            obj1_func=obj1_func, 
            obj2_func=obj2_func, 
            weight_bounds=weight_bounds,
            max_generations=58,
            particle_num=20
        )

        # 执行优化，得到最优权重向量（展平的）和优化记录
        optimal_weight_vector, mopso_history = mopso_optimizer.optimize()

        # 将展平的权重向量重新塑形为权重矩阵
        new_weights_matrix = optimal_weight_vector.reshape((num_drones, num_drones))
        
        # 权重归一化（确保每一行权重和为1）
        new_weights_matrix = np.maximum(new_weights_matrix, 0.001)
        row_sums = new_weights_matrix.sum(axis=1, keepdims=True)
        new_weights_matrix = new_weights_matrix / row_sums

        # 计算最终的目标函数值
        total_obj1 = mopso_history["best_obj1"][-1]
        total_obj2 = mopso_history["best_obj2"][-1]

        # 更新类的权重矩阵属性，供compute_control_input调用
        self.optimized_weights = new_weights_matrix
        print(
            f"[控制器] 时间步权重矩阵优化完成，全集群obj1总和：{total_obj1:.2f}，obj2总和：{total_obj2:.2f}"
        )
        return new_weights_matrix, total_obj1, total_obj2, [mopso_history]

    def _optimize_weights_by_mopso(self, swarm_state):
        """
        兼容你原有generate_dataset.py的接口，生成数据集时直接调用
        """
        weights_matrix, _, _, _ = self.update_weights_matrix(swarm_state)
        return weights_matrix
