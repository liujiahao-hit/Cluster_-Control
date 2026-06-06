import numpy as np

class MOPSOSwarmWeightOptimizer:
    """
    MOPSO，用于计算无人机集群的最优影响权重向量
    输入：集群速度目标函数obj1、集群结构目标函数obj2、最大代数Mg、粒子数目Np、权重上下界
    输出：最优影响权重向量 + 优化过程全量记录数据
    """

    def __init__(
        self,
        obj1_func,
        obj2_func,
        max_generations=58,
        particle_num=20,
        weight_bounds=None,
    ):
        # ========== 对应“输入”部分 ==========
        self.obj1 = obj1_func  # (2-12)的集群速度目标函数obj1^i
        self.obj2 = obj2_func  # (2-13)的集群结构目标函数obj2^i
        self.Mg = max_generations  # 最大代数Mg
        self.Np = particle_num  # 粒子数目Np
        # 权重上下界默认设置：每个权重范围[0.001, 1.0]
        self.weight_bounds = (
            weight_bounds
            if weight_bounds is not None
            else [(0.001, 1.0) for _ in range(5)]
        )
        self.weight_dim = len(self.weight_bounds)  # 影响权重向量的维度

        # ========== 对应“初始化”步骤 ==========
        self.particle_pos = self._init_particle_position()  # L^h：粒子位置=影响权重向量
        self.particle_vel = self._init_particle_velocity()  # S^h：粒子速度向量
        self.current_gen = 1  # 当前代数g，初始为1
        self.pareto_front = []  # 历史帕累托最优集合REP，初始为空
        self.pareto_obj_history = np.empty((0, 2))  # 帕累托前沿对应的目标函数值
        self.pbest = self.particle_pos.copy()  # 步骤3：个体最优pbest初始化为当前位置

        # ========== 优化过程记录（可视化） ==========
        self.optimize_history = {
            "generation": [],  # 迭代代数
            "pareto_obj1": [],  # 每代帕累托前沿的obj1值
            "pareto_obj2": [],  # 每代帕累托前沿的obj2值
            "best_obj1": [],  # 每代最优解的obj1值
            "best_obj2": [],  # 每代最优解的obj2值
            "best_weight": [],  # 每代最优权重向量
        }

    def _init_particle_position(self):
        """“初始化”：在上下界内随机初始化Np个粒子的位置（影响权重）"""
        bound_low = np.array([b[0] for b in self.weight_bounds])
        bound_high = np.array([b[1] for b in self.weight_bounds])
        # 直接生成符合要求的随机数组，提高效率
        particle_pos = np.random.rand(self.Np, self.weight_dim)
        # 将随机值映射到指定上下界
        particle_pos = bound_low + (bound_high - bound_low) * particle_pos
        return particle_pos

    def _init_particle_velocity(self):
        """“初始化”：随机初始化粒子速度（速度上下界取权重上下界的差值）"""
        bound_diff = np.array([b[1] - b[0] for b in self.weight_bounds])
        vel_low = -bound_diff
        vel_high = bound_diff
        # 直接生成符合要求的随机数组，提高效率
        particle_vel = np.random.rand(self.Np, self.weight_dim)
        # 将随机值映射到指定上下界
        particle_vel = vel_low + (vel_high - vel_low) * particle_vel
        return particle_vel

    def _calc_objective_values(self, positions):
        """
        “步骤1/步骤8”：计算所有粒子位置对应的目标函数值
        参数positions：粒子位置矩阵（即影响权重向量集合）
        返回：(Np, 2)的矩阵，每一行是[obj1值, obj2值]
        """
        obj_values = []
        for weight in positions:
            obj1_val = self.obj1(weight)
            obj2_val = self.obj2(weight)
            obj_values.append([obj1_val, obj2_val])
        return np.array(obj_values)

    def _get_pareto_front(self, positions, obj_values):
        """
        “步骤2/步骤10”：帕累托排序，筛选不被支配的粒子（帕累托前沿）
        支配规则：若粒子j的obj1、obj2均不劣于粒子i，且至少一个更优，则i被j支配
        """
        pareto_particles = []
        pareto_obj_values = []
        for i in range(len(positions)):
            is_dominated = False
            for j in range(len(positions)):
                if i == j:
                    continue
                # 满足支配条件
                if (obj_values[j, 0] <= obj_values[i, 0]) and (
                    obj_values[j, 1] <= obj_values[i, 1]
                ):
                    if (obj_values[j, 0] < obj_values[i, 0]) or (
                        obj_values[j, 1] < obj_values[i, 1]
                    ):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_particles.append(positions[i])
                pareto_obj_values.append(obj_values[i])
        return np.array(pareto_particles), np.array(pareto_obj_values)

    def _update_pareto_front(self, new_pareto, new_pareto_obj):
        """“步骤2/步骤10”：更新历史帕累托最优集合REP"""
        # 合并当前REP与新帕累托前沿，再重新筛选帕累托前沿（去重+保留最优）
        if len(self.pareto_front) == 0:
            combined_pos = new_pareto
            combined_obj = new_pareto_obj
        else:
            combined_pos = np.vstack([self.pareto_front, new_pareto])
            combined_obj = np.vstack([self.pareto_obj_history, new_pareto_obj])

        # 重新筛选帕累托前沿，作为新的REP
        self.pareto_front, self.pareto_obj_history = self._get_pareto_front(
            combined_pos, combined_obj
        )

    def _check_bounds(self):
        """
        “步骤7”：若粒子位置/速度超出上下界，则重新随机初始化
        """
        bound_low = np.array([b[0] for b in self.weight_bounds])
        bound_high = np.array([b[1] for b in self.weight_bounds])
        vel_diff = bound_high - bound_low
        vel_low = -vel_diff
        vel_high = vel_diff

        # 检查每个粒子的位置和速度是否超出上下界
        for i in range(self.Np):
            # 检查位置
            pos = self.particle_pos[i]
            pos_out_of_bounds = np.any(pos < bound_low) or np.any(pos > bound_high)
            
            # 检查速度
            vel = self.particle_vel[i]
            vel_out_of_bounds = np.any(vel < vel_low) or np.any(vel > vel_high)
            
            # 如果位置或速度超出上下界，重新随机初始化
            if pos_out_of_bounds or vel_out_of_bounds:
                # 重新初始化位置
                self.particle_pos[i] = bound_low + (bound_high - bound_low) * np.random.rand(self.weight_dim)
                # 重新初始化速度
                self.particle_vel[i] = vel_low + (vel_high - vel_low) * np.random.rand(self.weight_dim)

    def optimize(self):
        """
        “步骤1-11”：执行MOPSO迭代流程
        返回：
            optimal_weight: 最优影响权重向量
            optimize_history: 本次优化的全流程记录数据
        """
        # ========== 步骤1：计算初始粒子的目标函数值 ==========
        init_obj_vals = self._calc_objective_values(self.particle_pos)
        # ========== 步骤2：初始帕累托排序，更新REP ==========
        init_pareto, init_pareto_obj = self._get_pareto_front(
            self.particle_pos, init_obj_vals
        )
        self._update_pareto_front(init_pareto, init_pareto_obj)

        # ========== 迭代流程（步骤4-11循环） ==========
        while self.current_gen <= self.Mg:
            # 步骤4：从REP中随机选择gbest
            if len(self.pareto_front) == 0:
                gbest = self.pbest[np.random.randint(self.Np)]
            else:
                gbest = self.pareto_front[np.random.randint(len(self.pareto_front))]

            # 步骤5：更新粒子速度（式2-14）
            a = 0.4  # 文献指定的惯性权重
            r1 = np.random.uniform(0, 1, size=self.weight_dim)  # 随机向量r1
            r2 = np.random.uniform(0, 1, size=self.weight_dim)  # 随机向量r2
            self.particle_vel = (
                a * self.particle_vel
                + r1 * (self.pbest - self.particle_pos)
                + r2 * (gbest - self.particle_pos)
            )

            # 步骤6：更新粒子位置（式2-15）
            self.particle_pos = self.particle_pos + self.particle_vel

            # 步骤7：检查位置/速度边界，超出则重新初始化
            self._check_bounds()

            # 步骤8：计算更新后粒子的目标函数值
            current_obj_vals = self._calc_objective_values(self.particle_pos)

            # 步骤9：更新pbest（新位置不被pbest支配，则更新pbest）
            # 一次性计算所有pbest的目标函数值，避免循环中重复计算
            pbest_obj_vals = self._calc_objective_values(self.pbest)
            
            for i in range(self.Np):
                pbest_obj = pbest_obj_vals[i]
                current_obj = current_obj_vals[i]
                
                # 新位置不被pbest支配，则更新pbest
                # 支配条件：pbest的两个目标值都不大于当前值，且至少一个严格小于
                is_dominated_by_pbest = (pbest_obj[0] <= current_obj[0] and pbest_obj[1] <= current_obj[1]) and (pbest_obj[0] < current_obj[0] or pbest_obj[1] < current_obj[1])
                
                if not is_dominated_by_pbest:
                    self.pbest[i] = self.particle_pos[i].copy()

            # 步骤10：更新REP
            current_pareto, current_pareto_obj = self._get_pareto_front(
                self.particle_pos, current_obj_vals
            )
            self._update_pareto_front(current_pareto, current_pareto_obj)

            # ========== 记录当前代的优化数据 ==========
            # 找到当前代最优解（obj2最小）
            best_idx = np.argmin(current_pareto_obj[:, 1])
            best_obj1 = current_pareto_obj[best_idx, 0]
            best_obj2 = current_pareto_obj[best_idx, 1]
            best_weight = current_pareto[best_idx]

            self.optimize_history["generation"].append(self.current_gen)
            self.optimize_history["pareto_obj1"].append(current_pareto_obj[:, 0])
            self.optimize_history["pareto_obj2"].append(current_pareto_obj[:, 1])
            self.optimize_history["best_obj1"].append(best_obj1)
            self.optimize_history["best_obj2"].append(best_obj2)
            self.optimize_history["best_weight"].append(best_weight)

            # 步骤11：更新代数，判断是否继续迭代
            self.current_gen += 1

        # ========== 迭代结束，输出最优影响权重（论文指定规则：REP中obj2最小） ==========
        final_best_idx = np.argmin(self.pareto_obj_history[:, 1])
        optimal_weight = self.pareto_front[final_best_idx]

        # 权重归一化，保证和为1，符合论文物理意义
        # 如果权重维度是无人机数量的平方，说明是全局权重矩阵的展平形式
        # 需要将其重新塑形为矩阵后对每行进行归一化
        optimal_weight = np.maximum(optimal_weight, 0.001)
        if np.sqrt(self.weight_dim) == int(np.sqrt(self.weight_dim)):
            # 是平方数，说明是全局权重矩阵的展平形式
            num_drones = int(np.sqrt(self.weight_dim))
            weight_matrix = optimal_weight.reshape((num_drones, num_drones))
            # 对每行进行归一化
            row_sums = weight_matrix.sum(axis=1, keepdims=True)
            weight_matrix = weight_matrix / row_sums
            # 重新展平
            optimal_weight = weight_matrix.flatten()
        else:
            # 不是平方数，说明是单个无人机的权重向量
            optimal_weight = optimal_weight / optimal_weight.sum()

        # 返回2个值，和flocking_controller的调用匹配
        return optimal_weight, self.optimize_history
