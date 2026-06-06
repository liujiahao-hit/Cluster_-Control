"""
无人机集群仿真主流程
"""

import numpy as np
import os
import pickle
from src.drone_swarm import DroneSwarm
from src.flocking_controller import FlockingController


class SwarmSimulation:
    def __init__(
        self,
        num_drones,
        initial_positions,
        initial_velocities,
        obstacles=None,
        max_simulation_time=100.0,  # 最大仿真时间Tmax，默认100s
        time_step=0.5,  # 采样时间ts，默认0.5s
    ):
        """
        仿真初始化（算法2-1第1步）
        参数:
            num_drones: 无人机数量N
            initial_positions: 初始位置列表，每个元素是[x, y]
            initial_velocities: 初始速度列表，每个元素是[vx, vy]
            obstacles: 障碍物列表，每个元素是[x, y, radius]，默认无障碍物
            max_simulation_time: 最大仿真时间Tmax（秒）
            time_step: 采样时间步长ts（秒）
        """
        # ========== 1. 初始化仿真基础参数 ==========
        self.num_drones = num_drones
        self.max_simulation_time = max_simulation_time
        self.time_step = time_step
        self.current_time = 0.0  # 初始化t=0
        self.obstacles = obstacles if obstacles is not None else []

        # ===== 2. 初始化无人机集群（算法2-1无人机状态初始化） ==========
        self.swarm = DroneSwarm(
            num_drones=num_drones,
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
        )
        self.swarm.set_obstacles(self.obstacles)

        # ========== 3. 初始化集群控制器 ==========
        # 以第一架无人机为参考，读取期望速度等参数
        reference_drone = self.swarm.drones[0]
        self.controller = FlockingController(
            reference_drone=reference_drone, num_drones=num_drones
        )

        # ====== 4. 初始化仿真数据记录容器（可视化/神经网络用） ==========
        self.simulation_data = {
            "time_steps": [],  # 时间戳
            "weights_matrix": [],  # 每个时间步的N×N权重矩阵
            "total_obj1": [],  # 每个时间步全集群obj1总和
            "total_obj2": [],  # 每个时间步全集群obj2总和
            "mopso_history": [],  # 每个时间步的MOPSO优化过程记录
            # 每架无人机的状态记录
            "drone_positions": [],  # 形状：(时间步, 无人机数, 2)
            "drone_velocities": [],  # 形状：(时间步, 无人机数, 2)
            "drone_yaws": [],  # 形状：(时间步, 无人机数)
            "drone_expected_yaws": [],  # 形状：(时间步, 无人机数)
            "drone_expected_velocities": [],  # 形状：(时间步, 无人机数, 2)
        }
        print("=" * 60)
        print(
            f"仿真初始化完成！无人机数量：{num_drones}，最大仿真时间：{max_simulation_time}s，步长：{time_step}s"
        )
        print(f"障碍物数量：{len(self.obstacles)}")
        print("=" * 60)

    def run(self):
        """
        仿真主循环（算法2-1第2-11步）
        """
        print("开始仿真运行...")
        # 算法2-1第2步
        while self.current_time < self.max_simulation_time:
            print(
                f"\n------ 当前时间：{self.current_time:.1f}s / {self.max_simulation_time}s ------"
            )

            # ========== 前置计算：集群核心状态、避障相关参数 ==========
            # 1. 计算集群核心位置Q^core
            Q_core = self.controller._compute_swarm_core_position(self.swarm)
            # 2. 筛选障碍物集合S_o
            S_o, S_o_not_empty = self.controller.compute_S_o_set(Q_core, self.obstacles)
            # 3. 计算集群核心期望偏航角δ^core
            delta_core = self.controller.compute_core_yaw(Q_core, S_o,current_time=self.current_time)

            #== 预计算每架无人机的期望偏航角、期望速度（算法第4步） =======
            swarm_state = self.swarm.get_swarm_state()
            all_positions = swarm_state["positions"]
            all_velocities = swarm_state["velocities"]
            efv_list = []  # 每架无人机的期望速度
            expected_yaw_list = []  # 每架无人机的期望偏航角

            for drone_i in range(self.num_drones):
                drone_pos = all_positions[drone_i]
                # 计算单无人机期望偏航角δ^i（第4步）
                delta_i = self.controller.compute_drone_yaw(
                    drone_pos, Q_core, delta_core
                )
                # 计算期望速度efv^i
                efv_i = self.controller.compute_efv(delta_i)
                # 保存结果
                expected_yaw_list.append(delta_i)
                efv_list.append(efv_i)

            # ========== 第5步：计算最优影响权重向量Ŵ^i ==========
            mopso_swarm_state = {
                "positions": all_positions,
                "velocities": all_velocities,
                "S_o_not_empty": S_o_not_empty,
                "efv_list": efv_list,
            }
            weights_matrix, total_obj1, total_obj2, mopso_history = \
                self.controller.update_weights_matrix(mopso_swarm_state)

            # ========== 第6步：计算每架无人机的控制输入u^i ==========
            control_input_list = []
            for drone_i in range(self.num_drones):
                drone_pos = all_positions[drone_i]
                drone_vel = all_velocities[drone_i]
                efv_i = efv_list[drone_i]
                # 计算控制输入
                u_i = self.controller.compute_control_input(
                    drone_i=drone_i,
                    drone_pos=drone_pos,
                    drone_vel=drone_vel,
                    efv_i=efv_i,
                    all_positions=all_positions,
                    all_velocities=all_velocities,
                )
                control_input_list.append(u_i)

            # ========== 第7步：计算无人机的下一个状态 ==========
            # 注意：先计算所有无人机的控制输入，再统一更新状态，避免顺序导致的仿真误差
            for drone_i in range(self.num_drones):
                drone = self.swarm.drones[drone_i]
                u_i = control_input_list[drone_i]
                # 设置控制输入
                drone.set_control_input(u_i)
                # 更新无人机状态
                drone.update_state(dt=self.time_step)

            # ========== 记录当前时间步的全量数据 ==========
            self._record_step_data(
                weights_matrix=weights_matrix,
                total_obj1=total_obj1,
                total_obj2=total_obj2,
                mopso_history=mopso_history,
                expected_yaw_list=expected_yaw_list,
                efv_list=efv_list,
            )

            # ========== 算法2-1第10步：时间步进 ==========
            self.current_time += self.time_step

        # ========== 仿真结束 ==========
        print("\n" + "=" * 60)
        print(
            f"✅ 仿真运行完成！总运行时间步：{len(self.simulation_data['time_steps'])}"
        )
        print("=" * 60)
        return self.simulation_data

    def _record_step_data(
        self,
        weights_matrix,
        total_obj1,
        total_obj2,
        mopso_history,
        expected_yaw_list,
        efv_list,
    ):
        """
        记录单个时间步的所有数据（内部方法）
        """
        swarm_state = self.swarm.get_swarm_state()
        # 记录基础信息
        self.simulation_data["time_steps"].append(self.current_time)
        self.simulation_data["weights_matrix"].append(weights_matrix)
        self.simulation_data["total_obj1"].append(total_obj1)
        self.simulation_data["total_obj2"].append(total_obj2)
        self.simulation_data["mopso_history"].append(mopso_history)
        # 记录无人机状态
        self.simulation_data["drone_positions"].append(swarm_state["positions"])
        self.simulation_data["drone_velocities"].append(swarm_state["velocities"])
        self.simulation_data["drone_yaws"].append(
            np.radians(swarm_state["yaws"])
        )  # 转弧度，和期望偏航角统一
        self.simulation_data["drone_expected_yaws"].append(np.array(expected_yaw_list))
        self.simulation_data["drone_expected_velocities"].append(np.array(efv_list))
        # 同步更新集群的历史记录
        self.swarm.record_history(self.current_time)

    def get_simulation_data(self):
        """获取仿真全量数据"""
        return self.simulation_data

    def get_swarm_history(self):
        """获取集群的轨迹历史"""
        return self.swarm.get_history()

    def save_simulation_data(self, save_path):
        """保存仿真数据到文件，方便后续可视化和神经网络训练"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.simulation_data, f)
        print(f" 仿真数据已保存至：{save_path}")
