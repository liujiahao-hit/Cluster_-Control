"""
核心逻辑：基于MOPSO优化的无人机集群避障与速度跟踪仿真
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os
import sys
import time
import pickle

# 把src文件夹加入Python搜索路径，解决导入错误
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# 导入你已有的模块，100%匹配之前的代码
from src.drone_swarm import DroneSwarm
from src.flocking_controller import FlockingController
from src.mopso_optimizer import MOPSOSwarmWeightOptimizer


# ===================== 可视化设置（完全兼容你原有的样式） =====================
def setup_visualization():
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


# ===================== 仿真主逻辑（100%对齐论文算法2-1与实验参数） =====================
def run_simulation(initial_positions=None, obstacles=None, max_simulation_time=100.0):
    """论文2.6.1节的完整仿真实验
    
    参数:
        initial_positions: 自定义初始位置列表，如果为None则使用默认值
        obstacles: 自定义障碍物列表，如果为None则使用默认值
        max_simulation_time: 仿真时间，如果为None则使用默认值
    """
    # ===================== 论文实验固定参数（表2-1+正文参数） =====================
    # 默认障碍物列表（12个障碍物）
    default_obstacles = [
        [100, 40, 5],
        [230, 40, 5],
        [165, 75, 5],
        [295, 75, 5],
        [100, 110, 5],
        [230, 110, 5],
        [165, 150, 5],
        [295, 150, 5],
        [100, 180, 5],
        [230, 180, 5],
        [165, 215, 5],
        [295, 215, 5],
    ]
    
    params = {
        "num_drones": 5,  # 论文固定5架无人机
        "max_simulation_time": max_simulation_time,  # 使用传入的仿真时间
        "time_step": 0.5,  # 论文采样时间ts=0.5s
        "mopso_max_generations": 58,  # MOPSO迭代次数
        "mopso_particle_num": 20,  # MOPSO粒子数量
        # 无人机初始位置
        "initial_positions": initial_positions if initial_positions is not None else [
            [22.0222, 107.9526],
            [25.5856, 114.5043],
            [24.6951, 116.9755],
            [10.5040, 113.8653],
            [8.4631, 120.6620],
        ],
        # 无人机初始速度
        "initial_velocities": [
            [10.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
        ],
        # 障碍物列表
        "obstacles": obstacles if obstacles is not None else default_obstacles,
        # 期望速度
        "desired_velocity": np.array([10.0, 0.0]),
        "desired_speed": 10.0,
    }

    print("=" * 60)
    print("基于MOPSO优化的无人机集群避障与速度跟踪仿真")
    print("100%对齐论文2.6.1节实验参数")
    print("=" * 60)
    print(f"仿真参数：5架无人机 | 100秒仿真 | 步长0.5s | MOPSO 58代/20粒子")
    print(f"障碍物数量：{len(params['obstacles'])}个")
    print("-" * 60)

    # ===================== 初始化集群与控制器 =====================
    # 初始化无人机集群
    swarm = DroneSwarm(
        num_drones=params["num_drones"],
        initial_positions=params["initial_positions"],
        initial_velocities=params["initial_velocities"],
    )
    swarm.set_obstacles(params["obstacles"])

    # 初始化集群控制器（以第一架无人机为参考，复用drone.py的期望参数）
    reference_drone = swarm.drones[0]
    controller = FlockingController(
        reference_drone=reference_drone, num_drones=params["num_drones"]
    )

    # ===================== 仿真参数初始化 =====================
    num_steps = int(params["max_simulation_time"] / params["time_step"])
    current_time = 0.0

    # ===================== 仿真数据记录容器（可视化/后续分析用） =====================
    simulation_data = {
        "time_steps": [],
        "weights_matrix": [],
        "total_obj1": [],
        "total_obj2": [],
        "drone_positions": [],
        "drone_velocities": [],
        "drone_yaws": [],
        "drone_expected_yaws": [],
        "core_expected_yaw": [],  # 集群核心期望偏航角（论文公式2-22）
        # 新增：水平空速数据
        "drone_speeds": [],
        # 新增：无人机对相对距离数据
        "distance_pairs": [],
        "distance_pairs_labels": [],
        # 新增：避障信息
        "S_o_not_empty": [],
        "S_o": [], # 障碍物集合
        # 新增：控制输入数据
        "control_inputs": [],
        # 新增：期望速度数据
        "efv_list": [],
    }

    # ===================== 主仿真循环（严格对齐论文算法2-1） =====================
    for step in range(num_steps):
        current_time = step * params["time_step"]
        print(f"  步数 {step+1:3d}/{num_steps} | 时间 {current_time:5.1f}s", end="\r")

        # ========== 步骤1：计算集群核心状态与分段期望偏航角（论文公式2-22） ==========
        Q_core = controller._compute_swarm_core_position(swarm)
        S_o, S_o_not_empty = controller.compute_S_o_set(Q_core, params["obstacles"])

        # 严格按照论文公式(2-22)计算集群核心期望偏航角δ^core
        if current_time <= 50.0:
            # t≤50s：避障阶段，由障碍物计算核心偏航角
            delta_core = controller.compute_core_yaw(Q_core, S_o,current_time=current_time )
        elif 50.0 < current_time <= 80.0:
            # 50s<t≤80s：正弦动态偏航阶段（公式2-22第一段）
            delta_core = -np.pi / 3 * np.sin(2 * np.pi * (current_time - 50) / 30)
        else:
            # t>80s：回到期望偏航角（公式2-22第二段）
            delta_core = np.arctan2(
                params["desired_velocity"][1], params["desired_velocity"][0]
            )

        # ========== 步骤2：计算每架无人机的期望偏航角与期望速度 ==========
        swarm_state = swarm.get_swarm_state()
        all_positions = swarm_state["positions"]
        all_velocities = swarm_state["velocities"]
        efv_list = []
        expected_yaw_list = []

        for drone_i in range(params["num_drones"]):
            drone_pos = all_positions[drone_i]
            # 公式(2-8)计算单无人机期望偏航角δ^i
            delta_i = controller.compute_drone_yaw(drone_pos, Q_core, delta_core)
            # 公式(2-7)计算期望速度efv^i
            efv_i = controller.compute_efv(delta_i)
            expected_yaw_list.append(delta_i)
            efv_list.append(efv_i)

        # ========== 步骤3：MOPSO优化计算最优权重矩阵 ==========
        mopso_swarm_state = {
            "positions": all_positions,
            "velocities": all_velocities,
            "S_o_not_empty": S_o_not_empty,
            "efv_list": efv_list,
        }
        weights_matrix, total_obj1, total_obj2, _ = controller.update_weights_matrix(
            mopso_swarm_state, current_time=current_time
        )

        # ========== 步骤4：计算每架无人机的控制输入 ==========
        control_input_list = []
        for drone_i in range(params["num_drones"]):
            u_i = controller.compute_control_input(
                drone_i=drone_i,
                drone_pos=all_positions[drone_i],
                drone_vel=all_velocities[drone_i],
                efv_i=efv_list[drone_i],
                all_positions=all_positions,
                all_velocities=all_velocities,
            )
            control_input_list.append(u_i)

        # ========== 步骤5：更新无人机状态 ==========
        for drone_i in range(params["num_drones"]):
            drone = swarm.drones[drone_i]
            drone.set_control_input(control_input_list[drone_i])
            drone.update_state(dt=params["time_step"])

        # ========== 记录当前时间步数据 ==========
        swarm.record_history(current_time)
        simulation_data["time_steps"].append(current_time)
        simulation_data["weights_matrix"].append(weights_matrix)
        simulation_data["total_obj1"].append(total_obj1)
        simulation_data["total_obj2"].append(total_obj2)
        simulation_data["drone_positions"].append(all_positions)
        simulation_data["drone_velocities"].append(all_velocities)
        simulation_data["drone_yaws"].append(np.radians(swarm_state["yaws"]))
        simulation_data["drone_expected_yaws"].append(np.array(expected_yaw_list))
        simulation_data["core_expected_yaw"].append(delta_core)
        
        # 新增：记录水平空速
        speeds = [np.linalg.norm(v) for v in all_velocities]
        simulation_data["drone_speeds"].append(speeds)
        
        # 新增：记录避障信息
        simulation_data["S_o_not_empty"].append(S_o_not_empty)
        simulation_data["S_o"].append(S_o)
        
        # 新增：记录控制输入
        simulation_data["control_inputs"].append(control_input_list)
        
        # 新增：记录期望速度
        simulation_data["efv_list"].append(efv_list)

    # ===================== 仿真结束 =====================
    print("\n" + "-" * 60)
    print("✅ 仿真运行完成！")
    return swarm, params, simulation_data


# ===================== 可视化模块（1:1对齐论文结果图，保留你原有的样式） =====================
def visualize_results(swarm, params, simulation_data):
    """生成和论文完全一致的2行3列仿真结果图"""
    # 创建结果保存文件夹
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(
        results_dir, f"MOPSO集群仿真结果_{time.strftime('%Y%m%d_%H%M%S')}.png"
    )

    try:
        # 提取仿真数据
        time_history = np.array(simulation_data["time_steps"])
        positions_history = np.array(simulation_data["drone_positions"])
        velocities_history = np.array(simulation_data["drone_velocities"])
        yaw_history = np.array(simulation_data["drone_yaws"])
        core_expected_yaw_history = np.array(simulation_data["core_expected_yaw"])
        obj1_history = np.array(simulation_data["total_obj1"])
        obj2_history = np.array(simulation_data["total_obj2"])

        # 基础参数
        num_drones = params["num_drones"]
        desired_speed = params["desired_speed"]
        D_l1 = 2.0  # 最小安全距离
        D_d = 10.0  # 期望邻居距离

        # 配色1:1匹配论文示例
        drone_colors = ["#1E90FF", "#FF4500", "#32CD32", "#FFD700", "#8A2BE2"]
        drone_labels = [f"UAV{i+1}" for i in range(num_drones)]

        # 预计算绘图数据
        # 1. 每架无人机的水平空速
        speed_history = np.zeros((len(time_history), num_drones))
        for step in range(len(time_history)):
            for drone_i in range(num_drones):
                speed_history[step, drone_i] = np.linalg.norm(
                    velocities_history[step][drone_i]
                )

        # 2. 无人机间相对距离（每对无人机的距离曲线，和论文一致）
        # 计算所有无人机对的索引
        drone_pairs = []
        pair_labels = []
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                drone_pairs.append((i, j))
                pair_labels.append(f"RD{i+1}{j+1}")
        
        # 计算每对无人机的距离历史
        distance_pairs_history = np.zeros((len(time_history), len(drone_pairs)))
        for step in range(len(time_history)):
            positions = positions_history[step]
            for idx, (i, j) in enumerate(drone_pairs):
                distance_pairs_history[step, idx] = np.linalg.norm(positions[i] - positions[j])
        
        # 保存相对距离数据到simulation_data
        simulation_data["distance_pairs"] = distance_pairs_history
        simulation_data["distance_pairs_labels"] = pair_labels
        
        # 计算最小和平均距离（用于额外分析）
        min_distance_history = np.zeros(len(time_history))
        avg_distance_history = np.zeros(len(time_history))
        for step in range(len(time_history)):
            min_distance_history[step] = np.min(distance_pairs_history[step])
            avg_distance_history[step] = np.mean(distance_pairs_history[step])

        # 创建画布，2行3列，和论文完全一致
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 10
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#f0f0f5")
        fig.suptitle(
            "基于MOPSO优化的5架无人机集群避障与速度跟踪仿真结果",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------- (a) 轨迹曲线 --------------------------
        ax1 = axes[0, 0]
        # 绘制障碍物（红色，和论文一致）
        for obs in params["obstacles"]:
            # 障碍物本体
            ax1.add_patch(
                Circle(
                    (obs[0], obs[1]),
                    obs[2],
                    facecolor="#ff6666",
                    alpha=0.7,
                    edgecolor="#ff0000",
                )
            )
            # 障碍物包络线（黑色虚线圆，和论文图2-2一致）
            # 包络线半径 = 障碍物半径 + 无人机与障碍物最小避碰距离D_l2
            ax1.add_patch(
                Circle(
                    (obs[0], obs[1]),
                    obs[2] + 10.0 + 5.0,  # D_l2=10.0，无人机与障碍物最小避碰距离
                    facecolor="none",
                    edgecolor="black",
                    linestyle="--",
                    linewidth=2,
                    zorder=5,
                )
            )
        # 绘制每架无人机的轨迹、起点、终点
        for i in range(num_drones):
            traj = positions_history[:, i, :]
            ax1.plot(
                traj[:, 0],
                traj[:, 1],
                color=drone_colors[i],
                linewidth=2,
                label=drone_labels[i],
            )
            ax1.scatter(
                traj[0, 0],
                traj[0, 1],
                color=drone_colors[i],
                s=80,
                marker="o",
                edgecolor="black",
                zorder=10,
            )
            ax1.scatter(
                traj[-1, 0],
                traj[-1, 1],
                color=drone_colors[i],
                s=80,
                marker="s",
                edgecolor="black",
                zorder=10,
            )
        # 添加包络线标记（红色矩形）
        ax1.add_patch(
            Rectangle(
                (0, 40),  # 左下角坐标
                300, 140,  # 宽度和高度
                facecolor="none",
                edgecolor="red",
                linestyle="--",
                linewidth=2,
                zorder=5,
            )
        )
        # 坐标轴设置（和论文一致）
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(0, 220)
        ax1.set_xlabel("$Q_x$ (m)")
        ax1.set_ylabel("$Q_y$ (m)")
        ax1.set_title("(a) 轨迹曲线")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(alpha=0.3, linestyle="-")
        ax1.set_facecolor("white")

        # -------------------------- (b) 水平空速曲线 --------------------------
        ax2 = axes[0, 1]
        for i in range(num_drones):
            ax2.plot(
                time_history,
                speed_history[:, i],
                color=drone_colors[i],
                linewidth=1.5,
                label=drone_labels[i],
            )
        ax2.axhline(
            y=desired_speed,
            color="green",
            linestyle="--",
            linewidth=2,
            label="期望水平空速",
        )
        # 坐标轴设置（和论文一致，但纵轴范围扩大）
        ax2.set_xlim(0, 100)
        ax2.set_ylim(5.0, 15.0)  # 纵轴范围扩大，从5.0到15.0
        ax2.set_xlabel("$t$ (s)")
        ax2.set_ylabel("$V_{xy}$ (m/s)")
        ax2.set_title("(b) 水平空速曲线")
        ax2.legend(loc="upper right", fontsize=9, ncol=2)
        ax2.grid(alpha=0.3, linestyle="-")
        ax2.set_facecolor("white")

        # -------------------------- (c) 偏航角曲线 --------------------------
        ax3 = axes[0, 2]
        for i in range(num_drones):
            ax3.plot(
                time_history,
                yaw_history[:, i],
                color=drone_colors[i],
                linewidth=1.5,
                label=drone_labels[i],
            )
        ax3.plot(
            time_history,
            core_expected_yaw_history,
            color="green",
            linestyle="--",
            linewidth=2,
            label="期望集群偏航角",
        )
        # 坐标轴设置（和论文一致）
        ax3.set_xlim(0, 100)
        ax3.set_ylim(-1.2, 2.3)  # 修改纵轴范围为-1.2到2.3
        ax3.set_xlabel("$t$ (s)")
        ax3.set_ylabel("偏航角 (rad)")
        ax3.set_title("(c) 偏航角曲线")
        ax3.legend(loc="upper right", fontsize=9, ncol=2)
        ax3.grid(alpha=0.3, linestyle="-")
        ax3.set_facecolor("white")

        # -------------------------- (d) 相对距离曲线 --------------------------
        ax4 = axes[1, 0]
        # 绘制每对无人机的距离曲线
        for idx, (i, j) in enumerate(drone_pairs):
            ax4.plot(
                time_history,
                distance_pairs_history[:, idx],
                linewidth=1.5,
                label=pair_labels[idx],
            )
        # 只保留最小安全距离参考线，删除期望距离曲线
        ax4.axhline(
            y=D_l1,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label="$D_{l1}$ (最小安全距离)",
        )
        # 坐标轴设置（和论文一致）
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 30)
        ax4.set_xlabel("$t$ (s)")
        ax4.set_ylabel("相对距离 (m)")
        ax4.set_title("(d) 相对距离曲线")
        ax4.legend(loc="upper right", fontsize=8, ncol=2)
        ax4.grid(alpha=0.3, linestyle="-")
        ax4.set_facecolor("white")

        # -------------------------- (e) 集群速度目标函数 --------------------------
        ax5 = axes[1, 1]
        ax5.plot(
            time_history,
            obj1_history,
            color="#ff6666",
            linewidth=2,
            label=r"$\sum obj_1^i$",
        )
        # 设置不同时间段的期望值
        # 0-20秒和50-80秒期望值为-25，其他时间段为0
        ax5.axhline(y=-25, xmin=0, xmax=0.3, color="red", linestyle="--", linewidth=2, label="期望值")
        ax5.axhline(y=0, xmin=0.3, xmax=0.5, color="red", linestyle="--", linewidth=2)
        ax5.axhline(y=-25, xmin=0.5, xmax=0.8, color="red", linestyle="--", linewidth=2)
        ax5.axhline(y=0, xmin=0.8, xmax=1.0, color="red", linestyle="--", linewidth=2)
        
        # 在分界点添加垂直连接线
        ax5.vlines(x=30, ymin=0, ymax=-25, color="red", linestyle="--", linewidth=2)
        ax5.vlines(x=50, ymin=-25, ymax=0, color="red", linestyle="--", linewidth=2)
        ax5.vlines(x=80, ymin=0, ymax=-25, color="red", linestyle="--", linewidth=2)
        # 坐标轴设置（和论文一致）
        ax5.set_xlim(0, 100)
        ax5.set_ylim(-30, 10)  # 调整为和论文一致的范围
        ax5.set_xlabel("$t$ (s)")
        ax5.set_ylabel(r"$\sum obj_1^i$")
        ax5.set_title(r"(e) $\sum obj_1^i$曲线")
        ax5.legend(loc="upper right", fontsize=9)
        ax5.grid(alpha=0.3, linestyle="-")
        ax5.set_facecolor("white")

        # -------------------------- (f) 集群结构目标函数 --------------------------
        ax6 = axes[1, 2]
        ax6.plot(
            time_history,
            obj2_history,
            color="#6666ff",
            linewidth=2,
            label=r"$\sum obj_2^i$",
        )
        ax6.axhline(y=45, color="red", linestyle="--", linewidth=2, label="期望值")
        # 坐标轴设置（和论文一致）
        ax6.set_xlim(0, 100)
        ax6.set_ylim(0, 150)  # 调整为和论文一致的范围
        ax6.set_xlabel("$t$ (s)")
        ax6.set_ylabel(r"$\sum obj_2^i$")
        ax6.set_title(r"(f) $\sum obj_2^i$曲线")
        ax6.legend(loc="upper right", fontsize=9)
        ax6.grid(alpha=0.3, linestyle="-")
        ax6.set_facecolor("white")

        # 保存并显示图片
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="#f0f0f5")
        print(f"\n✅ 仿真结果图已保存到：{save_path}")
        plt.show()
        plt.close()

        # 保存仿真数据（方便后续分析/神经网络训练）
        data_save_path = os.path.join(results_dir, "simulation_data.pkl")
        with open(data_save_path, "wb") as f:
            pickle.dump(simulation_data, f)
        print(f"✅ 仿真原始数据已保存到：{data_save_path}")

    except Exception as e:
        print(f"\n❌ 可视化出错！错误信息：{str(e)}")
        import traceback

        traceback.print_exc()


# ===================== 主函数 =====================
def main():
    setup_visualization()
    try:
        swarm, params, simulation_data = run_simulation()
        visualize_results(swarm, params, simulation_data)
    except Exception as e:
        print(f"\n❌ 运行错误：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
