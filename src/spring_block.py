import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def solve_ode_euler(step_num, mass=1, k=1, initial_position=0,
                    initial_velocity=1):
    """
    使用欧拉法求解弹簧 - 质点系统的常微分方程。
    参数:
    step_num (int): 模拟的步数
    mass (float): 物块质量，默认为1
    k (float): 弹簧系数，默认为1
    initial_position (float): 初始位置，默认为0
    initial_velocity (float): 初始速度，默认为1
    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 创建存储位置和速度的数组，长度为step_num + 1
    position = np.zeros(step_num + 1)
    velocity = np.zeros(step_num + 1)
    # 计算时间步长
    time_step = 2 * np.pi / step_num
    # 设置初始位置和速度
    position[0] = initial_position
    velocity[0] = initial_velocity
    # 使用欧拉法迭代求解微分方程
    for i in range(step_num):
        # 根据微分方程更新位置
        position[i + 1] = position[i] + velocity[i] * time_step
        # 根据微分方程更新速度
        velocity[i + 1] = velocity[i] - (k / mass) * position[i] * time_step
    # 生成时间数组
    time_points = np.arange(step_num + 1) * time_step
    return time_points, position, velocity


def spring_mass_ode_func(state, time, mass=1, k=1):
    """
    定义弹簧 - 质点系统的常微分方程。
    参数:
    state (list): 包含位置和速度的列表
    time (float): 时间
    mass (float): 物块质量，默认为1
    k (float): 弹簧系数，默认为1
    返回:
    list: 包含位置和速度的导数的列表
    """
    position, velocity = state
    d_position_dt = velocity
    d_velocity_dt = - (k / mass) * position
    return [d_position_dt, d_velocity_dt]


def solve_ode_odeint(step_num, mass=1, k=1,
                     initial_position=0, initial_velocity=1):
    """
    使用odeint求解弹簧 - 质点系统的常微分方程。
    参数:
    step_num (int): 模拟的步数
    mass (float): 物块质量，默认为1
    k (float): 弹簧系数，默认为1
    initial_position (float): 初始位置，默认为0
    initial_velocity (float): 初始速度，默认为1
    返回:
    tuple: 包含时间数组、位置数组和速度数组的元组
    """
    # 设置初始条件
    initial_state = [initial_position, initial_velocity]
    # 创建时间点数组
    time_points = np.linspace(0, 2 * np.pi, step_num + 1)
    # 使用odeint求解微分方程
    solution = odeint(spring_mass_ode_func, initial_state,
                      time_points, args=(mass, k))
    # 从解中提取位置和速度
    position = solution[:, 0]
    velocity = solution[:, 1]
    return time_points, position, velocity


def plot_ode_solutions(time_euler, position_euler, velocity_euler,
                       time_odeint, position_odeint, velocity_odeint):
    """
    绘制欧拉法和odeint求解的位置和速度随时间变化的图像。
    参数:
    time_euler (np.ndarray): 欧拉法的时间数组
    position_euler (np.ndarray): 欧拉法的位置数组
    velocity_euler (np.ndarray): 欧拉法的速度数组
    time_odeint (np.ndarray): odeint的时间数组
    position_odeint (np.ndarray): odeint的位置数组
    velocity_odeint (np.ndarray): odeint的速度数组
    """
    # 创建图形并设置大小
    plt.figure(figsize=(12, 6))

    # 绘制位置对比图
    plt.subplot(1, 2, 1)
    plt.plot(time_euler, position_euler, 'ro--', label='Euler Position')
    plt.plot(time_odeint, position_odeint, 'b-', label='ODEint Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position Comparison')
    plt.legend()
    plt.grid(True)

    # 绘制速度对比图
    plt.subplot(1, 2, 2)
    plt.plot(time_euler, velocity_euler, 'gs--', label='Euler Velocity')
    plt.plot(time_odeint, velocity_odeint, 'm-', label='ODEint Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity Comparison')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟步数
    step_count = 100
    # 使用欧拉法求解
    time_euler, position_euler, velocity_euler = solve_ode_euler(step_count)
    # 使用odeint求解
    time_odeint, position_odeint, velocity_odeint = solve_ode_odeint(
        step_count)
    # 绘制对比结果
    plot_ode_solutions(time_euler, position_euler, velocity_euler,
                       time_odeint, position_odeint, velocity_odeint)
