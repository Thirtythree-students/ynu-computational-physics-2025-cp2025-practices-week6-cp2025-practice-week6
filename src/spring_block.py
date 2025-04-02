from shenji import solve_ode_euler, spring_mass_ode_func, solve_ode_odeint
import numpy as np

def test_solve_ode_euler():
    """测试欧拉法求解器"""
    step_num = 100
    time_points, position, velocity = solve_ode_euler(step_num)

    # 检查返回数组的长度
    assert len(time_points) == step_num + 1
    assert len(position) == step_num + 1
    assert len(velocity) == step_num + 1

def test_spring_mass_ode_func():
    """测试微分方程函数"""
    # 测试几个特定点
    test_cases = [
        ([0, 1], 0, [1, 0]),    # x=0, v=1
        ([1, 0], 0, [0, -1]),   # x=1, v=0
        ([-1, 0], 0, [0, 1]),   # x=-1, v=0
        ([0, -1], 0, [-1, 0]),  # x=0, v=-1
    ]

    for state, time, expected in test_cases:
        result = spring_mass_ode_func(state, time)
        assert np.allclose(result, expected)

def test_solve_ode_odeint():
    """测试 odeint 求解器"""
    step_num = 100
    time_points, position, velocity = solve_ode_odeint(step_num)

    # 检查返回数组的长度
    assert len(time_points) == step_num + 1
    assert len(position) == step_num + 1
    assert len(velocity) == step_num + 1
