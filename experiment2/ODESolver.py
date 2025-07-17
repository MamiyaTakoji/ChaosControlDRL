# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:25:31 2025

@author: Mamiya
"""

import numpy as np
import matplotlib.pyplot as plt

class ODESolver:
    def __init__(self, ode_func, t_span, y0, step_size=0.1, params=None):
        """
        自定义ODE求解器
        
        参数:
        ode_func - 微分方程函数，形式为: f(t, y, params)
        t_span   - 时间区间 (t_start, t_end)
        y0       - 初始状态
        step_size- 步长
        params   - 参数字典 (可在求解过程中更新)
        """
        self.ode_func = ode_func
        self.t_start, self.t_end = t_span
        self.t_current = self.t_start
        self.y_current = np.array(y0, dtype=float)
        self.step_size = step_size
        self.params = params if params is not None else {}
        
        # 存储结果
        self.t_points = [self.t_current]
        self.y_points = [self.y_current.copy()]
        self.param_history = [self.params]
        
        # 计数器
        self.step_count = 0
        
    def step(self):
        """执行单个积分步"""
        if self.t_current >= self.t_end:
            return False
        
        # 龙格-库塔4阶方法 (RK4)
        h = self.step_size
        k1 = self.ode_func(self.t_current, self.y_current, self.params)
        k2 = self.ode_func(self.t_current + h/2, self.y_current + h/2 * k1, self.params)
        k3 = self.ode_func(self.t_current + h/2, self.y_current + h/2 * k2, self.params)
        k4 = self.ode_func(self.t_current + h, self.y_current + h * k3, self.params)
        
        # 更新状态
        self.y_current += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.t_current += h
        
        # 存储结果
        self.t_points.append(self.t_current)
        self.y_points.append(self.y_current.copy())
        self.param_history.append(self.params)
        self.step_count += 1
        
        return True
    
    def integrate(self):
        """执行完整积分"""
        while self.step():
            pass
            
    def update_params(self, new_params):
        """在求解过程中更新参数"""
        self.params.update(new_params)
        
    def get_solution(self):
        """获取完整解"""
        return np.array(self.t_points), np.array(self.y_points)
    
    def plot(self, **kwargs):
        """绘制解"""
        t, y = self.get_solution()
        plt.figure(figsize=(10, 6))
        
        # 如果是向量系统，绘制每个分量
        if y.ndim > 1 and y.shape[1] > 1:
            for i in range(y.shape[1]):
                plt.plot(t, y[:, i], label=f'y[{i}]')
            plt.legend()
        else:
            plt.plot(t, y)
            
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.title('ODE Solution')
        plt.grid(True)
        plt.show()
# 定义洛伦兹系统
def lorenz_system(t, y, params):
    sigma = params.get('sigma', 10.0)
    rho = params.get('rho', 28.0)
    beta = params.get('beta', 8.0/3.0)
    
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])
if __name__ == "__main__":
    # 创建求解器
    solver = ODESolver(
        lorenz_system, 
        t_span=(0, 50),
        y0=[1.0, 1.0, 20.0],
        step_size=0.01,
        params={'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0}
    )
    
    # 手动控制积分过程（可随时修改参数）
    for _ in range(100):  # 先积分100步
        solver.step()
    
    # 修改参数
    print("修改参数rho为40...")
    solver.update_params({'rho': 40.0})
    
    for _ in range(100):  # 再积分100步
        solver.step()
        
    print("修改参数sigma为15...")
    solver.update_params({'sigma': 15.0})
    
    # 完成剩余积分
    solver.integrate()
    
    # 绘制结果
    solver.plot()
    
    # 提取解进行分析
    t, y = solver.get_solution()
    print(f"最终状态: t={t[-1]}, x={y[-1, 0]}, y={y[-1, 1]}, z={y[-1, 2]}")