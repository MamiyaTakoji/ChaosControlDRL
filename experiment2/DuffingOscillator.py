#定义参数
import numpy as np
xi = 0.04
omega = 0.84
F = 0.188
period = 2 * np.pi / omega
dt = 0.005*period
#求解器使用自己写的，Python那个不适合反复调用
from ODESolver import ODESolver
class model:
    def DuffingOscillator(self,t, X, u):
        #t是一个数字
        #X输入可以是2，也可以是2*n的
        #u是数字或者长度为n的向量
        x = X[0]; x_dot = X[1]
        x_ddot = -2*xi*x_dot + 0.5*x -0.5*x*x*x + (F+u)*omega*omega*np.sin(omega*t)
        return np.array([x_dot, x_ddot])
    # def DuffingOscillatorMulti(self, t, X, u):
    #     #假定输入的时间仍然为一个数，而X是[n, 2]的
    #     x = X[:,0]; x_dot = X[:,1]
    #     x_ddot = -2*xi*x_dot + 0.5*x -0.5*x*x*x + (F+u)*omega*omega*np.sin(omega*t)
    #     return np.hstack((x_dot, x_ddot))
    def Poincare(self,X,u):
        #每次都从0开始直到period
        t_span = [0, period]
        Xcopy = X.copy()
        solver = ODESolver(self.DuffingOscillator, t_span, Xcopy, dt, u)
        solver.integrate()
        solution = solver.get_solution()
        return solution[1][-1]
    def PoincareMulti(self, X, u):
        t_span = [0, period]
        Xcopy = X.copy()
        solver = ODESolver(self.DuffingOscillator, t_span, Xcopy, dt, u)
        solver.integrate()
        solution = solver.get_solution()
        return solution[1][-1,:,:]
if __name__ == "__main__":
    m = model()
    pointNum = 50
    x = np.linspace(-0.5, 0.5, pointNum)
    y = np.linspace(-0.5, 0.5, pointNum)
    X, Y = np.meshgrid(x, y)
    X = X.reshape([pointNum*pointNum])
    Y = Y.reshape([pointNum*pointNum])
    points = np.vstack((X, Y))
    u = np.zeros(pointNum*pointNum)
    # new_x = m.Poincare(x, u)
    #绘制庞加莱映射看看
    pointList = [points]
    import matplotlib.pyplot as plt
    # # t_span = [0, 10*period]
    # # X = np.array([0,0])
    # # solver = ODESolver(m.DuffingOscillator, t_span, X, dt, u)
    # # solver.integrate()
    # # solution = solver.get_solution()
    # # plt.plot(solution[1][:,0], solution[1][:,1])
    for i in range(100):
        print(i)
        x = m.PoincareMulti(pointList[-1].copy(), u)
        pointList.append(x)
    points = np.array(pointList)
    plt.scatter(points[-1,0,:], points[-1,1,:], s = 1, c = 'r')
    # t_span = [0, 100*period]
    # X = np.array([0,0])
    # solver = ODESolver(m.DuffingOscillator, t_span, X, dt, u)
    # solver.integrate()
    # solution = solver.get_solution()
    # plt.plot(solution[1][:,0], solution[1][:,1])
    x = points[:,0]; x_dot = points[:,1]
    mean_x = np.mean(x); std_x = np.std(x)
    print(f"x的均值是：{mean_x},x的方差是{std_x}")
    mean_x_dot = np.mean(x_dot); std_x_dot = np.std(x_dot)
    print(f"x_dot的均值是：{mean_x_dot},x_dot的方差是{std_x_dot}")















