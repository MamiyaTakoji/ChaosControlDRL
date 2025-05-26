from sympy import symbols, diff, lambdify, Matrix,Derivative,DiracDelta
#import cupy as cp
import numpy as np
class StateFunction:
    def __init__(self, F):
        namespace = {'DiracDelta': lambda *args: 0}
        # 使用正则表达式分离变量
        self.F = F
        vars = F[0].free_symbols
        l = len(F)
        self.l = l#状态函数的维度
        for i in range(l):
            vars = vars.union(F[i].free_symbols)
        x_vars = [var for var in vars if str(var).startswith('x')]
        x_vars = sorted(x_vars, key=lambda var: int(str(var).lstrip('x')))
        u_vars = [var for var in vars if str(var).startswith('u')]
        u_vars = sorted(u_vars, key=lambda var: int(str(var).lstrip('u')))
        self.x_vars = x_vars;self.u_vars = u_vars
        self.Fx = Matrix([[diff(F[i], x_var) for x_var in x_vars]for i in range(l)])
        self.Fu = Matrix([[diff(F[i], u_var) for u_var in u_vars]for i in range(l)])
        Fxx_Fist = []
        for i in range(l):
            Fxx = Matrix([[diff(diff(F[i], x1), x2) for x1 in x_vars] for x2 in x_vars])
            #Fxx_func = lambdify((x_vars, u_vars), Fxx, modules='numpy')
            Fxx_Fist.append(Fxx)
        self.Fxx = Fxx_Fist
        Fuu_Fist = []
        for i in range(l):
            Fuu = Matrix([[diff(diff(F[i], u1), u2) for u1 in u_vars] for u2 in u_vars])
            #Fuu_func = lambdify((x_vars, u_vars), Fuu, modules='numpy')
            Fuu_Fist.append(Fuu)
        self.Fuu = Fuu_Fist
        Fux_Fist = []
        for i in range(l):
            Fux = Matrix([[diff(diff(F[i], u1), x2) for u1 in u_vars] for x2 in x_vars])
            #Fux_func = lambdify((x_vars, u_vars), Fux, modules='numpy')
            Fux_Fist.append(Fux)
        self.Fxu = Fux_Fist
        self._F_func = lambdify((self.x_vars, self.u_vars), self.F, modules='numpy')
        self._Fx_func = lambdify((self.x_vars, self.u_vars), self.Fx, modules='numpy')
        self._Fu_func = lambdify((self.x_vars, self.u_vars), self.Fu, modules='numpy')
        Fxx_list = []
        for i in range(self.l):
            Fxx_list.append(lambdify((self.x_vars, self.u_vars), self.Fxx[i], modules=[namespace, 'numpy']))
        self._Fxx_func = Fxx_list
        Fux_list = []
        for i in range(self.l):
            Fux_list.append(lambdify((self.x_vars, self.u_vars), self.Fxu[i], modules='numpy'))
        self._Fux_func = Fux_list
        Fuu_list = []
        for i in range(self.l):
            Fuu_list.append(lambdify((self.x_vars, self.u_vars), self.Fuu[i], modules='numpy'))
        self._Fuu_func = Fuu_list
    def F_func(self,x,u):
        return np.array(self._F_func(x,u))
    def Fx_func(self,x,u):
        return np.array(self._Fx_func(x,u))
    def Fu_func(self,x,u):
        return np.array(self._Fu_func(x,u))
    def Fxx_func(self,x,u):
        return np.array([f(x,u) for f in self._Fxx_func])
    def Fxu_func(self,x,u):#不要太在意到底是Fxu还是Fux···知道是什么意思就行
        return np.array([f(x,u) for f in self._Fux_func])
    def Fuu_func(self,x,u):
        return np.array([f(x,u) for f in self._Fuu_func])
#下面来测试一下
# x = cp.array([[10,30,26],[20,40,10],[5,5,5],[10,10,10]])
# u = cp.array([[0,0,0],[0,0,0]])
# x_dim = x.shape[0]
# u_dim = u.shape[0]
# X = symbols(f'x:{x_dim}')
# U = symbols(f'u:{u_dim}')
# a = 0.008
# b = 0.016
# F = [a*X[0]*X[1]-b*X[0]+X[0]+U[0],
#      -a*X[0]*X[1]+b*X[0]+X[1]+U[1],
#      a*X[2]*X[3]-b*X[2]+X[2],
#      -a*X[2]*X[3]+b*X[2]+X[3]]
# l_Little = 1e-320*(X[0]*X[1]*X[2]*X[3]*U[0]*U[1])**2
# for i in range(len(F)):
#     F[i] = F[i]+l_Little
# StateFun = StateFunction(F)
# StateFun.Fxx_func(x, u)