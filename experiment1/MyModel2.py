# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:00:47 2024

@author: Mamiya
"""
import numpy as np
from igraph import Graph
from StateFunction import StateFunction

class MyModel2(StateFunction):
    def __init__(self,f,stateNum):
        super(MyModel2, self).__init__(F = f)
        self.state_num = stateNum
    def OneStepiteration(self,x,u):
        y = self.F_func(x, u)
        return y
    def OneStepiterationOnGraph(self,x,u):
        #输入：1*n大小的array
        y = self.OneStepiteration([x], [0])[0]
        y = self.GetState(y)
        y = y + u
        #输出：1*n大小的array
        return y
    def GetState(self,real_state):
        state = np.round(self.state_num*real_state)/self.state_num
        return state
    def GetGraphId(self,state):
        graphId = state*self.state_num
        return np.round(graphId).astype(int).tolist()
    def GraphId2State(self,graphId:int):
        state = graphId/self.state_num
        return state
    def SetStateMapingNetWork(self,ControlLim):
        X = np.linspace(0, 1 ,self.state_num+1)
        U = np.linspace(0, 0 ,self.state_num+1)
        Y = self.GetState(self.OneStepiteration(X.reshape(1,self.state_num+1),U.reshape(1,self.state_num+1)))
        Y = Y[0]
        G = np.abs(Y[:,np.newaxis]-X)<=ControlLim+0.0000001
        self.G = Graph.Adjacency(G)
    def GetControlItem(self,Start,Target):
        StartState = self.GetState(Start);TargetState = self.GetState(Target)
        StartId = self.GetGraphId(StartState)
        TargetId = self.GetGraphId(TargetState)
        G = self.G
        if not hasattr(self,"all_shortest_paths"):
            self.all_shortest_paths = np.array(G.shortest_paths(source=range(G.vcount()), target=TargetId))
            self.lastTargetId = TargetId
        if TargetId != self.lastTargetId:
            self.all_shortest_paths = np.array(G.shortest_paths(source=range(G.vcount()), target=TargetId))
            self.lastTargetId = TargetId
        if not hasattr(self,"adj_list"):
            self.adj_list = self.G.get_adjlist()
        adj_list = self.adj_list
        all_shortest_paths = self.all_shortest_paths
        all_shortest_paths = np.append(all_shortest_paths, np.inf)
        try:           
            PointList = [adj_list[Id] for Id in StartId]
            subDist = all_shortest_paths[PointList]
        except:
            maxL = 0;outer = len(all_shortest_paths)-1
            for i in range(len(PointList)):
                if maxL<len(PointList[i]):
                    maxL = len(PointList[i])
            for i in range(len(PointList)):
                if maxL>len(PointList[i]):
                    while len(PointList[i])<maxL:
                        PointList[i].append(outer)
            subDist = all_shortest_paths[PointList]
        if subDist.ndim == 1:
            subDist = np.array([subDist])
        index = np.argmin(subDist,axis=1)
        #可能是numpy升级后语法变了，这里需要修改一下
        #2024.11.4
        
        #print(min(subDist))
        #NextId = np.array([PointList[i][index[i][0]] for i in range(len(PointList))])
        NextId = np.array([PointList[i][index[i]] for i in range(len(PointList))])
        #NextId = PointList[index]
        n = self.state_num
        u = 1/n*NextId-self.OneStepiteration([Start],[0])[0] + self.H(1/(2*n) - abs(Target-1/n*NextId))*(Target-1/n*NextId)
        #u = 1/n*NextId-Start
        return u,NextId
    def H(self,x):
        # if x>= 0:
        #     return 1
        # else:
        #     return 0
        result = np.zeros(x.shape[0])
        result[x>=0] = 1
        return result
        
        
        
        
        
        
        
        
        
        
        
        