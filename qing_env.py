# 为了强化学习而搭建的迷宫避障方格世界模型：
"""
包括：
1. 迷宫长宽，长是横向的，宽是纵向的。
2. 障碍物位置，可以通过，但是会有-1的惩罚。
3. 目标位置，到达目标位置有奖励+1。
"""

import gymnasium as gym # 引入gym库,继承gym库的环境类，方便调用gym库的函数
import numpy as np # 引入numpy库，方便使用numpy库的函数

class MyGridEnv(gym.Env):
    def __init__(self,size,target,barriers):
        """
        width: int, 迷宫的宽度
        height: int, 迷宫的高度
        target: tuple, 目标位置坐标(x,y)
        barriers: list, 障碍物位置坐标列表[(x1,y1),(x2,y2),...]
        start: tuple, 起始位置坐标(x,y)
        action_space: gym.spaces.Discrete, 动作空间，离散空间,5个动作，分别是上下左右,不动，0,1,2,3,4
        observation_space: gym.spaces.Discrete, 观测空间，离散空间，宽*高个状态,每个状态是一个格子
        reward: float, 奖励值
        done: bool, 是否结束
        agentloc: tuple, agent的位置坐标(x,y)
        """
        self.size = size
        # 定义迷宫的长宽size
        self.target = target
        self.barriers = barriers
        # 定义起始位置,设定为(0,0)
        self.start = (0,0)
        self.state = self.start
        # 定义动作空间，离散空间，5个动作，分别是上下左右,不动，0,1,2,3,4
        self.action_space = gym.spaces.Discrete(5)
        # 定义观测空间，agent的位置，目标位置，障碍物位置，用连续空间表示，方便之后用函数拟合。
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "barrier": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })
        self.done = False
        self.info = {}
        self.step_num = 0
        self.action_to_direction = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0),
        }
        self.model_psa = None
        self.model_rsa = None
        # 初始化环境模型，包括状态转移概率模型和奖励概率模型
        self.init_model()

    def init_model(self):
        # 状态转移概率模型，即P(s'|s,a)
        # 全部初始化为0 sx,sy,a,s'x,s'y,返回的是一个5维的数组
        self.model_psa = np.zeros((self.size, self.size, self.action_space.n, self.size, self.size))
        # 设置状态转移概率模型，把s,a 的下一个状态的概率设置成1
        for i in range(self.size):
            for j in range(self.size):
                for action in range(self.action_space.n):
                    next_state = (i + self.action_to_direction[action][0], j + self.action_to_direction[action][1])
                    if next_state[0] < 0 or next_state[0] >= self.size or next_state[1] < 0 or next_state[1] >= self.size:
                        self.model_psa[i, j, action, i, j] = 1 # 撞墙的情况,下一个状态还是自己
                        continue
                    self.model_psa[i, j, action, next_state[0], next_state[1]] = 1
        
        # 奖励概率模型，即R(s,a)
        # 全部初始化为0 sx,sy,a
        self.model_rsa = np.zeros((self.size, self.size, self.action_space.n))
        # 设置奖励概率模型，到达目标位置奖励为1，撞墙奖励为-1
        for i in range(self.size):
            for j in range(self.size):
                for action in range(self.action_space.n):
                    next_state = (i + self.action_to_direction[action][0], j + self.action_to_direction[action][1])
                    if next_state[0] < 0 or next_state[0] >= self.size or next_state[1] < 0 or next_state[1] >= self.size:
                        self.model_rsa[i, j, action] = -1
                    elif next_state in self.barriers:
                        self.model_rsa[i, j, action] = -1
                    elif next_state == self.target:
                        self.model_rsa[i, j, action] = 1

    def pssa(self, state, action, next_state):
        """
        输入参数：状态，动作，下一个状态，状态用(x,y)表示
        返回参数：从状态s执行动作a到达状态s'的概率
        """
        return self.model_psa[state[0], state[1], action, next_state[0], next_state[1]]
    
    def rsa(self, state, action):
        """
        输入参数：状态，动作
        返回参数：在状态s执行动作a的奖励
        """
        return self.model_rsa[state[0], state[1], action]
    
    def get_successors(self, state,action):
        """
        输入参数：状态,动作
        返回参数：状态s的后继状态,概率(不为0的返回，别的不返回)
        """
        successors = []
        for i in range(self.size):
            for j in range(self.size):
                if self.pssa(state,action,(i,j)) > 0:
                    successors.append(((i,j),self.pssa(state,action,(i,j))))
        return successors

    def reset(self,start=None):
        """
        重置环境
        """
        # 先调用父类的reset方法
        super().reset()
        # 重置agent的位置
        self.state = self.start
        self.done = False
        self.step_num = 0

    def set(self,state):
        """
        重置环境
        """
        # 重置agent的位置
        self.state = state
        self.done = False
        self.step_num = 0

    def step(self, action):
        """
        输入参数：动作
        返回参数：状态，奖励，是否结束，信息
        """
        reward = 0
        done = False
        # 获取动作对应的方向
        direction = self.action_to_direction[action]
        # 计算agent的位置
        next_state = (self.state[0] + direction[0], self.state[1] + direction[1])
        # 判断是否越界
        if next_state[0] < 0 or next_state[0] >= self.size or next_state[1] < 0 or next_state[1] >= self.size:
            reward = -1
            return self.state, reward, done, {}
        if next_state in self.barriers:
            reward = -1
            
        # 更新agent的位置
        self.state = next_state
        #self.state = next_state # 即使撞墙也要移动，只是会有-1的惩罚
        if self.state == self.target:
            reward = 1
            done = True
        return self.state, reward, done, {}
    


    def render(self, mode='human'):
        return None
        
    def close(self):
        return None        
    
