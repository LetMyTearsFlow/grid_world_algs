import numpy as np
from qing_env import MyGridEnv
"""
包含策略的定义
"""

class Policy:
    """
    策略的基类
    """
    def __init__(self, env: MyGridEnv):
        """
        初始化策略
        :param num_states: 状态数
        :param num_actions: 动作数
        """
        raise NotImplementedError
    
    def get_probs(self, state, action):
        """
        获取在给定状态下，所有动作的概率
        :param state: 给定的状态
        :return: 所有动作的概率
        """
        raise NotImplementedError

    def sample_action(self, state):
        """
        根据给定的状态，按照策略选择一个动作
        :param state: 给定的状态
        :return: 选择的动作
        """
        raise NotImplementedError

    def showpolicy(self):
        raise NotImplementedError
    
    def play(self):
        raise NotImplementedError
            

class TabularPolicy(Policy):

    def showpolicy(self):
        """
        用小方格的方式输出策略，每个小方格表示一个状态，如果是起始点则用'S'表示，如果是终止点则用'T'表示,障碍物用'X'表示，其他用箭头表示动作
        ，如果不动就用'*'表示。
        """
        for i in range(0, self.env.size):
            print('-'*(5+4*(self.env.size-1)))
            out = '| '
            for j in range(0, self.env.size):
                token=""
                # 选出概率最大的
                possibilities = self.probs[i][j]
                action = np.argmax(possibilities)
                if action == 0:
                    token = '↑'
                elif action == 1:
                    token = '↓'
                elif action == 2:
                    token = '←'
                elif action == 3:
                    token = '→'   
                else:
                    token = '*'            
                if (i,j) in self.env.barriers:
                    token = '■'
                #elif (i,j) == self.env.start:
                    #token = 'S'
                elif (i,j) == self.env.target:
                    token = 'T'
                out += token + ' | '
            print(out)
        print('-'*(5+4*(self.env.size-1)))

    def play(self):
        length = 0
        total_reward = 0
        while True:
            action = self.sample_action(self.env.state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            match action:
                case 0 :
                    arrow = '↑'
                case 1 :
                    arrow = '↓'
                case 2 :
                    arrow = '←'
                case 3 :
                    arrow = '→'
                case _ :
                    arrow = '*'
            print(state, reward, done, arrow)
            length += 1
            if done:
                break
        print("length:", length)
        print("total reward:", total_reward)
    
    def sample_action(self, state):
        """
        根据给定的状态，按照策略选择一个动作
        :param state: 给定的状态
        :return: 选择的动作
        """
        return np.random.choice(self.env.action_space.n, p=self.get_probs(state))

    def __init__(self, env: MyGridEnv):
        """
        初始化策略
        :param num_states: 状态数
        :param num_actions: 动作数
        """
        self.env = env
        self.probs = np.zeros((env.size, env.size, env.action_space.n))
        for i in range(env.size):
            for j in range(env.size):
                self.probs[i, j] = np.full(env.action_space.n, 1.0 / env.action_space.n)
    def get_probs(self, state,action=None):
        """
        获取在给定状态下，所有动作的概率
        :param state: 给定的状态
        :return: 所有动作的概率
        """
        if action is not None:
            return self.probs[state[0], state[1], action]
        return self.probs[state[0], state[1]]
    


