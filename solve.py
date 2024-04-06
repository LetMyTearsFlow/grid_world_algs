"""
存放各种强化学习算法的实现，依赖环境和策略两个模块。
"""
import numpy as np
from policies import TabularPolicy
from qing_env import MyGridEnv


class Solve:
    def __init__(self, policy):
        self.policy = policy
        self.state_values = np.zeros((self.policy.env.size,self.policy.env.size))
        self.action_values = np.zeros((self.policy.env.size,self.policy.env.size,self.policy.env.action_space.n))
        self.threshold = 1e-3
        self.discounted_rate = 0.9


    def train(self):
        pass

    def test(self):
        pass

class ValueIteration(Solve):
    def __init__(self, policy):
        super().__init__(policy)
        # 初始化价值函数和动作函数

    def train(self):
        threshold = 1e-3
        discounted_rate = 0.9
        training_step = 0
        while True:
            previous_state_values = np.copy(self.state_values)
            #对于每个状态：
            for i in range(self.policy.env.size):
                for j in range(self.policy.env.size):
                    #对于每个动作：
                    for action in range(self.policy.env.action_space.n):
                        # 计算动作价值函数
                        qk_sa = 0
                        qk_sa += self.policy.env.rsa((i,j),action)
                        # 获得后继状态列表
                        successors = self.policy.env.get_successors((i,j),action)
                        for next_state, prob in successors:
                            qk_sa += discounted_rate * prob * self.state_values[next_state[0],next_state[1]]
                        self.action_values[i,j,action] = qk_sa
                    # 找到最大的动作价值函数及其对应的动作，用argmax
                    best_action = np.argmax(self.action_values[i,j])
                    # 更新策略，用最大的动作价值函数为1，其他为0
                    # 先清零
                    self.policy.probs[i,j] = np.zeros(self.policy.env.action_space.n)
                    self.policy.probs[i,j][best_action] = 1
                    # 更新状态价值函数
                    self.state_values[i,j] = self.action_values[i,j,best_action]
            # 判断是否收敛,用数组的平方和
            delta = np.sum(np.square(self.state_values - previous_state_values))
            if delta < threshold:
                break
            training_step += 1
            print(f"第{training_step}次训练")
        print(f"训练完成，共进行了{training_step}次训练")

    def test(self):
        self.policy.play()
        self.policy.showpolicy()
        print(self.state_values)
                        
                        


print("test")
"""
size = 5
target = (4,4)
barriers = [(1,1),(2,2),(3,3)]

en = MyGridEnv(size, target, barriers)
tb = TabularPolicy(en)
vi = ValueIteration(tb)
tb.play()
tb.showpolicy()
print(vi.state_values)
"""
#diag_grid = MyGridEnv(5, (4, 4), [(1, 1), (2, 2), (3, 3)]) #对角线上有障碍物
#tb = TabularPolicy(diag_grid)
#vi = ValueIteration(tb)
#vi.train()
#vi.test()

class PolicyIteration(Solve):
    def __init__(self, policy):
        super().__init__(policy)
        # 初始化价值函数和动作函数

    def train(self):
        threshold = 1e-3
        discounted_rate = 0.9
        training_step = 0
        while True:
            previous_state_values = np.copy(self.state_values)
            # 策略评估
            # initializaion
            v_pik0 = np.zeros((self.policy.env.size,self.policy.env.size))
            while True:
                v_pik_jminus1 = np.copy(v_pik0)
                # for every state
                for x in range(self.policy.env.size):
                    for y in range(self.policy.env.size):
                        # calculate v_pik_j
                        # 获取s状态下采取策略pik的动作概率
                        actions_probs = self.policy.get_probs((x,y))
                        v_pik_j_xy=0
                        # 动作的概率乘以动作价值函数
                        for action in range(self.policy.env.action_space.n):
                        # 计算动作价值函数
                            qk_sa = 0
                            qk_sa += self.policy.env.rsa((x,y),action)
                            # 获得后继状态列表
                            successors = self.policy.env.get_successors((x,y),action)
                            for next_state, prob in successors:
                                qk_sa += discounted_rate * prob * v_pik0[next_state[0],next_state[1]]
                            v_pik_j_xy += actions_probs[action] * qk_sa
                        v_pik0[x,y] = v_pik_j_xy 
                # 判断是否收敛
                delta = np.sum(np.square(v_pik0 - v_pik_jminus1))
                if delta < threshold:
                    break
            self.state_values = v_pik0
            # 策略改进
            #计算动作价值函数，然后选择最大的动作
            for i in range(self.policy.env.size):
                for j in range(self.policy.env.size):
                    #对于每个动作：
                    for action in range(self.policy.env.action_space.n):
                        # 计算动作价值函数
                        qk_sa = 0
                        qk_sa += self.policy.env.rsa((i,j),action)
                        # 获得后继状态列表
                        successors = self.policy.env.get_successors((i,j),action)
                        for next_state, prob in successors:
                            qk_sa += discounted_rate * prob * self.state_values[next_state[0],next_state[1]]
                        self.action_values[i,j,action] = qk_sa
                    # 找到最大的动作价值函数及其对应的动作，用argmax
                    best_action = np.argmax(self.action_values[i,j])
                    # 更新策略，用最大的动作价值函数为1，其他为0
                    # 先清零
                    self.policy.probs[i,j] = np.zeros(self.policy.env.action_space.n)
                    self.policy.probs[i,j][best_action] = 1
            # 判断是否收敛,用数组的平方和
            delta = np.sum(np.square(self.state_values - previous_state_values))
            if delta < threshold:
                break
            training_step += 1
            print(f"第{training_step}次训练")
        print(f"训练完成，共进行了{training_step}次训练")

    def test(self):
        self.policy.play()
        self.policy.showpolicy()
        print(self.state_values)


class McBasic(Solve):
    def __init__(self, policy):
        super().__init__(policy)
        self.length_of_episodes = 20 #每次训练的episode长度
        self.num_episodes = 30 #每个位置训练的次数


    def train(self):
        training_step = 0
        while True:
            previous_action_values = np.copy(self.action_values)
            for x in range(self.policy.env.size):
                for y in range(self.policy.env.size):
                    for action in range(self.policy.env.action_space.n):
                        rewards = np.zeros(self.num_episodes)
                        for i in range(self.num_episodes):
                            rewards[i] = self.one_episode((x,y),action)
                            #求平均
                        avg_reward = np.mean(rewards)
                        self.action_values[x,y,action] = avg_reward
            # 改进策略
            for i in range(self.policy.env.size):
                for j in range(self.policy.env.size):
                    best_action = np.argmax(self.action_values[i,j])
                    self.policy.probs[i,j] = np.zeros(self.policy.env.action_space.n)
                    self.policy.probs[i,j][best_action] = 1
            # 判断是否收敛
            delta = np.sum(np.square(self.action_values - previous_action_values))
            if delta < self.threshold:
                break
            training_step += 1
            print(f"第{training_step}次训练")
            self.policy.showpolicy()
        print(f"训练完成，共进行了{training_step}次训练")

    def one_episode(self,state,action):
        """
        从(s,a)开始，执行一个episode，返回总的discounted reward
        """
        self.policy.env.set(state)
        total_reward = 0
        dis = self.discounted_rate
        next_state, reward, done, _ = self.policy.env.step(action) #先走出第一步，后面按照策略走
        total_reward += reward
        for _ in range(self.length_of_episodes):
            action = self.policy.sample_action(next_state)
            next_state, reward, done, _ = self.policy.env.step(action)
            total_reward += dis * reward
            dis *= self.discounted_rate
        return total_reward
    
    def test(self):
        self.policy.env.reset()
        self.policy.play()
        self.policy.showpolicy()
        

class McExploring(Solve):
    def __init__(self, policy):
        super().__init__(policy)
        self.num_episodes = 2 #训练次数
        self.length_of_episodes = 100 #每次训练的episode长度
        self.epsilon = 0.5 #epsilon贪心策略

    def one_episode(self,start_state):
        """
        一直走，走到指定长度，返回所有状态动作奖励序列
        """
        self.policy.env.reset()
        state = start_state
        self.policy.env.set(state)
        state_action_rewards = np.zeros((self.length_of_episodes,4),dtype=int)
        for i in range(self.length_of_episodes):
            action = self.policy.sample_action(state)
            next_state, reward, done, _ = self.policy.env.step(action)
            state_action_rewards[i][0] = state[0]
            state_action_rewards[i][1] = state[1]
            state_action_rewards[i][2] = action
            state_action_rewards[i][3] = reward
            state = next_state
        return state_action_rewards
    def train(self):
        # 存储episode中的动作序列，状态序列和奖励序列，走一步存一次
        for _ in range(self.num_episodes):
            self.action_values = np.zeros((self.policy.env.size,self.policy.env.size,self.policy.env.action_space.n))
            #清空动作价值函数，每次训练都要重新计算
            for _ in range(5):
                #生成一个episode
                Num = np.zeros((self.policy.env.size,self.policy.env.size,self.policy.env.action_space.n),dtype=int)
                returns = np.zeros((self.policy.env.size,self.policy.env.size,self.policy.env.action_space.n))
                state_action_rewards = self.one_episode(self.policy.env.start)
                g = 0
                for t in range(self.length_of_episodes-1,-1,-1):
                    statex,statey,action,reward = state_action_rewards[t]
                    state = (int(statex),int(statey))
                    g = self.discounted_rate * g + reward
                    #对应的状态动作对出现的次数加1
                    Num[state[0],state[1],action] += 1
                    #加到returns中
                    returns[state[0],state[1],action] += g
            # 每一个episode结束，更新策略
            #计算动作价值函数，然后选择最大的动作
            for i in range(self.policy.env.size):
                for j in range(self.policy.env.size):
                    for action in range(self.policy.env.action_space.n):
                        if Num[i,j,action] != 0:
                            self.action_values[i,j,action] = returns[i,j,action]/Num[i,j,action]
            # 改进策略，epsilon贪心策略
            for i in range(self.policy.env.size):
                for j in range(self.policy.env.size):
                    best_action = np.argmax(self.action_values[i,j])
                    self.policy.probs[i,j] = np.zeros(self.policy.env.action_space.n)
                    self.policy.probs[i,j][best_action] = 1-self.epsilon
                    self.policy.probs[i,j] += self.epsilon/self.policy.env.action_space.n
            #print(f"第{x}次训练完成")
    
    def test(self):
        self.policy.env.reset()
        self.policy.showpolicy()
            
                
diag_grid = MyGridEnv(5, (3,2), [(1,1),(1,2),(2,2),(3,3),(3,1),(4,1),(3,3)]) #对角线上有障碍物
tb = TabularPolicy(diag_grid)
pi = McBasic(tb)
pi.train()
pi.test()
