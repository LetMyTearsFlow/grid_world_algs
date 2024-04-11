import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)


def relu(x):
    return np.maximum(x, 0)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def mean_squared_error(y, t):
    # 均方差
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))  # 交叉熵误差


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h) 的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


class PolicyNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  # 常数项
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x, isln=False):
        """
        前向传播
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)  # 激活函数
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        if isln:
            return np.log(y)
        return y

    def get_params(self):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        return np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))

    def set_params(self, parameters):
        W1_shape = self.params['W1'].shape
        b1_shape = self.params['b1'].shape
        W2_shape = self.params['W2'].shape
        b2_shape = self.params['b2'].shape

        W1_size = np.prod(W1_shape)
        b1_size = np.prod(b1_shape)
        W2_size = np.prod(W2_shape)
        b2_size = np.prod(b2_shape)

        self.params['W1'] = parameters[:W1_size].reshape(W1_shape)
        self.params['b1'] = parameters[W1_size:W1_size + b1_size].reshape(b1_shape)
        self.params['W2'] = parameters[W1_size + b1_size:W1_size + b1_size + W2_size].reshape(W2_shape)
        self.params['b2'] = parameters[W1_size + b1_size + W2_size:].reshape(b2_shape)

    def gradient(self, x_value, action, isln=False):
        """
        求神经网络ln之后关于所有隐藏层参数的梯度/神经网络的梯度
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        delta = 1e-5

        params = self.get_params()
        grads = np.zeros_like(params)

        for idx in range(params.size):
            tmp_val = params[idx]
            params[idx] = tmp_val + delta
            self.set_params(params)
            fxh1 = self.predict(x_value, isln=isln)[action]
            params[idx] = tmp_val - delta
            self.set_params(params)
            fxh2 = self.predict(x_value, isln=isln)[action]
            grads[idx] = (fxh1 - fxh2) / (2 * delta)
            params[idx] = tmp_val

        return grads


pn = PolicyNet(2, 100, 5)
print(pn.gradient((3, 4), 2))


class ValueNet:
    def __init__(self, input_size=3, hidden_size=100, output_size=1, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  # 常数项
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x, isln=False):
        """
        前向传播
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)  # 激活函数
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        if isln:
            return np.log(y)
        return y

    def get_params(self):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        return np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))

    def set_params(self, parameters):
        W1_shape = self.params['W1'].shape
        b1_shape = self.params['b1'].shape
        W2_shape = self.params['W2'].shape
        b2_shape = self.params['b2'].shape

        W1_size = np.prod(W1_shape)
        b1_size = np.prod(b1_shape)
        W2_size = np.prod(W2_shape)
        b2_size = np.prod(b2_shape)

        self.params['W1'] = parameters[:W1_size].reshape(W1_shape)
        self.params['b1'] = parameters[W1_size:W1_size + b1_size].reshape(b1_shape)
        self.params['W2'] = parameters[W1_size + b1_size:W1_size + b1_size + W2_size].reshape(W2_shape)
        self.params['b2'] = parameters[W1_size + b1_size + W2_size:].reshape(b2_shape)

    def gradient(self, x_value, isln=False):
        """
        求神经网络ln之后关于所有隐藏层参数的梯度/神经网络的梯度
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        delta = 1e-5

        params = self.get_params()
        grads = np.zeros_like(params)

        for idx in range(params.size):
            tmp_val = params[idx]
            params[idx] = tmp_val + delta
            self.set_params(params)
            fxh1 = self.predict(x_value, isln=isln)
            params[idx] = tmp_val - delta
            self.set_params(params)
            fxh2 = self.predict(x_value, isln=isln)
            grads[idx] = (fxh1 - fxh2) / (2 * delta)
            params[idx] = tmp_val

        return grads
