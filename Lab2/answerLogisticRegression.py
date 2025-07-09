import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 2  # 学习率
wd = 0.1 # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    ans = X @ weight + bias
    return ans
    raise NotImplementedError

def sigmoid(x):
    '''
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    '''
    '''
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] >= 0:
            result[i] = 1 / (1 + np.exp(-x[i]))
        else:
            result[i] = np.exp(x[i]) / (1 + np.exp(x[i]))
    return result
    '''
    mask = x >= 0
    result = np.zeros_like(x)
    result[mask] = 1 / (1 + np.exp(-x[mask]))
    result[~mask] = np.exp(x[~mask]) / (1 + np.exp(x[~mask]))
    return result

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    haty = predict(X, weight, bias)
    n = X.shape[0]
    d = X.shape[1]
    loss = np.zeros(1)
    grad_w = np.zeros(d)
    grad_b = np.zeros(1)
    EPS = 1e-6
    prob = sigmoid(Y * haty)
    loss = np.mean(np.log(prob + EPS) + wd * np.sum(weight ** 2))
    grad_w = -(X.T @ ((1 - prob) * Y)) / n + 2 * wd * weight
    grad_b = -np.mean((1 - prob) * Y)
    new_w = weight - lr * grad_w
    new_b = bias - lr * grad_b
    return (haty, loss, new_w, new_b)
    # raise NotImplementedErrorpython
