from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 30     # 树的数量
ratio_data = 0.6   # 采样的数据比例
ratio_feat = 0.6 # 采样的特征比例
hyperparams = {
    "depth":10, 
    "purity_bound":0.1,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    forest = []
    for i in range(num_tree):
        rand_index = np.random.choice(X.shape[0], int(ratio_data * X.shape[0]), replace = False)
        rand_X = np.array([X[i] for i in rand_index])
        rand_Y = np.array([Y[i] for i in rand_index])
        rand_feat = np.random.choice(X.shape[1], int(ratio_feat * X.shape[1]), replace = True)
        forest.append(buildTree(rand_X, rand_Y, rand_feat.tolist(), hyperparams["depth"], hyperparams["purity_bound"], hyperparams["gainfunc"]))
    return forest
    # raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
