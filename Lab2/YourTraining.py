import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *
from scipy.ndimage import rotate, shift

choice_num = 2000
epoch_num = 20
lr = 0.001
wd1 = 1e-5
wd2 = 1e-5
batchsize = 128
rotate_list = [-10, -7, -4, -1, 1, 4, 7, 10]
shift_list = [-4, -2, 2, 4]
setseed(0)
save_path = "model/mymodel.npy"

def buildGraph(Y):
    nodes = [BatchNorm(784), 
             Linear(mnist.num_feat, 512), relu(), Dropout(0.2), 
             BatchNorm(512),
             Linear(512, 256), relu(), Dropout(0.2),
             BatchNorm(256), 
             Linear(256, mnist.num_class), Softmax(), CrossEntropyLoss(Y)]
    graph = Graph(nodes)
    return graph

trn_X = mnist.trn_X.reshape(-1, 28, 28)
trn_Y = mnist.trn_Y
val_X = mnist.val_X.reshape(-1, 28, 28)
val_Y = mnist.val_Y
choices = np.random.choice(val_X.shape[0], choice_num, replace=False)
add_X = val_X[choices]
add_Y = val_Y[choices]
X = np.concatenate((trn_X, add_X), axis=0)
Y = np.concatenate((trn_Y, add_Y), axis=0)
aug_X = []
aug_Y = []
for x, y in zip(X, Y):
    for angle in rotate_list:
        aug_X.append(rotate(x, angle, reshape=False))
        aug_Y.append(y)
    for delta_x in shift_list:
        for delta_y in shift_list:
            aug_X.append(shift(x, (delta_x, delta_y)))
            aug_Y.append(y)
X = np.concatenate((X, aug_X), axis = 0)
Y = np.concatenate((Y, aug_Y), axis = 0)
X = X.reshape(-1, 784)

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, epoch_num+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)