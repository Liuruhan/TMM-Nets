import os
import numpy as np
from sklearn import metrics

root = './FP_22/'
models = os.listdir(root)

for model_name in range(len(models)):
    for iter in range(1, 51):
        result = np.loadtxt(root+models[model_name]+'/'+str(iter)+'val_pred.csv', delimiter=',')
        # print(result.shape)

        acc = metrics.accuracy_score(result[:, 0], result[:, 1])
        pre = metrics.precision_score(result[:, 0], result[:, 1])
        recall = metrics.recall_score(result[:, 0], result[:, 1])
        f1 = metrics.f1_score(result[:, 0], result[:, 1])
        print(models[model_name], 'epoch_'+str(iter), 'acc:', acc, 'pre:', pre, 'recall:', recall, 'f1:', f1)
