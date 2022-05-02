import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


df_train = pd.read_csv('../dataset/test.csv')
train_features = df_train.iloc[:, :-1].values
train_labels = df_train.iloc[:, -1].values


df = pd.read_csv('../dataset/mammography_label.csv')
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=123)
# rbf核函数，设置数据权重
svc = SVC(kernel='rbf', class_weight='balanced', )
c_range = np.logspace(-5, 15, 11, base=2)
gamma_range = np.logspace(-9, 3, 13, base=2)
# 网格搜索交叉验证的参数范围，cv=3,3折交叉，n_jobs=-1，多核计算
param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
grid = RandomizedSearchCV(svc, param_grid, cv=3, n_jobs=-1)
# 训练模型
# clf = grid.fit(x_train, y_train)
print('=============================================================')
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print('total:', x_train.shape[0] + x_test.shape[0], y_train.shape[0] + y_test.shape[0])
# print(train_features.shape, train_labels.shape)
# clf = grid.fit(train_features, train_labels)
clf = grid.fit(x_train, y_train)

# pred = clf.predict(features)
pred = clf.predict(x_test)
# print('准确率：', accuracy_score(labels, pred))
# print('召回率：', recall_score(labels, pred))
# print('精确率：', precision_score(labels, pred))
# print('f1_score：', 2 * (precision_score(labels, pred) * recall_score(labels, pred)) / (precision_score(labels, pred) + recall_score(labels, pred)))
# print('混淆矩阵：\n', confusion_matrix(labels, pred))
# print('分类报告：', classification_report(labels, pred))
# print('roc_auc_score：', roc_auc_score(labels, pred))

print('准确率：', accuracy_score(y_test, pred))
print('召回率：', recall_score(y_test, pred))
print('精确率：', precision_score(y_test, pred))
print('f1_score：', 2 * (precision_score(y_test, pred) * recall_score(y_test, pred)) / (precision_score(y_test, pred) + recall_score(y_test, pred)))
print('混淆矩阵：\n', confusion_matrix(y_test, pred))
cls_report = classification_report(y_test, pred)
print('分类报告:\n', cls_report)
print('roc_auc_score：', roc_auc_score(y_test, pred))


df_compare = pd.DataFrame({'labels': y_test, 'pred': pred})
df_compare.to_csv('pure_svm.csv', index=False)
cls_report_df = pd.DataFrame(cls_report.split('\n')[:-2])
cls_report_df.to_csv('pure_svm_cls_report.csv', index=False)

# print('精度为%s' % score)
# print('精度为%s' % precision)
# print('召回率为%s' % recall)
# print('f1score为%s' % f1score)


# 输出模型最优参数
print('最优参数：', grid.best_params_)
# 输出模型的训练评估结果
# means = grid.cv_results_['mean_test_score']
# stds = grid.cv_results_['std_test_score']
# params = grid.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))




