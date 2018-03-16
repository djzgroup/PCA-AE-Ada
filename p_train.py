#-----------------------------
# PCA_data + Adaboost

import scipy.io
import PEensemble.Evaluate_Function as Evaluate_Function

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# this is the size of our encoded representations
np.random.seed(1337)  # for reproducibility

encoding_dim = 2 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


mat1 = scipy.io.loadmat('/home/djz/code/dataset/cancer/PCA_data/GSE2034_PCA.mat')
x_train = mat1['PCA_MaxDim']
y_train= mat1['co']

mat2=scipy.io.loadmat('/home/djz/code/dataset/cancer/PCA_data/GSE7390_PCA.mat')
x_test = mat2['PCA_MaxDim']
y_test= mat2['co']

# padding
pad_num = x_train.shape[1] - x_test.shape[1]
x_test = np.pad(x_test, ((0,0), (0, pad_num)), 'constant')

y_train = np.reshape(y_train, x_train.shape[0])
y_test = np.reshape(y_test, x_test.shape[0])


print('x_train:',x_train.shape)
print('y_train:',y_train.shape)
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)


# ------------------adaboost------------------
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
bdt.fit(x_train, y_train)
pred = bdt.predict(x_test[:])
print(pred)

Evaluate_Function.Evaluate_Fun(pred, y_test, x_test)


