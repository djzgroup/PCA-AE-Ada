#----------------------------------------

from keras.layers import Input, Dense
from keras import regularizers
import numpy as np
import scipy.io
from keras.models import Sequential
import random
from keras import backend as K
import PEensemble.Evaluate_Function as Evaluate_Function
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# this is the size of our encoded representations
np.random.seed(1337)  # for reproducibility

encoding_dim = 2 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

mat1=scipy.io.loadmat('/home/djz/code/dataset/cancer/PCA_data/GSE2034_PCA.mat')
x_train=mat1['ma2']
y_train=mat1['co']

train_size = x_train.shape[0]
y_train = np.reshape(y_train, train_size)

#---------------------------------
# random fetch data
List = list(range(1, 276))
random_num = random.sample(List, 275)
train_list = random_num[:220]
test_list = random_num[220:]

train_data = []
train_lable = []
num = len(train_list)
for i in range(num):
    #print(random_num[i])
    train_data.append(x_train[train_list[i]])
    train_lable.append(y_train[train_list[i]])

train_data = np.array(train_data)
train_lable = np.array(train_lable)
print('train_data:',train_data.shape)

dim = train_data.shape[1]
input_img = Input(shape=(dim,))#

model = Sequential()
model.add(Dense(64, activation='elu', input_dim = dim, activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(32, activation='elu', activity_regularizer=regularizers.l1(10e-5)))

model.add(Dense(64, activation='elu', activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(dim, activation='elu', activity_regularizer=regularizers.l1(10e-5)))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(x_train, x_train, nb_epoch=200, batch_size=64, shuffle=True, verbose=1, validation_split= 0.2)
model.summary()
model.save('my_model.h5')


#------------------------------------
# ----------train adaboost-----------
encode_train_output = K.function([model.layers[0].input], [model.layers[2].output])
train_output = encode_train_output([train_data, 0])[0]
print('encode_output:',train_output.shape)
print(train_output)

merge_data = np.concatenate([train_output, train_data], axis=1)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                   algorithm="SAMME",
                   n_estimators=200, learning_rate=0.8)
bdt.fit(merge_data, train_lable)


#---------------------------------
#-------------test----------------
test_data = []
test_lable = []
for i in range(len(test_list)):
    if i not in random_num:
        test_data.append(x_train[test_list[i]])
        test_lable.append(y_train[test_list[i]])

test_data = np.array(test_data)
test_lable = np.array(test_lable)
print('train_data:',train_data.shape)

encode_train_output = K.function([model.layers[0].input], [model.layers[2].output])
test_output = encode_train_output([test_data, 0])[0]
print('encode_output:',test_output.shape)
print(test_output)

merge_data = np.concatenate([test_output, test_data], axis=1)
pred = bdt.predict(merge_data)
print(pred)
Evaluate_Function.Evaluate_Fun(pred, test_lable, merge_data)
