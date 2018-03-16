#-----------------------------
# Encode + Raw_data + Adaboost

from keras.layers import Input, Dense
from keras import regularizers
import numpy as np
import scipy.io
from keras.models import Sequential
from keras import backend as K
import PEensemble.Evaluate_Function as Evaluate_Function
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# this is the size of our encoded representations
np.random.seed(1337)  # for reproducibility

encoding_dim = 2 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

mat1 = scipy.io.loadmat('/home/djz/code/dataset/cancer/PCA_data/GSE2034_PCA.mat')
x_train = mat1['PCA_MaxDim']
y_train = mat1['co']

mat2 = scipy.io.loadmat('/home/djz/code/dataset/cancer/Raw_data/GSE2034_ma2.mat')
x_train_raw = mat2['ma2']

test_mat1= scipy.io.loadmat('/home/djz/code/dataset/cancer/Raw_data/GSE7390_ma2.mat')
test_mat_raw = test_mat1['ma2']
test_lable = test_mat1['co']
test_mat2 = scipy.io.loadmat('/home/djz/code/dataset/cancer/PCA_data/GSE7390_PCA.mat')
test_mat_PCA = test_mat2['PCA_MaxDim']

merge_data = np.concatenate([x_train, x_train_raw], axis=1)


y_train = np.reshape(y_train, x_train.shape[0])
print('merge_data:',merge_data.shape)
print('y_train:',y_train.shape)

dim = merge_data.shape[1]

input_img = Input(shape=(dim,))#


model = Sequential()
model.add(Dense(64, activation='elu', input_dim = dim, activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(32, activation='elu', activity_regularizer=regularizers.l1(10e-5)))

model.add(Dense(64, activation='elu', activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(dim, activation='elu', activity_regularizer=regularizers.l1(10e-5)))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(merge_data, merge_data, nb_epoch=100, batch_size=64, shuffle=True, verbose=1, validation_split= 0.2)
model.summary()
model.save('my_model.h5')

#---------------------------------------
#  get intermediate layer encode
encode_train_output = K.function([model.layers[0].input], [model.layers[2].output])
train_output = encode_train_output([merge_data, 0])[0]
print('encode_output:',train_output.shape)
print(train_output)

#---------------------------------------------
# -----------------adaboost-------------------
encode_PCA_data = np.concatenate([train_output, x_train], axis=1)
test_merge = np.concatenate([test_mat_raw, test_mat_PCA], axis=1)
#padding
pad_num = test_merge.shape[1] - encode_PCA_data.shape[1]
encode_PCA_data = np.pad(encode_PCA_data, ((0,0),(0,pad_num)), 'constant')

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                   algorithm="SAMME",
                   n_estimators=200, learning_rate=0.8)
bdt.fit(encode_PCA_data, y_train)
pred = bdt.predict(test_merge)
print(pred)
Evaluate_Function.Evaluate_Fun(pred, test_lable, test_mat_PCA)


