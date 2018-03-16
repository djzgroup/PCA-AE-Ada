import random
import numpy as np
import scipy.io

List = list(range(1, 276))
random_num = random.sample(List, 220)  # 从list中随机获取35个元素，作为一个片断返回
# random_num.sort()
print(random_num)
print(List)

mat1 = scipy.io.loadmat('/home/djz/share_djz/data/PCA_data/GSE2034_PCA.mat')
x_train = mat1['ma2']
y_train = mat1['co']

train_size = x_train.shape[0]
y_train = np.reshape(y_train, train_size)
print('x_train:', x_train.shape)

#random data
train_data = []
train_lable = []
num = len(random_num)
for i in range(num):
    train_data.append(x_train[random_num[i]])
    train_lable.append(y_train[random_num[i]])

train_data = np.array(train_data)
train_lable = np.array(train_lable)
print('train_data:', train_data.shape)

test_data = []
test_lable = []
for i in range(len(List)):
    if i not in random_num:
        # print(i)
        test_data.append(x_train[i])
        test_lable.append(y_train[i])

test_data = np.array(test_data)
test_lable = np.array(test_lable)
print('test_data:', test_data.shape)
