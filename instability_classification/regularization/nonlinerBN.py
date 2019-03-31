
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy.random import *
X = uniform(0,1000,(1000,2))


# In[2]:


X.shape


# In[3]:


X


# In[4]:


X[:,1] = X[:,1]/100


# In[5]:


X


# In[6]:


df = pd.DataFrame({"x1": X[0:1000,0],"x2": X[0:1000,1]})


# In[7]:


df.tail()


# In[8]:


def calc_log(num):
    return math.log(num)


# In[9]:


df["a"] = df['x1'].apply(calc_log)


# In[10]:


df


# In[11]:


t1= np.where(df["a"] >= df["x2"],0,1)
t2= np.where(df["a"] >= df["x2"],1,0)


# In[12]:


df["off"] = t1
df["on"] = t2


# In[13]:


df.head()


# In[14]:


df1 = df[df.on==1]


# In[15]:


df2 = df[df.on==0]


# In[16]:


df2.head()


# In[17]:


plt.scatter(df1["x1"], df1["x2"],
            color='red', marker='x', label='off', s= 12)
plt.scatter(df2["x1"], df2["x2"],
            color='blue', marker='o', label='on', s = 12)
plt.legend(loc='upper left')
plt.savefig('non-liner dataset')
plt.show()


# In[18]:


t = np.array([t1,t2])


# In[19]:


t.T


# In[20]:


t[:700].shape


# In[21]:


t_train = t.T[:700,:]
print(t_train)


# In[22]:


t_test = t.T[701:,:]
print(t_test)


# In[23]:


t_train.shape


# In[24]:


t_test.shape


# In[25]:


(x_train, x_test) = (X[:700],X[701:])


# In[26]:


x_train.shape


# In[27]:


x_test.shape


# In[28]:


train_size = x_train.shape[0]


# In[29]:


train_size


# In[40]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[41]:


train_acc_list10, bn_train_acc_list10 = __train(0.1)


# In[42]:


train_acc_list11, bn_train_acc_list11 = __train(0.01)


# In[43]:


train_acc_list13, bn_train_acc_list13 = __train('He')


# In[44]:


train_acc_list12, bn_train_acc_list12 = __train('Xavier')


# In[45]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list10))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list10, label='0.1', markevery=2)
plt.plot(x, bn_train_acc_list11, label='0.01', markevery=2)
plt.plot(x, bn_train_acc_list12, label='Xavier', markevery=2)
plt.plot(x, bn_train_acc_list13, label='He', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Weight initial')


# ## ニューロン数を変更していく
# * 100,1000,10000

# In[49]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000, 1000, 1000], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000, 1000, 1000], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[50]:


bn_train_acc_list00 = __train('He')


# In[51]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[500, 500, 500], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[500, 500, 500], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[52]:


bn_train_acc_list01 = __train('He')


# In[53]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

max_epochs = 100
train_size = x_train.shape[0]
batch_size = 1000
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[10, 10, 10], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[10, 10, 10], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[54]:


bn_train_acc_list02 = __train('He')


# In[57]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
a = np.arange(len(bn_train_acc_list00[0]))
b = np.arange(len(bn_train_acc_list13))
plt.subplot(1,2,1)
plt.plot(a, bn_train_acc_list00[1], label='1000', markevery=2)
plt.plot(a, bn_train_acc_list01[1], label='500', markevery=2)
plt.plot(b, bn_train_acc_list13, label='100', markevery=2)
plt.plot(a, bn_train_acc_list02[1], label='10', markevery=2)
plt.xlabel('epochs')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Newron Size')


# ## 層の数

# In[58]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[59]:


train_acc_list30, bn_train_acc_list30 = __train('He')


# In[60]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100,100,100,100,100,100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100,100,100,100,100,100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[61]:


train_acc_list31, bn_train_acc_list31 = __train('He')


# In[62]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list30))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list30, label='1 layer', markevery=2)
plt.plot(x, bn_train_acc_list13, label='3 layer', markevery=2)
plt.plot(x, bn_train_acc_list31, label='6 layer', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Layer Num')


# ## バッチ数

# In[69]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 10
learning_rate = 0.001
iter_nums = 10000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)


            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

    return train_acc_list, bn_train_acc_list


# In[70]:


train_acc_list10, bn_train_acc_list10 = __train(0.1)


# In[71]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[72]:


train_acc_list11, bn_train_acc_list11 = __train(0.1)


# In[73]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 700
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[100, 100, 100], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# In[74]:


train_acc_list12, bn_train_acc_list12 = __train(0.1)


# In[75]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list10))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list10, label='10', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Batch Num')


# In[77]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list11))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list11, label='100', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Batch Num')


# In[78]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list13))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list13, label='700', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Batch Num')


# In[82]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 700
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000,1000,1000,1000,1000,1000], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000,1000,1000,1000,1000,1000], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []
    bn_test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            bn_test_acc = bn_network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            bn_test_acc_list.append(bn_test_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(bn_train_acc) + " - " + str(bn_test_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return bn_train_acc_list, bn_test_acc_list


# In[83]:


bn_train_acc_list, bn_test_acc_list = __train('He')


# In[84]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list50))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list, label='train', markevery=2)
plt.plot(x, bn_test_acc_list, label='test', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Log\n6 layer,1000 Newron')


# In[85]:


max_epochs = 100
train_size = x_train.shape[0]
batch_size = 700
learning_rate = 0.001
iter_nums = 30000

def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000,1000,1000], output_size=2,
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=2, hidden_size_list=[1000,1000,1000], output_size=2,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []
    bn_test_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(iter_nums):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            bn_test_acc = bn_network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
            bn_test_acc_list.append(bn_test_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(bn_train_acc) + " - " + str(bn_test_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return bn_train_acc_list, bn_test_acc_list


# In[86]:


bn_train_acc_list10, bn_test_acc_list10 = __train('He')


# In[87]:


# グラフの描画
plt.figure(figsize=(14,4)) # figureの縦横の大きさ
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(bn_train_acc_list50))
plt.subplot(1,2,1)
plt.plot(x, bn_train_acc_list, label='train', markevery=2)
plt.plot(x, bn_test_acc_list, label='test', markevery=2)
plt.xlabel('epoch')
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title('Log\n3 layer,1000 Newron')
