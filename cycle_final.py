import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv
import sys

# Sigmoid函数
def sigmoid(x):
    s = 1 / (1 + np / exp(-x))
    return s

# 均方误差
def MSE(y, Y):
    return np.mean((y - Y)**2)

# NN
# 一层隐藏层
class NeuralNetwork(object):
    def __init__( self, input_nodes, hidden_nodes, output_nodes, learning_rate ):
        # 各层节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 学习率
        self.lr = learning_rate

        # 权重初始化
        self.weights_input_to_hidden = np.random.normal( 0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes) )
        self.b_input_to_hidden = np.zeros(self.hidden_nodes)
        self.weights_hidden_to_output = np.random.normal( 0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes) )
        self.b_hidden_to_output = np.zeros(self.output_nodes)

        # 激活函数Sigmoid
        self.activation_function = lambda x : sigmoid(x)

    # 训练网络
    def train(self, features, targets):
        # 初始化
        n_record = features.shape[0]    # 特征数
        delta_weights_i_h = np.zeros( self.weights_input_to_hidden.shape )  # 输入至隐层
        delta_b_i_h = np.zeros(self.b_input_to_hidden.shape)
        delta_weights_h_o = np.zeros( self.weights_hidden_to_output.shape ) # 隐层至输出
        delta_b_h_o = np.zeros(self.b_hidden_to_output.shape)


        for X, y in zip( features, targets ):
            # 正向传播
            # 输入层
            hidden_inputs = np.dot( X, self.weights_input_to_hidden ) + self.b_input_to_hidden
            hidden_outputs = self.activation_function( hidden_inputs )

            # 输出层
            final_inputs = np.dot( hidden_outputs, self.weights_hidden_to_output ) + self.b_hidden_to_output
            final_outputs = final_inputs

            # BP
            error = y - final_outputs
            output_error_term = error # error * 1

            hidden_error = np.dot( self.weights_hidden_to_output, output_error_term )
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs) # f'(hidden_input)

            # delta_weight
            delta_weights_i_h += hidden_error_term * X[:,None]
            delta_b_i_h += hidden_error_term

            delta_weights_h_o += output_error_term * hidden_outputs[:,None]
            delta_b_h_o += output_error_term

        # 权重更新
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_record
        self.b_input_to_hidden += self.lr * delta_b_i_h/n_record
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_record
        self.b_hidden_to_output = self.lr * delta_b_h_o/n_record

    # 运行网络
    def run(self, features):
        hidden_inputs = np.dot( features, self.weights_input_to_hidden ) + self.b_input_to_hidden
        hidden_output = self.activation_function( hidden_inputs )

        final_inputs = np.dot( hidden_output, self.weights_hidden_to_output ) + self.b_hidden_to_output
        final_outputs = final_inputs

        return final_outputs

print('start')
# data
data_path = './data/train.csv'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 去除离群数据
outliers_removed = train[np.abs(train['count'] - train['count'].mean()) <= (3 * train['count'].std())]
train = outliers_removed.reset_index(drop = True)

# 日期分为年月时备用
train['hour'] = train.datetime.apply(lambda x: x.split()[1].split(':')[0])
train['month'] = train.datetime.apply(lambda x: x.split('/')[1])
train['year'] = train.datetime.apply(lambda x: x.split('/')[0])
test['hour'] = test.datetime.apply(lambda x: x.split()[1].split(':')[0])
test['month'] = test.datetime.apply(lambda x: x.split('/')[1])
test['year'] = test.datetime.apply(lambda x: x.split('/')[0])

# 处理哑变量
dummies_fields = ['season', 'weather', 'hour', 'month', 'year']
for each in dummies_fields:
    dummies = pd.get_dummies(train.loc[:, each], prefix=each)
    train = pd.concat([train, dummies], axis=1)
    dummies = pd.get_dummies(test.loc[:, each], prefix=each)
    test = pd.concat([test, dummies], axis=1)

# 删去处理前数据
# 温度与体感温度基本成线性相关，故删去体感温度atemp
drop_fields = ['season', 'weather', 'atemp', 'datetime', 'hour', 'month', 'year']
train_data = train.drop(drop_fields, axis=1)
test_data = test.drop(drop_fields, axis=1)

# normalization
# 提取train set与test set的温度湿度风速部分得到均值与标准差
target_fields = ['temp', 'humidity', 'windspeed']
train_features = train_data.loc[:, target_fields]
test_features = test_data.loc[:, target_fields]
total_data = pd.concat( [train_features, test_features], axis=0 )
total_data = total_data.reset_index(drop = True)
train_features = {}
test_features = {}
# 对这三类特征进行归一化处理
standard_field = ['temp', 'humidity', 'windspeed']
scaled_feature = {}
for each in standard_field:
    mean = total_data[each].mean()
    std = total_data[each].std()
    scaled_feature[each] = [mean, std]
    train_data.loc[:, each] = (train_data.loc[:, each] - mean) / std
    test_data.loc[:, each] = (test_data.loc[:, each] - mean) / std

# 对train_set中的结果值进行处理
# 为确保不为负，在归一化前先取对数
standard_field = ['casual', 'registered', 'count']
for each in standard_field:
    train_data.loc[:, each] = train_data.loc[:, each] + np.ones(train_data.loc[:, each].shape)
    train_data.loc[:, each] = np.log(train_data.loc[:, each] )
    train_mean = train_data[each].mean()
    train_std = train_data[each].std()
    scaled_feature[each] = [train_mean, train_std]
    train_data.loc[:, each] = (train_data.loc[:, each] - train_mean) / train_std

# 分割特征与count，取最后一部分数据作为验证集
target_fields = ['casual', 'registered', 'count']
features = train_data.drop(target_fields, axis=1)
targets = train_data.loc[:, target_fields]
val_features = features[-15*24:]
val_targets = targets[-15*24:]
train_features = features[:-15*24]
train_targets = targets[:-15*24]

# 记录处理完的特征与data，作为csv文件输出
write = pd.DataFrame(data=train_targets)
write.to_csv('./train_targets.csv',encoding='gbk')
write = pd.DataFrame(data=train_features)
write.to_csv('./train_features.csv',encoding='gbk')
write = pd.DataFrame(data=test_data)
write.to_csv('./test_features.csv',encoding='gbk')
print('特征处理完成')

# 对网络进行训练
print('训练网络')
# 设置网络参数
iterations = 5000
learning_rate = 0.25
hidden_nodes = 12
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    batch = np.random.choice(train_features.index, size=int(train_features.shape[0]/10))
    X = train_features.iloc[batch].values
    y = train_targets.iloc[batch]['count']

    network.train(X, y)

    # 实时输出loss并记录数据以便后续作图
    train_loss = MSE(network.run(train_features).T, train_targets['count'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['count'].values)

    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

# 作loss曲线
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.show()

# 作验证集上的预测结果与实际数据对比曲线
ig, ax = plt.subplots(figsize=(8,4))
mean, std = scaled_feature['count']
predictions = network.run(val_features).T*std + mean
predictions = np.exp(predictions) - 1

ax.plot(predictions[0], label='Prediction')
ax.plot((np.exp(val_targets['count']*std + mean)-1).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(train.loc[test_data.index]['datetime'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

plt.show()

# 使用训练好的网络预测结果
print('预测结果')
fig, ax = plt.subplots(figsize=(8,4))
test_features = test_data
mean, std = scaled_feature['count']
test_targets = network.run(test_features).T * std + mean
test_targets = test_targets * std + mean
test_targets = np.exp(test_targets) - 1
test_targets = test_targets.astype(np.int16)    #取整输出
print(test_targets)

# 将预测结果写为csv文件
print('csv文件输出')
with  open('./prediction.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(test_targets)

print('end')
