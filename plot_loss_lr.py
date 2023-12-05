# coding:utf-8
'''
**************************************************
@File   ：LSTM -> loss_bs
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , Plot loss-lr
@Date   ：2023/12/4 13:52
**************************************************
'''

import matplotlib.pyplot as plt
import numpy as np

# plot figure
plt.figure(figsize=(15, 25))
fig, splot = plt.subplots(1)
xmax = 0  # max epoch number among all the records
# load the loss records from npy files and plot

train = np.load("loss_record/lr_1e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_1e-06_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g', label="1e-06 train", c='b')
splot.plot(domain, test, 'g', label="1e-06 test", c='b')

train = np.load("loss_record/lr_2e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_2e-06_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':b', label="2e-06 train", c='y')
splot.plot(domain, test, 'b', label="2e-06 test", c='y')

# train = np.load("loss_record/lr_4e-06_bs_8_AdamW_train.npy")
# test = np.load("loss_record/lr_4e-06_bs_8_AdamW_test.npy")
# train = np.delete(train, np.where(train == 0))
# test = np.delete(test, np.where(test == 0))
# max_epochs = len(train)
# xmax = max(xmax, max_epochs)
# domain = np.arange(max_epochs)
# splot.plot(domain, train, ':b', label="4e-06 train", c='g')
# splot.plot(domain, test, 'b', label="4e-06 test", c='g')

train = np.load("loss_record/lr_5e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':c', label="5e-06 train", c='m')
splot.plot(domain, test, 'c', label="5e-06 test", c='m')

train = np.load("loss_record/lr_8e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_8e-06_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':y', label="8e-06 train", c='c')
splot.plot(domain, test, 'y', label="8e-06 test", c='c')

train = np.load("loss_record/lr_1e-05_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_1e-05_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="1e-05 train", c='r')
splot.plot(domain, test, 'r', label="1e-05 test", c='r')

# set figure limits etc
plt.xlim([0, xmax - 1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
# plt.ylim([0, 19.5])#doesn't work with log scale
# plt.yscale("log")
splot.legend()
splot.title.set_text("Loss over time with different learning rate")
plt.xlabel("epoch")
plt.ylabel("loss")

# save figure
plt.savefig("figure/loss_lr.png")

# show figure
plt.show()