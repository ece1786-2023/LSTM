# coding:utf-8
'''
**************************************************
@File   ：LSTM -> loss_bs
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , Plot loss_bs
@Date   ：2023/12/4 13:47
**************************************************
'''
import matplotlib.pyplot as plt
import numpy as np

# plot figure
fig, splot = plt.subplots(1)
xmax = 0  # max epoch number among all the records
# load the loss records from npy files and plot
train = np.load("loss_record/lr_5e-06_bs_2_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_2_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="batchsize=2 train", c='r')
splot.plot(domain, test, 'r', label="batchsize=2 test", c='r')

train = np.load("loss_record/lr_5e-06_bs_4_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_4_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="batchsize=4 train", c='b')
splot.plot(domain, test, 'r', label="batchsize=4 test", c='b')

train = np.load("loss_record/lr_5e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_8_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="batchsize=8 train", c='g')
splot.plot(domain, test, 'r', label="batchsize=8 test", c='g')

train = np.load("loss_record/lr_5e-06_bs_16_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_16_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="batchsize=16 train", c='y')
splot.plot(domain, test, 'r', label="batchsize=16 test", c='y')

train = np.load("loss_record/lr_5e-06_bs_32_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_32_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':r', label="batchsize=32 train", c='m')
splot.plot(domain, test, 'r', label="batchsize=32 test", c='m')

# set figure limits etc
plt.xlim([0, xmax - 1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
# plt.ylim([0, 19.5])#doesn't work with log scale
# plt.yscale("log")
splot.legend()
splot.title.set_text("Loss over time with different batchsize")
plt.xlabel("epoch")
plt.ylabel("loss")

# save figure
plt.savefig("figure/loss_batchsize.png")

# show figure
plt.show()
