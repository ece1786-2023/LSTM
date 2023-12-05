# coding:utf-8
'''
**************************************************
@File   ：LSTM -> plot_loss_opt
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , Plot loss_opt
@Date   ：2023/12/4 13:54
**************************************************
'''
import matplotlib.pyplot as plt
import numpy as np

# plot figure
plt.figure(figsize=(15, 25))
fig, splot = plt.subplots(1)
xmax = 0  # max epoch number among all the records
# load the loss records from npy files and plot

train = np.load("loss_record/lr_5e-06_bs_16_AdamW_train.npy")
test = np.load("loss_record/lr_5e-06_bs_16_AdamW_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g', label="AdamW train", c='b')
splot.plot(domain, test, 'g', label="AdamW test", c='b')

train = np.load("loss_record/lr_5e-06_bs_16_Adagrad_train.npy")
test = np.load("loss_record/lr_5e-06_bs_16_Adagrad_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g', label="Adagrad train", c='y')
splot.plot(domain, test, 'g', label="Adagrad test", c='y')

train = np.load("loss_record/lr_5e-06_bs_16_SGD_train.npy")
test = np.load("loss_record/lr_5e-06_bs_16_SGD_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g', label="SGD train", c='g')
splot.plot(domain, test, 'g', label="SGD test", c='g')

train = np.load("loss_record/lr_5e-06_bs_16_ASGD_train.npy")
test = np.load("loss_record/lr_5e-06_bs_16_ASGD_test.npy")
train = np.delete(train, np.where(train == 0))
test = np.delete(test, np.where(test == 0))
max_epochs = len(train)
xmax = max(xmax, max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g', label="ASGD train", c='r')
splot.plot(domain, test, 'g', label="ASGD test", c='r')

# set figure limits etc
plt.xlim([0, xmax - 1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
# plt.ylim([0, 19.5])#doesn't work with log scale
# plt.yscale("log")
splot.legend(prop={"size": 8}, loc="right")
splot.title.set_text("Loss over time with different optimizer")
plt.xlabel("epoch")
plt.ylabel("loss")

# save figure
plt.savefig("figure/loss_opt.png")

# show figure
plt.show()
