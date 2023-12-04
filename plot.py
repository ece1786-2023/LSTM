import matplotlib.pyplot as plt
import numpy as np


# plot figure
fig, splot = plt.subplots(1)
xmax=0 #max epoch number among all the records

# load the loss records from npy files and plot
train = np.load("loss_record/lr_1e-05_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_1e-05_bs_8_AdamW_test.npy")
max_epochs=len(train)
xmax=max(xmax,max_epochs)
domain = np.arange(max_epochs)
print(max_epochs)
splot.plot(domain, train, ':r',label="1e-5 train")
splot.plot(domain, test, 'r', label="1e-5 test")

train = np.load("loss_record/lr_2e-05_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_2e-05_bs_8_AdamW_test.npy")
max_epochs=len(train)
xmax=max(xmax,max_epochs)
print(max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':g',label="2e-5 train")
splot.plot(domain, test, 'g', label="2e-5 test")

train = np.load("loss_record/lr_5e-05_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_5e-05_bs_8_AdamW_test.npy")
max_epochs=len(train)
xmax=max(xmax,max_epochs)
print(max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':b',label="5e-5 train")
splot.plot(domain, test, 'b', label="5e-5 test")

train = np.load("loss_record/lr_3e-05_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_3e-05_bs_8_AdamW_test.npy")
max_epochs=len(train)
xmax=max(xmax,max_epochs)
print(max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':c',label="3e-5 train")
splot.plot(domain, test, 'c', label="3e-5 test")

train = np.load("loss_record/lr_7e-06_bs_8_AdamW_train.npy")
test = np.load("loss_record/lr_7e-06_bs_8_AdamW_test.npy")
max_epochs=len(train)
xmax=max(xmax,max_epochs)
print(max_epochs)
domain = np.arange(max_epochs)
splot.plot(domain, train, ':y',label="7e-6 train")
splot.plot(domain, test, 'y', label="7e-6 test")


# set figure limits etc
plt.xlim([0, xmax - 1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
#plt.ylim([0, 19.5])#doesn't work with log scale
plt.yscale("log")
splot.legend()
splot.title.set_text("Loss over time")
plt.xlabel("epoch")
plt.ylabel("loss")

# save figure
plt.savefig("figure/loss overtime3.png")

# show figure
plt.show()
