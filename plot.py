import matplotlib.pyplot as plt
import numpy as np


# load the loss records from npy files
loss_ot_train = np.load("loss_record/lr_5e-06_bs_8_AdamW_train.npy")
loss_ot_test = np.load("loss_record/lr_5e-06_bs_8_AdamW_test.npy")


# compute figure dimensions
max_epochs=len(loss_ot_train)
domain = np.arange(max_epochs)


# plot figure
fig, splot = plt.subplots(1)
splot.plot(domain, loss_ot_train, 'g', label="train")
splot.plot(domain, loss_ot_test, 'r', label="test")

plt.xlim([0, max_epochs - 1])
# plt.ylim([0, np.ceil(loss_ot_train[0])])
# plt.ylim([0, 3])#doesn't work with log scale
plt.yscale("log")
splot.legend()
splot.title.set_text("Loss over time")
plt.xlabel("epoch")
plt.ylabel("loss")

# save figure
plt.savefig("figure/loss overtime.png")

# show figure
plt.show()
